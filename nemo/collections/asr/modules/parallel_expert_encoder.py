# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel Expert Speech Encoder (PEE-v2).

Hosts the three PEE-v2 experts -- a multilingual-ASR MoE **speech** expert
(``d_model`` 2048), a Sortformer **speaker** expert (1024), and a **sound** expert
(2048) -- inside one :class:`GGEMMTransformerEncoder`, so a single
``forward_packed`` call runs all three: one batched SDPA over 40 packed heads
(16 + 8 + 16, all ``head_dim`` 128) plus one grouped GEMM per FFN bucket.

The speaker expert's states go through the Sortformer head
(``encoder_proj`` -> ``forward_speaker_sigmoids``) to produce per-frame speaker
activities, which are fused into the speech states (LayerNorm + sinusoidal
speaker kernel + ADD).

The sound expert is merged in per ``merge_sound_expert_to_asr``:

* ``False`` (default since v2.1.0) -- the SoundToken route. The sound expert's CTC
  head reads per-frame ``<ev:...>`` event and ``<sty:stt|end:...>`` style-span
  probabilities out of its states, and those are thresholded, LayerNorm-ed and
  injected through sinusoidal kernels: the direct analogue of the speaker branch, on
  disjoint sets of sinusoid rows. What reaches the ASR states is only the tags, a
  signal of rank <= ``n_sound_events + n_sound_styles``.
* ``True`` -- the whole sound representation instead: its encoder states are
  LayerNorm-ed, scaled and added onto the ASR states. This was the v2.0.0 default,
  used while the CTC head was still being trained.

Speakers, events and styles are three separate families: each has its own LayerNorm over
its own tags, its own block of sinusoid rows and its own scale. A single norm spanning
two families would make each one's code shift with the other's activity, and the sound
expert is much weaker on styles than on events, so the scales need to move apart.

Every family's rows are strided (see ``_TAG_ROW_STRIDE``) and its kernel calibrated so a
single active tag injects a vector of norm ``sqrt(d_model)``. That makes each ``*_scale``
a plain fraction of the ASR state magnitude, unaffected by how many tags the family holds
or which rows it was given -- so widening a family, or moving it, no longer silently
changes how loudly it speaks.

Order matters: sound joins the ASR states FIRST, so speech + sound together form
the backbone that ``asr_norm`` normalizes, and the speaker kernel is then added on
top of that normalized sum.

Only the **speaker** expert is frozen by default. Its kernel is built from a hard
threshold on the speaker activities, so no gradient reaches it through the fusion
regardless; speech and sound both train.

Two encoding modes, same fusion:

* :meth:`ParallelExpertEncoder._forward` -- one pass over the whole utterance.
  All three experts share ``T``, so there is no prefix and no padding.
* :meth:`ParallelExpertEncoder._forward_online` -- windowed long-form decoding
  where the speaker expert additionally attends over its streaming cache. The
  cache is passed as a ``prefix`` to ``forward_packed``, which right-pads the
  speech and sound experts to the speaker's longer ``T`` and masks each expert
  with its own length.

I/O matches :class:`ConformerEncoder` (drop-in). Expects un-normalised mels and
re-applies ``per_feature`` normalization internally. Because all three experts now
share one packed call over one input tensor, that normalization is computed once
and every expert sees the same normalised features. Only self-contained PE bundles
(inline ``speech_expert_cfg`` / ``speaker_expert_cfg`` / ``sound_expert_cfg`` /
``sortformer_modules_cfg`` in ``model_config.yaml``) are supported.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import math
import os
import re
import shutil
import tarfile
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from nemo.collections.asr.modules.ggemm_transformer_encoder import GGEMMTransformerEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    'ParallelExpertEncoder',
    'ParallelExpertEncoderPT',
]

# Container roles, in the order they are packed into the attention group.
EXPERT_ROLES = ('speech', 'speaker', 'sound')
# role -> decoder family recorded on the container (see PEE_EXPERT_TASKS).
EXPERT_TASKS = {'speech': 'asr_tdt', 'speaker': 'diarization', 'sound': 'sound_rnnt'}

# The two tagged families the sound expert's CTC head contributes to the ASR states:
# event tags such as `<ev:laughter>` (10 in the v2.1.0 checkpoint) and style-span
# delimiters such as `<sty:stt:anger_contempt>` / `<sty:end:anger_contempt>` (22 = 11
# styles x start/end). These prefixes are the ones the sound checkpoint itself declares
# in `token_weighting.{event,style}_prefixes`, which is where to look first if a
# retrained expert stops matching.
#
# `stt`/`end` are kept as 22 INDEPENDENT point tags rather than being folded into 11
# "style is active over this span" signals. Reconstructing the span would need a state
# machine carried across window boundaries in `_forward_windowed`, and any disagreement
# between that carry-over and the offline path shows up as a train/inference mismatch.
# Emitting the raw delimiters keeps the fusion stateless and lets the decoder's attention
# tie a start to its end, which is the kind of long-range binding attention is good at.
#
# `<spk:0>` / `<spk:1>` are deliberately left out: PEE gets speakers from the Sortformer
# expert, which owns sinusoid rows 0..n_spk-1, and injecting the sound expert's weaker
# 2-speaker guesses would put a second opinion on the same question into the states.
_SOUND_EVENT_TOKEN_PREFIX = '<ev:'
_SOUND_EVENT_TOKEN_RE = re.compile(r'^<ev:[^>]+>$')
_SOUND_STYLE_TOKEN_PREFIX = '<sty:'
_SOUND_STYLE_TOKEN_RE = re.compile(r'^<sty:(?:stt|end):[^>]+>$')

# Where each tag family's sinusoid rows live, and how far apart consecutive rows sit.
#
# Rows are STRIDED rather than consecutive, and each family gets its own far-apart block,
# because these rows are identity codes and not positions. Adjacent rows of the sinusoid
# table sit at ~0.97 cosine -- the very property that makes the table good at encoding
# position makes it bad at encoding identity.
#
# Two things go wrong with consecutive rows, both measured on the per-tag injection
# vectors rather than the raw rows:
#
#   * Tags become hard to tell apart. Worst within-family cosine at stride 1 is 0.858
#     (speakers, n_spk=8) and 0.927 (a 32-tag sound family); at stride 16 it is 0.163 and
#     0.231. Across families, the worst event-vs-speaker pair goes from 0.485 to 0.006.
#   * Tags become unequally loud. A tag's code is proportional to ``row_i - mean(rows)``,
#     so with overlapping rows the slots in the MIDDLE of a block sit near the mean and
#     inject faintly while the edges inject strongly -- a 1.64x spread across 8 speakers,
#     and 2.22x across 4. Once the rows are near-orthogonal, ``|row_i - mean|`` is the
#     same for every i and the spread falls to 1.12x.
#
# The blocks are spaced far enough apart to stay disjoint as families grow: speakers can
# reach n_spk=32 before running into the event block.
_TAG_ROW_STRIDE = 16
_SPEAKER_ROW_OFFSET = 0
_SOUND_EVENT_ROW_OFFSET = 512
_SOUND_STYLE_ROW_OFFSET = 1024

# The v2.0.0 speaker layout: consecutive rows 0..n_spk-1, no calibration, scale 1.0. The
# published bundle's speech expert was trained against exactly this, so it stays the
# default and such a bundle reloads bit-identically. New bundles opt in to the calibrated,
# strided layout by writing spk_kernel_row_stride and spk_kernel_calibrate.
#
# Calibration is tied to the layout rather than defaulted on because the two are only
# ACCIDENTALLY close: at n_spk=8, d_model=2048 the calibration factor happens to be
# 1.33325, so the new 0.75 default reproduces the old 1.0 to 6e-5 -- near enough to look
# interchangeable, not near enough to be (it moves 16% of elements by one bf16 ulp), and
# the near-miss does not hold at any other n_spk.
_LEGACY_SPEAKER_ROW_STRIDE = 1
_LEGACY_SPK_KERNEL_SCALE = 1.0
_CALIBRATED_KERNEL_SCALE = 0.75


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default float dtype.

    Makes ``SortformerModules.init_streaming_state`` allocate its dtype-less
    speaker-cache / FIFO buffers in the speaker expert's dtype, avoiding fp32/bf16
    mismatch.
    """
    prev = torch.get_default_dtype()
    if dtype == prev or not dtype.is_floating_point:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextlib.contextmanager
def _disable_dist_feature_sync():
    """Temporarily make ``torch.distributed`` look uninitialized.

    Skips cross-rank ``all_reduce``s in the Sortformer streaming helpers, which are
    unnecessary and unsafe for single-recording inference (e.g. a vLLM worker).
    The original ``dist.is_initialized`` is always restored.
    """
    if not (hasattr(dist, "is_initialized") and dist.is_initialized()):
        yield
        return
    orig_is_initialized = dist.is_initialized
    dist.is_initialized = lambda: False
    try:
        yield
    finally:
        dist.is_initialized = orig_is_initialized


def _disable_max_seq_length_sync(module: nn.Module) -> None:
    """Turn off ``sync_max_audio_length`` on every encoder under ``module``.

    That flag makes ``update_max_seq_length`` issue an ``all_reduce`` on the
    **default** process group. Any such collective inside a data-dependent branch lets
    ranks emit different numbers of default-PG collectives on the same step, which
    NCCL cannot match positionally -- the run then deadlocks until the watchdog
    aborts it.

    Dropping it is numerically neutral: the reduced value only sizes the
    positional-encoding buffer (grown on demand per rank, and sliced
    length-relative), while attention masks are always built from the local
    sequence length. Applies to the flex ``TransformerEncoder`` family the same way
    it did to the Conformer branches -- both expose ``sync_max_audio_length`` and
    both all-reduce inside ``update_max_seq_length``.
    """
    for submodule in module.modules():
        if getattr(submodule, "sync_max_audio_length", False):
            submodule.sync_max_audio_length = False


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a ``DictConfig`` without resolving interpolations.

    ``from_config_dict`` mutates its input in place, so sub-target builders get a copy.
    """
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _unwrap_cls(cls):
    """Step through a ``wrapt`` proxy chain to the underlying class.

    ``inspect.unwrap`` is not enough: NeMo's ``@experimental`` wraps the *class* in a
    ``wrapt`` proxy that forwards ``__repr__``/``__class__``, so the object
    ``inspect.unwrap`` returns still resolves ``__init__`` to the proxy's
    ``(*args, **kwargs)``.
    """
    seen = {id(cls)}
    cur = cls
    for _ in range(8):
        nxt = getattr(cur, "__wrapped__", None)
        if nxt is None or id(nxt) in seen:
            return cur
        seen.add(id(nxt))
        cur = nxt
    return cur


def _resolve_target(target: str) -> type:
    """Resolve a Hydra ``_target_`` string to a class."""
    module_path, _, cls_name = target.rpartition(".")
    try:
        return getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import _target_ '{target}'.") from exc


def _init_param_names(cls) -> set:
    """Names ``cls.__init__`` accepts.

    Refuses to fall back to a ``**kwargs`` signature: filtering config keys against
    one would drop *every* key and silently build a default-shaped encoder.
    """
    for cand in (cls, _unwrap_cls(cls)):
        try:
            params = inspect.signature(cand.__init__).parameters
        except (TypeError, ValueError):
            continue
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return set(params) - {"self"}
    raise TypeError(
        f"Could not introspect a concrete __init__ signature for {cls!r}; refusing to "
        "filter config keys against a **kwargs signature (would build a default module)."
    )


def _build_from_cfg(cfg: DictConfig, what: str) -> nn.Module:
    """Instantiate a module from a config section carrying ``_target_``.

    Keys the installed class does not accept are reported rather than silently
    ignored -- tuning-only extras (e.g. ``flex_kernel_options``) are harmless, but
    anything architectural showing up here means a branch mismatch.
    """
    if cfg is None or '_target_' not in cfg:
        raise ValueError(f"{what} config must be a mapping carrying a `_target_` key.")
    cls = _resolve_target(str(cfg['_target_']))
    raw = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k != '_target_'}
    accepted = _init_param_names(cls)
    kwargs = {k: v for k, v in raw.items() if k in accepted}
    dropped = sorted(set(raw) - accepted)
    if dropped:
        logging.warning(
            "[ParallelExpertEncoder] %s: ignoring config keys not accepted by %s: %s",
            what,
            getattr(_unwrap_cls(cls), '__name__', cls),
            dropped,
        )
    return cls(**kwargs)


@experimental
class ParallelExpertEncoderPT(ModelPT):
    """ModelPT shell so a :class:`ParallelExpertEncoder` can be saved/restored as a
    ``.nemo`` archive (inline ``speech_expert_cfg`` / ``speaker_expert_cfg`` /
    ``sound_expert_cfg`` / ``sortformer_modules_cfg``).
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = ParallelExpertEncoder(
            speech_expert_cfg=self._cfg.get('speech_expert_cfg', None),
            speaker_expert_cfg=self._cfg.get('speaker_expert_cfg', None),
            sound_expert_cfg=self._cfg.get('sound_expert_cfg', None),
            sortformer_modules_cfg=self._cfg.get('sortformer_modules_cfg', None),
            sound_ctc_head_cfg=self._cfg.get('sound_ctc_head_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', 'per_feature'),
            freeze_speaker=self._cfg.get('freeze_speaker', True),
            freeze_speech=self._cfg.get('freeze_speech', False),
            freeze_sound=self._cfg.get('freeze_sound', False),
            online_inference_length=self._cfg.get('online_inference_length', 375),
            chunk_left_context=self._cfg.get('chunk_left_context', 50),
            chunk_right_context=self._cfg.get('chunk_right_context', 50),
            diar_fifo_len=self._cfg.get('diar_fifo_len', 0),
            diar_spkcache_update_period=self._cfg.get('diar_spkcache_update_period', 375),
            diar_spkcache_len=self._cfg.get('diar_spkcache_len', 200),
            missing_rttm_target=self._cfg.get('missing_rttm_target', -1.0),
            speaker_activity_threshold=self._cfg.get('speaker_activity_threshold', 0.5),
            spk_kernel_scale=self._cfg.get('spk_kernel_scale', None),
            spk_kernel_row_stride=self._cfg.get('spk_kernel_row_stride', _LEGACY_SPEAKER_ROW_STRIDE),
            spk_kernel_calibrate=self._cfg.get('spk_kernel_calibrate', False),
            sync_max_audio_length=self._cfg.get('sync_max_audio_length', False),
            always_run_diarization=self._cfg.get('always_run_diarization', True),
            moe_mode=self._cfg.get('moe_mode', 'dense'),
            fused_forward_in_training=self._cfg.get('fused_forward_in_training', False),
            ggemm_backend=self._cfg.get('ggemm_backend', 'baddbmm'),
            online_prefix_mode=self._cfg.get('online_prefix_mode', 'replace'),
            merge_sound_expert_to_asr=self._cfg.get('merge_sound_expert_to_asr', False),
            sound_merge_scale=self._cfg.get('sound_merge_scale', 0.3),
            sound_event_threshold=self._cfg.get('sound_event_threshold', 0.5),
            sound_kernel_scale=self._cfg.get('sound_kernel_scale', 0.75),
            inject_sound_styles=self._cfg.get('inject_sound_styles', True),
            sound_style_scale=self._cfg.get('sound_style_scale', 0.75),
        )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        pass

    @staticmethod
    def is_pe_nemo(nemo_path: str) -> bool:
        """Detect whether a ``.nemo`` archive is a :class:`ParallelExpertEncoderPT` bundle.

        Reads only ``model_config.yaml`` and checks its ``target:``.

        Args:
            nemo_path (str): Path to a ``.nemo`` archive.

        Returns:
            ``True`` if ``target`` ends with ``ParallelExpertEncoderPT``, else ``False``.
        """
        if not (isinstance(nemo_path, str) and nemo_path.endswith('.nemo') and os.path.isfile(nemo_path)):
            return False
        try:
            with tarfile.open(nemo_path, mode='r') as tf:
                for member in tf.getmembers():
                    if os.path.basename(member.name) == 'model_config.yaml':
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            return False
                        cfg = OmegaConf.create(fobj.read().decode('utf-8'))
                        return str(cfg.get('target', '')).endswith('ParallelExpertEncoderPT')
        except (tarfile.TarError, OSError) as exc:
            logging.warning("[ParallelExpertEncoder] Could not inspect %s: %s", nemo_path, exc)
            return False
        return False

    @classmethod
    def load_from_nemo(
        cls,
        model_path_or_name: str,
        *,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True,
    ) -> ParallelExpertEncoder:
        """Load a self-contained PE bundle and return its inner encoder.

        Follows the standard NeMo :class:`~nemo.core.classes.common.Model`
        convention for resolving a checkpoint reference:

        * a local ``.nemo`` file is read directly by
          :meth:`_load_encoder_from_archive` -- ``ModelPT.restore_from`` cannot
          instantiate an ``@experimental`` target (see that method's docstring);
        * otherwise ``model_path_or_name`` is treated as a pretrained model
          identifier -- a HuggingFace Hub repo id (``{repo}/{name}``) or an NGC
          alias -- and resolved with :meth:`Model.from_pretrained`, which
          downloads/caches the ``.nemo`` (honouring the HuggingFace cache and
          ``HF_HUB_OFFLINE``, so a prefetched cache works on offline nodes).

        Args:
            model_path_or_name (str): Local ``.nemo`` path or pretrained model id.
            map_location (str | torch.device): Device to map weights onto.
            strict (bool): Enforce exact state-dict match.

        Returns:
            The restored :class:`ParallelExpertEncoder`.
        """
        if (
            isinstance(model_path_or_name, str)
            and model_path_or_name.endswith('.nemo')
            and os.path.isfile(model_path_or_name)
        ):
            return cls._load_encoder_from_archive(model_path_or_name, map_location=map_location, strict=strict)
        bundle = cls.from_pretrained(
            model_name=model_path_or_name,
            map_location=map_location,
            strict=strict,
        )
        return bundle.encoder

    @classmethod
    def _load_encoder_from_archive(
        cls,
        nemo_path: str,
        *,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True,
    ) -> ParallelExpertEncoder:
        """Build the encoder straight from a ``.nemo`` archive, bypassing ``restore_from``.

        ``ModelPT.restore_from`` routes the bundle's ``target:`` through NeMo's
        config-instantiation allow-list, which rejects any ``@experimental``-decorated
        class: the decorator wraps the class in a ``wrapt`` proxy, ``issubclass()``
        raises ``TypeError`` on it, and the allow-list turns that into "unsafe target".
        Both :class:`ParallelExpertEncoderPT` and the flex ``TransformerEncoder`` the
        experts are built from are decorated, so a PE bundle can never be restored the
        normal way. Fixing that belongs in ``nemo/core/classes/common.py``, which is
        deliberately out of scope here.

        A PE bundle is only ``model_config.yaml`` + ``model_weights.ckpt``, and the
        encoder is fully described by the inline expert configs, so reading those two
        directly is both sufficient and strictly less machinery than ``restore_from``.
        The archive's ``target:`` is still verified first, so this does not widen what
        gets instantiated.

        Args:
            nemo_path (str): Local ``.nemo`` bundle.
            map_location: Device to map weights onto.
            strict (bool): Enforce exact state-dict match.

        Returns:
            The restored :class:`ParallelExpertEncoder`.
        """
        import tempfile

        if not cls.is_pe_nemo(nemo_path):
            raise ValueError(f"{nemo_path!r} is not a ParallelExpertEncoderPT .nemo bundle.")

        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(nemo_path, mode='r') as tf:
                members = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}
                for name in ('model_config.yaml', 'model_weights.ckpt'):
                    if name not in members:
                        raise RuntimeError(f"{nemo_path} is missing '{name}'.")
                    member = members[name]
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        raise RuntimeError(f"Could not read '{name}' from {nemo_path}.")
                    with open(os.path.join(td, name), 'wb') as out:
                        shutil.copyfileobj(fobj, out)

            cfg = OmegaConf.load(os.path.join(td, 'model_config.yaml'))
            shell = cls(cfg=cfg, trainer=None)
            state = torch.load(
                os.path.join(td, 'model_weights.ckpt'),
                map_location=map_location,
                weights_only=True,
            )

        # The bundle stores the PT shell's state dict, so the encoder's own tensors
        # sit under an `encoder.` prefix. Anything else belongs to the shell and has
        # no counterpart on the bare encoder.
        prefix = 'encoder.'
        enc_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not enc_state:
            raise RuntimeError(
                f"No '{prefix}*' tensors found in {nemo_path}; the bundle does not look "
                "like a saved ParallelExpertEncoderPT."
            )
        missing, unexpected = shell.encoder.load_state_dict(enc_state, strict=strict)
        if missing or unexpected:
            logging.warning(
                "[ParallelExpertEncoder] load_from_nemo(%s): %d missing / %d unexpected keys.",
                nemo_path,
                len(missing),
                len(unexpected),
            )
        return shell.encoder.to(map_location)

    @classmethod
    def save_to_nemo(
        cls,
        encoder: ParallelExpertEncoder,
        output_nemo_path: str,
        *,
        template_bundle_path: str,
    ) -> None:
        """Save ``encoder`` as a self-contained PE ``.nemo``, reusing ``model_config.yaml``
        from ``template_bundle_path``.

        The template must describe the same architecture (speech ``d_model``,
        ``num_spks``); mismatches raise :class:`ValueError` fail-fast.

        Args:
            encoder (ParallelExpertEncoder): The encoder whose weights are persisted.
            output_nemo_path (str): Destination ``.nemo`` path.
            template_bundle_path (str): Existing PE ``.nemo`` whose ``model_config.yaml`` is reused.
        """
        if not isinstance(encoder, ParallelExpertEncoder):
            raise TypeError(f"save_to_nemo expects a ParallelExpertEncoder, got {type(encoder).__name__}")
        if not os.path.isfile(template_bundle_path):
            raise FileNotFoundError(f"template_bundle_path does not exist: {template_bundle_path}")

        template_cfg: Optional[DictConfig] = None
        with tarfile.open(template_bundle_path, mode='r') as tf:
            for member in tf.getmembers():
                if os.path.basename(member.name) == 'model_config.yaml':
                    fobj = tf.extractfile(member)
                    if fobj is not None:
                        template_cfg = OmegaConf.create(fobj.read().decode('utf-8'))
                    break
        if template_cfg is None:
            raise RuntimeError(f"Could not read 'model_config.yaml' from template bundle: {template_bundle_path}")

        missing = [
            key
            for key in ('speech_expert_cfg', 'speaker_expert_cfg', 'sound_expert_cfg', 'sortformer_modules_cfg')
            if template_cfg.get(key, None) in (None, {}, '')
        ]
        if missing:
            raise ValueError(
                f"Template bundle {template_bundle_path} is not self-contained (missing {missing}); "
                "it cannot be used as a save template."
            )

        # Only required on the SoundToken path, so it is not in `missing` above -- but
        # without it the saved bundle would carry sound_ctc_head weights that its own
        # config never builds, and fail strict reload.
        if not encoder.merge_sound_expert_to_asr and template_cfg.get('sound_ctc_head_cfg', None) in (None, {}, ''):
            raise ValueError(
                f"Encoder uses merge_sound_expert_to_asr=False (SoundToken injection) but template "
                f"bundle {template_bundle_path} has no sound_ctc_head_cfg; the saved bundle would "
                "not reload. Use a template built by build_pee_v2_bundle.py at v2.1.0 or later."
            )

        tmpl_d_model = int(template_cfg.speech_expert_cfg.get('d_model', -1))
        tmpl_n_spk = int(template_cfg.sortformer_modules_cfg.get('num_spks', -1))
        enc_d_model = int(encoder.d_model)
        enc_n_spk = int(encoder.n_spk)
        if tmpl_d_model != enc_d_model:
            raise ValueError(
                f"Template speech_expert_cfg.d_model={tmpl_d_model} does not match "
                f"encoder.d_model={enc_d_model}; the saved bundle would fail strict reload."
            )
        if tmpl_n_spk != enc_n_spk:
            raise ValueError(
                f"Template sortformer_modules_cfg.num_spks={tmpl_n_spk} does not match "
                f"encoder.n_spk={enc_n_spk}; the saved bundle would fail strict reload."
            )

        # Fresh PT shell from the template cfg to reuse NeMo's save_to; swap in encoder.
        shell = cls(cfg=template_cfg, trainer=None)
        shell.encoder = encoder
        # Pin `_cfg` to the verbatim template so save_to round-trips it exactly.
        shell._cfg = template_cfg

        shell.save_to(output_nemo_path)
        logging.info(
            "[ParallelExpertEncoder] Saved PE bundle to %s using template config from %s",
            output_nemo_path,
            template_bundle_path,
        )


@experimental
class ParallelExpertEncoder(nn.Module):
    """PEE-v2 three-expert encoder; I/O identical to :class:`ConformerEncoder`.

    Reconstructed from inline configs in the PE bundle's ``model_config.yaml``.

    Args:
        speech_expert_cfg (DictConfig): Inline config for the speech MoE expert
            (``MoETransformerEncoder``, ``d_model`` 2048). Its output is the
            backbone that speaker activities are fused into.
        speaker_expert_cfg (DictConfig): Inline config for the Sortformer speaker
            expert (``TransformerEncoder``, ``d_model`` 1024).
        sound_expert_cfg (DictConfig): Inline config for the sound expert
            (``TransformerEncoder``, ``d_model`` 2048). Runs in the packed group;
            its output is returned unfused.
        sortformer_modules_cfg (DictConfig): Inline config for
            :class:`SortformerModules`, which supplies the speaker head
            (``encoder_proj`` + ``forward_speaker_sigmoids``) and the streaming
            speaker-cache logic. ``tf_d_model`` must match the checkpoint (192 on
            the v2 speaker), not be assumed equal to ``fc_d_model``.
        sound_ctc_head_cfg (DictConfig, optional): Inline config for the sound expert's
            CTC head (a :class:`ConvASRDecoder`), lifted from the sound checkpoint's
            ``decoder:`` block. Like the Sortformer head this is NOT part of the
            expert's encoder, so it is built here and loaded separately. Its
            ``vocabulary`` is what the ``<ev:...>`` column indices are read from.
            Required when ``merge_sound_expert_to_asr=False``, unused otherwise.
        asr_normalize_type (str, optional): Normalization applied to the shared mel
            input. Defaults to ``per_feature``, which every expert then sees --
            there is one packed call over one input tensor, so the three experts
            cannot be normalised differently. Pass ``None`` to feed raw mels.
        freeze_speaker (bool): Freeze the speaker expert + head. Defaults to ``True``.
            The speaker kernel is built from a hard threshold on the speaker
            activities, so no gradient reaches this branch through the fusion
            anyway -- freezing makes that explicit and saves the optimizer state.
        freeze_speech (bool): Freeze the speech expert. Defaults to ``False``.
        freeze_sound (bool): Freeze the sound expert. Defaults to ``False`` -- the
            sound expert trains alongside speech, and its states reach the loss
            through the merge. Only the speaker expert (the Sortformer diarizer,
            whose head is non-differentiable past the activity threshold) is frozen
            by default.
        online_inference_length (int): Generation-time window in encoder output
            frames (default ``375`` = 30 s at subsampling 8); ``<= 0`` disables
            windowing. Unused by training and validation, which encode in one pass.
        chunk_left_context (int): Left context (output frames) per window. Default ``50``.
        chunk_right_context (int): Right context (output frames) per window. Default ``50``.
        diar_fifo_len (int): Sortformer streaming ``fifo_len``. Default ``0``.
        diar_spkcache_update_period (int): Sortformer streaming
            ``spkcache_update_period``. Default ``375``. Values below
            ``chunk_len`` cannot be honoured -- the effective period is
            ``max(period, chunk_len)`` -- so the default matches ``chunk_len``.
        diar_spkcache_len (int): Sortformer streaming ``spkcache_len``. Default ``200``.
        missing_rttm_target (float): Sentinel marking rows that should use diarization
            predictions. Defaults to ``-1.0``.
        speaker_activity_threshold (float): Binarization threshold applied to RTTM and
            diarization targets before speaker-kernel fusion. Defaults to ``0.5``.
        spk_kernel_scale (float, optional): Weight of the speaker-kernel contribution.
            Defaults to ``0.75`` when ``spk_kernel_calibrate`` and ``1.0`` otherwise,
            because the two layouts measure it in different units -- see that argument.
        spk_kernel_row_stride (int): Spacing between the sinusoid rows the speaker kernel
            takes. Defaults to ``1`` -- rows ``0 … n_spk-1``, the layout v2.0.0 was built
            and trained against, so a bundle that does not set this reloads unchanged.
            New bundles set ``16``, which makes the speaker rows near-orthogonal: the
            worst speaker-pair cosine drops from 0.858 to 0.163 and the spread in
            per-speaker injection strength from 1.64x to 1.12x. See ``_TAG_ROW_STRIDE``.
        spk_kernel_calibrate (bool): Rescale the speaker kernel so one active speaker
            injects ``sqrt(d_model)``, making ``spk_kernel_scale`` a *fraction of the ASR
            state magnitude* directly comparable to ``sound_kernel_scale`` and independent
            of ``n_spk``. Uncalibrated, the same scale injects 0.36 of the state magnitude
            at ``n_spk=4`` but 0.75 at ``n_spk=8``. Defaults to ``False`` so that v2.0.0
            bundles, whose speech expert was trained on the raw kernel, reload
            bit-identically; new bundles set it. Changing it on a trained checkpoint moves
            the kernel out from under the weights that learned to read it.
        sync_max_audio_length (bool): Let the experts all-reduce their maximum sequence
            length on the default process group. Defaults to ``False``; leave it off
            unless every rank is guaranteed to run the encoder on every step, since the
            reduction is emitted from inside a data-dependent branch.
        always_run_diarization (bool): Run the speaker head on every single-pass forward
            instead of only when some row requests predicted diarization. Defaults to
            ``True`` so the collective schedule cannot depend on batch content.
        moe_mode (str): ``'dense'`` or ``'topk'`` for the speech MoE inside the grouped
            FFN. Defaults to ``'dense'``. Only reached on the fused path.
        fused_forward_in_training (bool): Use the fused packed path while training too.
            Defaults to ``False``: training runs each expert on its own path, which is
            slower per step but holds far less memory, because fusion has to stack every
            expert's FFN input into one tensor and (under ``moe_mode='dense'``) evaluate
            every MoE expert on every token. Inference is unaffected either way -- it
            always fuses. Set ``True`` only to A/B the two paths, and expect to shrink
            the batch or ``encoder_chunk_size_seconds`` to fit.
        ggemm_backend (str): Grouped-GEMM backend. Defaults to ``'baddbmm'``.
        online_prefix_mode (str): How the speaker's streaming cache is spliced in the
            windowed path. ``'replace'`` (default) walks each window's start back by
            the cache length and lets the cache stand in for the speaker's leading
            frames, so speech and sound spend those slots on real left context instead
            of zero padding -- same FLOPs, and the packed group stays
            FlashAttention-2 eligible. ``'extend'`` is the older behaviour: the cache
            lengthens the speaker and the other experts are zero-padded to match.
            Windows too near the start of a recording to walk back far enough fall
            back to ``'extend'`` automatically.
        merge_sound_expert_to_asr (bool): How the sound expert reaches the ASR states.
            ``False`` (default since v2.1.0) selects the **SoundToken** path: the CTC
            head reads per-frame ``<ev:...>`` and ``<sty:stt|end:...>`` probabilities
            out of the sound states, and those are thresholded at
            ``sound_event_threshold``, LayerNorm-ed per family and injected through
            ``sound_token_kernel`` / ``sound_style_kernel`` -- exactly as the speaker
            sigmoids go through ``diar_kernel``. Requires ``sound_ctc_head_cfg``.
            ``True`` instead adds the sound expert's **encoder states**, LayerNorm-ed
            and scaled by ``sound_merge_scale``; that was the v2.0.0 behaviour and
            needs no CTC head.
        sound_merge_scale (float): Relative weight of the sound stream in the merged
            backbone, mirroring ``spk_kernel_scale``. Both streams are normalized
            first, and they are near-orthogonal in practice (measured cosine
            ~0.004), so a scale of ``s`` gives sound a variance share of roughly
            ``s^2 / (1 + s^2)``: 0.3 -> ~8%, 0.5 -> ~20%, 1.0 -> ~50%.
            Defaults to ``0.3``, a deliberately modest starting weight because the
            sound expert is trained on far less data (~2.6 kh) than the speech
            backbone. This is a starting point, not a ceiling: ``sound_norm`` keeps
            a learnable affine gain and the sound expert itself trains, so the
            model can grow the contribution if the loss rewards it.
            Ignored when ``merge_sound_expert_to_asr=False``.
        sound_event_threshold (float): Probability above which an event or style tag
            counts as present, before the kernel injection. Defaults to ``0.5``,
            mirroring ``speaker_activity_threshold``. Only read on the SoundToken path.
        sound_kernel_scale (float): Weight of the event-kernel contribution, the sound
            twin of ``spk_kernel_scale`` and on the same calibrated footing: a fraction of
            the ASR state magnitude, independent of how many event tags there are.
            ``asr_encoded`` arrives from the speech expert's ``final_norm``, so it too has
            norm ``~sqrt(d_model)``. Defaults to ``0.75``, level with the speaker kernel.
            Only read on the SoundToken path.
        inject_sound_styles (bool): Whether to also inject the 22 ``<sty:stt|end:...>``
            span delimiters as their own tag family. Defaults to ``True``. When
            ``False`` only the event tags are injected and no style kernel is built,
            which is the v2.1.0-rc behaviour. Only read on the SoundToken path.
        sound_style_scale (float): Weight of the style-kernel contribution, on the same
            calibrated footing, so it is directly comparable to ``sound_kernel_scale``
            despite the family being 22 tags rather than 10. Defaults to ``0.75``. Kept a
            separate knob because the sound expert is markedly weaker on styles than on
            events, and this ships at the value that *balances* styles against the states
            rather than one shown to help: ``0.0`` mutes them while leaving the event path
            bit-identical. Only read on the SoundToken path.
    """

    def __init__(
        self,
        speech_expert_cfg: DictConfig,
        speaker_expert_cfg: DictConfig,
        sound_expert_cfg: DictConfig,
        sortformer_modules_cfg: DictConfig,
        sound_ctc_head_cfg: Optional[DictConfig] = None,
        asr_normalize_type: Optional[str] = 'per_feature',
        freeze_speaker: bool = True,
        freeze_speech: bool = False,
        freeze_sound: bool = False,
        online_inference_length: int = 375,
        chunk_left_context: int = 50,
        chunk_right_context: int = 50,
        diar_fifo_len: int = 0,
        diar_spkcache_update_period: int = 375,
        diar_spkcache_len: int = 200,
        missing_rttm_target: float = -1.0,
        speaker_activity_threshold: float = 0.5,
        spk_kernel_scale: Optional[float] = None,
        spk_kernel_row_stride: int = _LEGACY_SPEAKER_ROW_STRIDE,
        spk_kernel_calibrate: bool = False,
        sync_max_audio_length: bool = False,
        always_run_diarization: bool = True,
        moe_mode: str = 'dense',
        fused_forward_in_training: bool = False,
        ggemm_backend: str = 'baddbmm',
        online_prefix_mode: str = 'replace',
        merge_sound_expert_to_asr: bool = False,
        sound_merge_scale: float = 0.3,
        sound_event_threshold: float = 0.5,
        sound_kernel_scale: float = 0.75,
        inject_sound_styles: bool = True,
        sound_style_scale: float = 0.75,
    ):
        super().__init__()

        cfgs = {
            'speech': speech_expert_cfg,
            'speaker': speaker_expert_cfg,
            'sound': sound_expert_cfg,
        }
        absent = [role for role, cfg in cfgs.items() if cfg is None]
        if absent or sortformer_modules_cfg is None:
            raise ValueError(
                "ParallelExpertEncoder requires speech/speaker/sound expert configs and "
                f"sortformer_modules_cfg; missing: {absent + (['sortformer_modules_cfg'] if sortformer_modules_cfg is None else [])}. "
                "Self-contained PE bundles supply these inline in their model_config.yaml."
            )

        experts = {role: _build_from_cfg(_clone_config(cfgs[role]), f"{role}_expert_cfg") for role in EXPERT_ROLES}
        self.pee = GGEMMTransformerEncoder(experts, expert_tasks=dict(EXPERT_TASKS))

        # The Sortformer head. `extract_encoder_state_dict(..., encoder_attr='encoder')`
        # pulls only the speaker *encoder*, so the head (encoder_proj + the sigmoid
        # stack) and the streaming cache logic are built here and loaded separately
        # from the speaker .nemo's `sortformer_modules.*` keys.
        self.sortformer_modules = _build_from_cfg(_clone_config(sortformer_modules_cfg), 'sortformer_modules_cfg')

        speaker_d_model = self.pee.experts['speaker'].d_model
        if int(self.sortformer_modules.fc_d_model) != int(speaker_d_model):
            raise ValueError(
                f"sortformer_modules.fc_d_model={self.sortformer_modules.fc_d_model} must match the "
                f"speaker expert d_model={speaker_d_model}; the head projects the speaker states and "
                "the streaming cache is allocated at fc_d_model."
            )

        self.asr_normalize_type = asr_normalize_type
        self._feat_in = self.pee.experts['speech']._feat_in

        # The experts' `update_max_seq_length` all-reduces on the default process
        # group, and neither the online loop (data-dependent window count) nor a
        # content-gated head is reached uniformly by every rank.
        self.sync_max_audio_length = bool(sync_max_audio_length)
        if not self.sync_max_audio_length:
            _disable_max_seq_length_sync(self.pee)
        self.always_run_diarization = bool(always_run_diarization)

        self.freeze_speaker = freeze_speaker
        self.freeze_speech = freeze_speech
        self.freeze_sound = freeze_sound

        self.moe_mode = moe_mode
        self.fused_forward_in_training = bool(fused_forward_in_training)
        self.ggemm_backend = ggemm_backend
        if online_prefix_mode not in ('replace', 'extend'):
            raise ValueError(f"online_prefix_mode must be 'replace' or 'extend', got {online_prefix_mode!r}.")
        self.online_prefix_mode = online_prefix_mode

        # Long-form / online inference configuration.
        self.online_inference_length = int(online_inference_length)
        # Opt-in, and never on during training or validation: see `online_inference`.
        self.online_inference_enabled = False
        # Overlap-and-trim context (output frames), shared by every expert.
        self.chunk_left_context = max(0, int(chunk_left_context))
        self.chunk_right_context = max(0, int(chunk_right_context))
        # Online-inference window + context in input mel frames (constant per session).
        self.chunk_feat_len = self.online_inference_length * self.subsampling_factor
        self.left_ctx_feat_len = self.chunk_left_context * self.subsampling_factor
        self.right_ctx_feat_len = self.chunk_right_context * self.subsampling_factor
        self.diar_fifo_len = int(diar_fifo_len)
        self.diar_spkcache_update_period = int(diar_spkcache_update_period)
        self.diar_spkcache_len = int(diar_spkcache_len)
        self.missing_rttm_target = float(missing_rttm_target)
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.spk_kernel_calibrate = bool(spk_kernel_calibrate)
        # The default scale follows the layout: calibrated kernels are a fraction of the
        # state magnitude, uncalibrated ones are the raw v2.0.0 kernel at unit weight.
        # These are NOT interchangeable numbers, so neither can serve as both defaults.
        if spk_kernel_scale is None:
            spk_kernel_scale = _CALIBRATED_KERNEL_SCALE if self.spk_kernel_calibrate else _LEGACY_SPK_KERNEL_SCALE
        self.spk_kernel_scale = float(spk_kernel_scale)

        self.n_spk = int(self.sortformer_modules.n_spk)
        # The speech MoE expert is the backbone the speaker kernel is fused into.
        self.asr_d_model = int(self.pee.experts['speech'].d_model)

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.spk_kernel_row_stride = max(1, int(spk_kernel_row_stride))
        spk_last_row = _SPEAKER_ROW_OFFSET + self.spk_kernel_row_stride * (self.n_spk - 1)
        if spk_last_row >= _SOUND_EVENT_ROW_OFFSET:
            raise ValueError(
                f"n_spk={self.n_spk} at spk_kernel_row_stride={self.spk_kernel_row_stride} "
                f"reaches sinusoid row {spk_last_row}, which is inside the sound event "
                f"block at row {_SOUND_EVENT_ROW_OFFSET}. Speakers and sound tags would "
                "share rows and become indistinguishable; lower the stride or move the "
                "sound blocks."
            )
        self.register_buffer(
            "diar_kernel",
            self._build_tag_kernel(
                self.n_spk,
                _SPEAKER_ROW_OFFSET,
                self.asr_d_model,
                stride=self.spk_kernel_row_stride,
                calibrate=self.spk_kernel_calibrate,
            ),
            persistent=False,
        )

        # --- sound expert -> ASR merge -------------------------------------
        self.merge_sound_expert_to_asr = bool(merge_sound_expert_to_asr)
        self.sound_merge_scale = float(sound_merge_scale)
        self.sound_event_threshold = float(sound_event_threshold)
        self.sound_kernel_scale = float(sound_kernel_scale)
        self.inject_sound_styles = bool(inject_sound_styles)
        self.sound_style_scale = float(sound_style_scale)
        self.sound_ctc_head = None
        self.n_sound_events = 0
        self.n_sound_styles = 0
        if not self.merge_sound_expert_to_asr:
            # SoundToken path: the sound expert reaches ASR as discrete EVENT TOKENS
            # rather than as encoder states. Exactly the speaker branch's shape --
            # posteriors -> threshold -> LayerNorm -> matmul(kernel) -> scaled add --
            # with the CTC head standing in for forward_speaker_sigmoids.
            if sound_ctc_head_cfg is None:
                raise ValueError(
                    "merge_sound_expert_to_asr=False needs sound_ctc_head_cfg: the event "
                    "tokens come from the sound expert's CTC head, which is not part of "
                    "its encoder. Bundles built by build_pee_v2_bundle.py carry it as the "
                    "sound checkpoint's `decoder:` block. Pass merge_sound_expert_to_asr="
                    "True to use the encoder-state merge instead, which needs no head."
                )
            self.sound_ctc_head = _build_from_cfg(_clone_config(sound_ctc_head_cfg), 'sound_ctc_head_cfg')
            vocabulary = list(sound_ctc_head_cfg.get('vocabulary') or ())
            if not vocabulary:
                raise ValueError(
                    "sound_ctc_head_cfg has no `vocabulary`; the tag column indices are "
                    "read from it and cannot be guessed from num_classes."
                )
            # Read the tag ids out of the vocabulary rather than hard-coding them: they
            # are an artifact of how the sound expert's tokenizer was built and will move
            # the next time it is retrained.
            event_ids = [i for i, tok in enumerate(vocabulary) if _SOUND_EVENT_TOKEN_RE.match(str(tok))]
            if not event_ids:
                raise ValueError(
                    f"sound_ctc_head_cfg.vocabulary has no {_SOUND_EVENT_TOKEN_PREFIX}... "
                    f"tokens among its {len(vocabulary)} entries, so there is nothing to "
                    "inject. Is this the CTC-head sound expert?"
                )
            self.sound_event_tokens = tuple(str(vocabulary[i]) for i in event_ids)
            self.n_sound_events = len(event_ids)
            self.register_buffer("sound_event_token_ids", torch.tensor(event_ids, dtype=torch.long), persistent=False)

            style_ids = []
            if self.inject_sound_styles:
                style_ids = [i for i, tok in enumerate(vocabulary) if _SOUND_STYLE_TOKEN_RE.match(str(tok))]
                if not style_ids:
                    raise ValueError(
                        f"inject_sound_styles=True but the vocabulary has no "
                        f"{_SOUND_STYLE_TOKEN_PREFIX}stt|end:... tokens among its "
                        f"{len(vocabulary)} entries. Pass inject_sound_styles=False to "
                        "inject only the event tags."
                    )
            self.sound_style_tokens = tuple(str(vocabulary[i]) for i in style_ids)
            self.n_sound_styles = len(style_ids)
            self.register_buffer("sound_style_token_ids", torch.tensor(style_ids, dtype=torch.long), persistent=False)

            # Events and styles are two SEPARATE families, each with its own LayerNorm
            # over its own tags and its own block of sinusoid rows.
            #
            # A single LayerNorm spanning both would couple them: its mean and variance
            # run over every tag, so each style that fires would rewrite the value placed
            # on an active event dim. Measured on a shared norm over all 32 tags, the
            # event code moves to 0.949 cosine (and 149.7 -> 119.3 in magnitude) when an
            # unrelated style co-fires, i.e. the event signal would silently depend on
            # style activity. With per-family norms that cross-talk is exactly zero.
            #
            # It also keeps the two scales independent, which matters because the expert
            # is markedly weaker on styles than on events: `sound_style_scale=0.0`
            # recovers the event-only behaviour without disturbing the event path.
            # Each family also gets its own strided block of sinusoid rows, disjoint from
            # the speaker block, so an event, a style and a speaker never push the ASR
            # states in the same direction.
            self.sound_token_norm = nn.LayerNorm(self.n_sound_events)
            self.register_buffer(
                "sound_token_kernel",
                self._build_tag_kernel(self.n_sound_events, _SOUND_EVENT_ROW_OFFSET, self.asr_d_model),
                persistent=False,
            )
            if self.n_sound_styles:
                self.sound_style_norm = nn.LayerNorm(self.n_sound_styles)
                self.register_buffer(
                    "sound_style_kernel",
                    self._build_tag_kernel(self.n_sound_styles, _SOUND_STYLE_ROW_OFFSET, self.asr_d_model),
                    persistent=False,
                )

            # Frozen, like the speaker head. The threshold below is a hard binarization,
            # so no gradient would survive the fusion anyway; freezing makes that
            # explicit and keeps the head out of the optimizer.
            self.sound_ctc_head.eval()
            for p in self.sound_ctc_head.parameters():
                p.requires_grad = False

            # The same argument applies to the expert behind the head: on this route it
            # reaches ASR only through those binarized tags, so it cannot receive
            # gradient either. Leaving it unfrozen is not wrong, just inert -- and it
            # silently costs a full set of optimizer states for ~0.5 B parameters that
            # cannot move. Warn rather than override, since an auxiliary sound loss
            # added elsewhere is a legitimate reason to want it trainable.
            if not freeze_sound:
                logging.warning(
                    "merge_sound_expert_to_asr=False with freeze_sound=False: the sound "
                    "expert reaches the ASR states only through hard-thresholded event "
                    "tags, so it will receive NO gradient from this path and its "
                    "optimizer state is wasted. Set freeze_sound=True unless an "
                    "auxiliary sound loss trains it separately."
                )

        else:
            sound_d_model = int(self.pee.experts['sound'].d_model)
            if sound_d_model != self.asr_d_model:
                raise ValueError(
                    f"merge_sound_expert_to_asr requires the sound expert d_model "
                    f"({sound_d_model}) to match the speech expert d_model ({self.asr_d_model}); "
                    "the merge is an elementwise add onto the ASR states."
                )
            # Both streams are normalized before the add so `sound_merge_scale` is a true
            # RELATIVE weight rather than an absolute magnitude.
            #
            # This matters more than it looks. The merge happens before `asr_norm`, so the
            # speech states arrive raw -- measured RMS ~0.04 on the PEE-v2 speech expert,
            # against unit variance for a LayerNorm-ed sound stream. Adding those directly
            # would let sound take ~98% of the merged variance even at scale=0.3, burying
            # the speech expert. Normalizing speech here puts the two on the same footing,
            # so scale=s gives sound a variance share of s^2/(1+s^2) (the streams are
            # near-orthogonal in practice: measured cosine ~0.004).
            #
            # The speech-side norm is affine-free: it exists only to fix the scale, and
            # `asr_norm` right after the merge already carries a learnable affine. The
            # sound-side norm keeps its affine so training can still adjust the sound gain.
            #
            # Built only on this branch: on the SoundToken branch they would be
            # parameters in the state dict that no forward pass ever reads.
            self.merge_speech_norm = nn.LayerNorm(self.asr_d_model, elementwise_affine=False)
            self.sound_norm = nn.LayerNorm(self.asr_d_model)

        self._apply_freezing()

    def _apply_freezing(self) -> None:
        """Put each frozen branch in eval and drop its grads."""
        frozen = {
            'speech': self.freeze_speech,
            'speaker': self.freeze_speaker,
            'sound': self.freeze_sound,
        }
        for role, is_frozen in frozen.items():
            if not is_frozen:
                continue
            expert = self.pee.experts[role]
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False
        if self.freeze_speaker:
            # The head travels with the speaker expert.
            self.sortformer_modules.eval()
            for p in self.sortformer_modules.parameters():
                p.requires_grad = False
        if self.sound_ctc_head is not None:
            # Unconditionally frozen, unlike the speaker head: the event tags are
            # binarized before the kernel, so nothing could train it through the fusion.
            self.sound_ctc_head.eval()
            for p in self.sound_ctc_head.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "ParallelExpertEncoder":
        """Set training mode, but keep frozen experts in eval.

        The parent ``model.train()`` recurses into every sub-module, which would
        re-enable dropout in a frozen branch. This re-asserts ``eval()`` on the frozen
        experts so their outputs stay deterministic.

        Args:
            mode (bool): Whether to set training mode (``True``) or eval mode (``False``).

        Returns:
            ParallelExpertEncoder: ``self``, matching ``nn.Module.train``.
        """
        super().train(mode)
        for role, is_frozen in (
            ('speech', self.freeze_speech),
            ('speaker', self.freeze_speaker),
            ('sound', self.freeze_sound),
        ):
            if is_frozen:
                self.pee.experts[role].eval()
        if self.freeze_speaker:
            self.sortformer_modules.eval()
        if self.sound_ctc_head is not None:
            self.sound_ctc_head.eval()
        return self

    # ConformerEncoder-compatible properties (drop-in for SALM perception).
    @property
    def d_model(self) -> int:
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        return self.pee.experts['speech'].subsampling_factor

    @property
    def pre_encode(self):
        return self.pee.experts['speech'].pre_encode

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Recompute each expert's layers in backward instead of storing them.

        SALM's generic helper wraps ``encoder.layers[i]`` in ``checkpoint_wrapper``,
        which finds nothing here: this module holds no ``layers`` of its own, they live
        one level down in ``pee.experts[role]``. So it owns the policy instead, and
        forwards to each expert. Takes effect on the per-expert training path; the fused
        inference path (:meth:`GGEMMTransformerEncoder.forward_packed`) never calls the
        layer modules and is deliberately left alone.
        """
        for expert in self.pee.experts.values():
            if hasattr(expert, 'activation_checkpointing'):
                expert.activation_checkpointing = bool(enabled)
        if enabled and self.fused_forward_in_training:
            logging.warning(
                "set_activation_checkpointing(True) with fused_forward_in_training=True has "
                "no effect: the fused path runs the layers itself instead of calling the "
                "layer modules, so there is nothing to recompute. Leave "
                "fused_forward_in_training at its default False to get the memory back."
            )

    # freeze/unfreeze parity (plain nn.Module re-exposing the standalone helpers).
    def freeze(self) -> None:
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        unfreeze(self, partial=partial)

    # Fusion helpers
    @staticmethod
    def _build_sinusoid_position_encoding(max_position: int, embedding_dim: int) -> torch.Tensor:
        """Mirror of ``MSEncDecMultiTaskModel.get_sinusoid_position_encoding``."""
        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_position, embedding_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @classmethod
    def _build_tag_kernel(
        cls,
        n_tags: int,
        row_offset: int,
        embedding_dim: int,
        stride: int = _TAG_ROW_STRIDE,
        calibrate: bool = True,
    ) -> torch.Tensor:
        """Take ``n_tags`` sinusoid rows from ``row_offset``, spaced ``stride``, and calibrate.

        Every tag family slices the one shared table, so no two families can alias onto
        each other by construction. See ``_TAG_ROW_STRIDE`` for why the rows are strided
        and offset rather than taken consecutively.

        When ``calibrate``, the kernel is then rescaled so that a single active tag injects
        a vector of norm ``sqrt(embedding_dim)`` -- the norm of the LayerNorm-ed states it
        is added to. Without this the injection strength depends on ``n_tags`` and on the
        row choice, both layout decisions rather than modelling ones: a LayerNorm over
        ``n`` inputs emits a larger code as ``n`` grows, so widening a family silently
        turns its contribution up. With it, the ``*_scale`` arguments mean a
        straightforward *fraction of the ASR state magnitude* and stay meaningful when the
        layout changes. Pass ``calibrate=False`` only to reproduce the v2.0.0 speaker
        kernel exactly.

        Returns:
            Tag kernel. Shape ``(n_tags, embedding_dim)``.
        """
        rows = [row_offset + stride * i for i in range(n_tags)]
        table = cls._build_sinusoid_position_encoding(rows[-1] + 1, embedding_dim)
        kernel = table[rows].contiguous()
        if not calibrate:
            return kernel

        # The mean single-tag code, computed with the LayerNorm at its init values
        # (gamma=1, beta=0). The norms themselves stay learnable, so the model is free to
        # move away from this starting point.
        eye = torch.eye(n_tags, dtype=kernel.dtype)
        centred = (eye - eye.mean(dim=1, keepdim=True)) / eye.std(dim=1, unbiased=False, keepdim=True)
        mean_norm = (centred @ kernel).norm(dim=1).mean()
        return (kernel * (math.sqrt(embedding_dim) / mean_norm)).contiguous()

    def _check_spk_target_width(self, spk_targets: Optional[torch.Tensor]) -> None:
        """Reject speaker targets whose speaker axis disagrees with ``n_spk``.

        ``n_spk`` is baked into the bundle: it fixes the row count of ``diar_kernel`` and
        the normalized shape of ``diar_norm``, so it cannot adapt to the batch. Left
        unchecked, a narrower target broadcasts against the predictions inside
        :meth:`_fuse_diar_and_asr` and reports a bare size mismatch several frames away
        from the setting that caused it.
        """
        if spk_targets is None:
            return
        n_given = int(spk_targets.shape[-1])
        if n_given != self.n_spk:
            raise ValueError(
                f"spk_targets carry {n_given} speaker slots but this ParallelExpertEncoder "
                f"was built with n_spk={self.n_spk}. Set the data-side speaker count to "
                f"{self.n_spk} (in SALM: data.multispeaker_cfg.num_speakers) so unused "
                "speakers arrive as empty columns, or export a bundle whose speaker expert "
                f"has n_spk={n_given}."
            )

    @staticmethod
    def _align_diar_frames(spk_targets: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad-by-repeat or truncate ``spk_targets`` to ``target_len`` along time."""
        cur_len = spk_targets.shape[1]
        if cur_len < target_len:
            last = spk_targets[:, -1:, :]
            spk_targets = torch.cat([spk_targets, last.repeat(1, target_len - cur_len, 1)], dim=1)
        elif cur_len > target_len:
            spk_targets = spk_targets[:, :target_len, :]
        return spk_targets

    def _match_module_io(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast ``tensor`` to the experts' device & dtype (mels arrive fp32, experts run bf16)."""
        param = next(self.pee.parameters(), None)
        if param is None:
            return tensor
        return tensor.to(device=param.device, dtype=param.dtype)

    def _speaker_head(self, speaker_encoded: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Speaker states -> per-frame speaker activity sigmoids.

        Mirrors ``SortformerEncLabelModel.frontend_encoder`` + ``forward_infer``: the
        encoder output is projected by ``encoder_proj`` (1024 -> ``tf_d_model``), then
        read out by ``forward_speaker_sigmoids`` and masked to the valid frames. The
        v2 checkpoint's intervening ``transformer_encoder`` has ``num_layers: 0``, i.e.
        the identity, so it is not instantiated here.

        Args:
            speaker_encoded (Tensor): Speaker expert output. Shape ``(B, D_spk, T)``.
            length (Tensor): Valid frame counts. Shape ``(B,)``.

        Returns:
            Speaker activity probabilities. Shape ``(B, T, n_spk)``.
        """
        emb_seq = speaker_encoded.transpose(1, 2)  # (B, T, fc_d_model)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)  # (B, T, tf_d_model)
        preds = self.sortformer_modules.forward_speaker_sigmoids(emb_seq)  # (B, T, n_spk)
        mask = self.sortformer_modules.length_to_mask(length, emb_seq.shape[1])
        return preds * mask.unsqueeze(-1).to(preds.dtype)

    def _sound_tag_posteriors(self, sound_encoded: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sound states -> per-frame event and style tag probabilities.

        The analogue of :meth:`_speaker_head`: run the frozen CTC head once and keep the
        ``<ev:...>`` and ``<sty:stt|end:...>`` columns.

        The softmax spans the FULL CTC vocabulary before the columns are selected -- the
        head is trained as a distribution over all 13k tokens, so a bare softmax over the
        tag columns alone would renormalize away the (usually dominant) mass on blank and
        on ordinary word pieces, and report near-certainty on the most likely tag for
        every frame including silence.

        Args:
            sound_encoded (Tensor): Sound expert output. Shape ``(B, D_sound, T)``.

        Returns:
            ``(events, styles)`` with shapes ``(B, T, n_sound_events)`` and
            ``(B, T, n_sound_styles)``; ``styles`` is ``None`` when styles are not injected.
        """
        log_probs = self.sound_ctc_head(encoder_output=sound_encoded)  # (B, T, V+1)
        events = log_probs.index_select(-1, self.sound_event_token_ids).exp()
        styles = None
        if self.n_sound_styles:
            styles = log_probs.index_select(-1, self.sound_style_token_ids).exp()
        return events, styles

    def _inject_sound_tokens(self, asr_encoded: torch.Tensor, sound_encoded: torch.Tensor) -> torch.Tensor:
        """Fuse sound tags into the ASR states (threshold + LayerNorm + kernel + ADD).

        Deliberately the same shape as :meth:`_fuse_diar_and_asr`, on disjoint sets of
        sinusoid rows, so an event, a style and a speaker never push the states the same
        way. The result is low rank: at most ``n_sound_events + n_sound_styles``
        directions into ``d_model``.

        Events and styles are normalized and projected SEPARATELY, then summed. Sharing a
        LayerNorm would make each family's code depend on how many tags of the other
        family happen to be firing -- see the note where the two norms are built.

        Args:
            asr_encoded (Tensor): Speech states. Shape ``(B, D, T)``.
            sound_encoded (Tensor): Sound expert states. Shape ``(B, D_sound, T)``.

        Returns:
            States with the tag signal added. Shape ``(B, D, T)``.
        """
        if sound_encoded.shape[-1] != asr_encoded.shape[-1]:
            raise ValueError(
                f"sound expert produced {sound_encoded.shape[-1]} frames but the ASR states "
                f"have {asr_encoded.shape[-1]}; the two experts must share a frame grid."
            )
        with torch.no_grad():
            events, styles = self._sound_tag_posteriors(sound_encoded)
            # Hard threshold, matching the speaker branch: the kernels are defined over
            # tag PRESENCE, so the same binary signal is injected at train and at
            # inference time regardless of how confident the head happens to be.
            events = (events > self.sound_event_threshold).to(asr_encoded.dtype)
            if styles is not None:
                styles = (styles > self.sound_event_threshold).to(asr_encoded.dtype)

        states = asr_encoded.transpose(1, 2)  # (B, T, D)
        events = self.sound_token_norm(events)
        infusion = self.sound_kernel_scale * torch.matmul(events, self.sound_token_kernel.to(events.dtype))
        if styles is not None:
            styles = self.sound_style_norm(styles)
            infusion = infusion + self.sound_style_scale * torch.matmul(
                styles, self.sound_style_kernel.to(styles.dtype)
            )
        fused = infusion.to(states.dtype) + states
        return fused.transpose(1, 2).to(asr_encoded.dtype)  # (B, D, T)

    def _merge_sound_and_asr(self, asr_encoded: torch.Tensor, sound_encoded: torch.Tensor) -> torch.Tensor:
        """Add the sound expert's encoder states onto the ASR states.

        The alternative to :meth:`_inject_sound_tokens`, selected by
        ``merge_sound_expert_to_asr=True``: the whole sound representation is added,
        rather than only the discrete event tags the CTC head reads out of it.

        Both experts run over the same frames in the same packed group, so the two
        tensors are already aligned in time and need no resampling.

        Args:
            asr_encoded (Tensor): Speech (and speaker-fused) states. Shape ``(B, D, T)``.
            sound_encoded (Tensor): Sound expert states. Shape ``(B, D, T)``.

        Returns:
            Merged states, shape ``(B, D, T)``.
        """
        if not self.merge_sound_expert_to_asr:
            raise RuntimeError(
                "_merge_sound_and_asr is the encoder-state path, but "
                "merge_sound_expert_to_asr is False; _inject_sound_tokens is the one to call."
            )
        if sound_encoded.shape[-1] != asr_encoded.shape[-1]:
            raise ValueError(
                f"sound expert produced {sound_encoded.shape[-1]} frames but the ASR states "
                f"have {asr_encoded.shape[-1]}; the two experts must share a frame grid."
            )
        # Normalize BOTH streams so the scale is a relative weight (see __init__).
        speech_states = self.merge_speech_norm(asr_encoded.transpose(1, 2))  # (B, T, D)
        sound_states = self.sound_norm(sound_encoded.transpose(1, 2))  # (B, T, D)
        merged = speech_states + self.sound_merge_scale * sound_states.to(speech_states.dtype)
        return merged.transpose(1, 2).to(asr_encoded.dtype)  # (B, D, T)

    def _fuse_diar_and_asr(
        self,
        asr_encoded: torch.Tensor,
        spk_targets: torch.Tensor,
        *,
        diarization_preds: Optional[torch.Tensor] = None,
        use_diarization: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse speech states with speaker-activity preds (LayerNorm + sinusoidal kernel + ADD).

        Args:
            asr_encoded (Tensor): Speech expert output. Shape ``(B, D, T_asr)``.
            spk_targets (Tensor): RTTM or Sortformer speaker activity. Shape
                ``(B, T_diar, n_spk)``.
            diarization_preds (Tensor, optional): Diarization predictions used for
                rows selected by ``use_diarization``. Callers must supply these whenever
                ``use_diarization`` can select a row; :meth:`_forward` guarantees it.
            use_diarization (Tensor, optional): Bool mask with shape ``(B,)``.

        Returns:
            Fused encoder output. Shape ``(B, D, T_asr)``.
        """
        asr_enc_states = asr_encoded.transpose(1, 2)  # (B, T, D)
        spk_targets = self._align_diar_frames(spk_targets, asr_enc_states.shape[1]).to(asr_enc_states.dtype)
        # Select per row on device. Reading `use_diarization.any()` here to skip the select
        # would block the host on a device transfer, and with activation checkpointing this
        # runs again inside the backward pass, where stalling the host reorders the
        # surrounding FSDP gradient reduce-scatters relative to peer ranks.
        if use_diarization is not None and diarization_preds is not None:
            if use_diarization.numel() != spk_targets.shape[0]:
                raise ValueError(
                    f"use_diarization size ({use_diarization.numel()}) must match "
                    f"the speaker-target batch size ({spk_targets.shape[0]})."
                )
            diarization_preds = self._align_diar_frames(diarization_preds, asr_enc_states.shape[1]).to(
                asr_enc_states.dtype
            )
            spk_targets = torch.where(
                use_diarization.to(device=spk_targets.device, dtype=torch.bool).view(-1, 1, 1),
                diarization_preds,
                spk_targets,
            )

        # RTTM targets and Sortformer sigmoid outputs must produce the same
        # binary speaker kernel at train and inference time.
        spk_targets = (spk_targets > self.speaker_activity_threshold).to(asr_enc_states.dtype)

        asr_enc_states = self.asr_norm(asr_enc_states)
        spk_targets = self.diar_norm(spk_targets)
        speaker_infusion = torch.matmul(spk_targets, self.diar_kernel.to(spk_targets.dtype))
        fused = self.spk_kernel_scale * speaker_infusion + asr_enc_states

        return fused.transpose(1, 2)  # (B, D, T)

    @contextlib.contextmanager
    def online_inference(self, enabled: bool = True):
        """Route :meth:`forward` through the windowed long-form path inside this block.

        Generation always opens this; training and validation never do. Off by default, and
        deliberately not inferred from ``self.training``, because validation also runs in
        eval mode and must stay on the single-pass path. The windowed loop calls the packed
        encoder once per window, so the number of collectives it emits tracks each rank's
        own audio length -- fine in one process, a deadlock in a distributed step.
        """
        previous = self.online_inference_enabled
        self.online_inference_enabled = bool(enabled)
        try:
            yield
        finally:
            self.online_inference_enabled = previous

    def _prepare_input(self, audio_signal, length):
        """Normalize and cast the shared mel input every expert consumes.

        There is a single packed call over a single input tensor, so this runs once
        and all three experts see the same ``per_feature``-normalised features -- they
        cannot be normalised independently the way the old serial branches were.
        """
        if self.asr_normalize_type:
            audio_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.asr_normalize_type)
        audio_signal = self._match_module_io(audio_signal)
        return audio_signal, length.to(device=audio_signal.device)

    # Forward — identical signature to ConformerEncoder.forward
    def forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
        return_experts: bool = False,
    ):
        """Encode ``audio_signal``, fusing speaker activity into the speech states.

        Fusion is per row and the same in every mode: a row with RTTM uses its
        ``spk_targets``, a row of ``-1`` (no RTTM) uses the Sortformer prediction.

        Only the encoding differs:

        1. Training and validation always take :meth:`_forward`, a single pass over the
           whole utterance, so every rank issues the same collectives.
        2. Generation opens :meth:`online_inference` and takes :meth:`_forward_online`,
           which walks long-form audio window by window with a live speaker cache.

        Args:
            audio_signal (Tensor): Un-normalised mel features; `per_feature` normalization
                is re-applied internally. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            spk_targets (Tensor, optional): ``(B, T, n_spk)`` RTTM/oracle speaker activity.
                ``None`` predicts for the whole batch; a row of ``-1`` predicts for that row,
                so RTTM and non-RTTM examples can share a batch.
            return_experts (bool): Also return the per-expert outputs (including the
                unfused sound expert and the speaker activity predictions). Off by
                default so the 2-tuple return stays drop-in for :class:`ConformerEncoder`.

        Returns:
            ``(outputs, encoded_lengths)`` with ``outputs`` of shape ``(B, D, T_asr)``,
            or ``(outputs, encoded_lengths, experts)`` when ``return_experts``.
        """
        # Off unless a generation call opened `online_inference()`, so training and
        # validation both take the single-pass path no matter what the batch holds.
        use_online = self.online_inference_enabled and self.online_inference_length > 0

        runner = self._forward_online if use_online else self._forward
        outputs, lengths, experts = runner(audio_signal=audio_signal, length=length, spk_targets=spk_targets)
        if return_experts:
            return outputs, lengths, experts
        return outputs, lengths

    def _forward(self, audio_signal, length, spk_targets=None):
        """Offline (non-chunked) forward pass. See :meth:`forward` for argument semantics.

        Inference takes one :meth:`GGEMMTransformerEncoder.forward_packed` call for all
        three experts: they share the same input and therefore the same ``T``, so there is
        no streaming prefix and no padding, and the packed attention group stays
        FlashAttention-2 eligible. That fusion is what makes generation ~4x faster than
        running the experts one at a time.

        Training takes the per-expert path instead (see ``fused_forward_in_training``).
        Fusing costs memory that only matters once activations have to be kept for
        backward: it stacks every expert's FFN input into one ``(E_total, N, target_d)``
        tensor, and evaluates all ``moe_num_experts`` experts on every token so it can
        weight them by a router matrix that is zero outside the top ``k``. Both are a good
        trade when nothing is stored and a bad one when everything is.
        """
        self._check_spk_target_width(spk_targets)
        use_diarization = (
            None if spk_targets is None else (spk_targets <= self.missing_rttm_target).flatten(start_dim=1).any(dim=1)
        )
        # `use_diarization` is rank-local: whether this rank's batch holds a row needing
        # predicted diarization depends on which cuts the sampler handed it. Branching on it
        # makes the head -- and every collective and FSDP hook underneath it -- run on some
        # ranks and not others, which desynchronises them. Keep the decision batch-independent,
        # and never touch the tensor: `bool(...)` on it blocks the host on a device transfer.
        if spk_targets is None or self.always_run_diarization:
            run_diarization = True
        else:
            # Opt-out path. Reads batch content, so it both syncs the host and lets ranks
            # disagree; only safe in a single process.
            run_diarization = bool(use_diarization.any())

        signal, signal_length = self._prepare_input(audio_signal, length)
        with torch.set_grad_enabled(not (self.freeze_speech and self.freeze_speaker and self.freeze_sound)):
            if self.training and not self.fused_forward_in_training:
                # Each expert on its own unmodified path: no cross-expert stacking, and
                # the speech MoE dispatches only its top-k pairs. Equivalent to the fused
                # path within the tolerance README 4.1 documents (flex vs SDPA attention).
                packed = self.pee.forward_all(signal, signal_length)
            else:
                packed = self.pee.forward_packed(
                    signal, signal_length, backend=self.ggemm_backend, moe_mode=self.moe_mode
                )

        asr_encoded, asr_encoded_len = packed['speech']
        sound_encoded, sound_encoded_len = packed['sound']

        diarization_preds = None
        if run_diarization:
            speaker_encoded, speaker_len = packed['speaker']
            with torch.set_grad_enabled(not self.freeze_speaker):
                diarization_preds = self._speaker_head(speaker_encoded, speaker_len)
            if spk_targets is None:
                spk_targets = diarization_preds

        # Sound merges into the ASR states BEFORE the speaker fusion, so the two
        # together form the backbone that `asr_norm` normalizes and the speaker
        # kernel is then added on top of. Merging afterwards would instead bolt sound
        # onto an already-normalized sum, leaving its scale outside the norm.
        asr_states = asr_encoded
        if self.merge_sound_expert_to_asr:
            asr_states = self._merge_sound_and_asr(asr_states, sound_encoded)
        else:
            asr_states = self._inject_sound_tokens(asr_states, sound_encoded)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(
                asr_states,
                spk_targets,
                diarization_preds=diarization_preds,
                use_diarization=use_diarization,
            )
        else:
            outputs = asr_states

        experts = {
            'speech': (asr_encoded, asr_encoded_len),
            'sound': (sound_encoded, sound_encoded_len),
            'speaker_preds': diarization_preds,
        }
        return outputs, asr_encoded_len, experts

    def _forward_online(self, audio_signal, length, spk_targets=None):
        """Long-form generation path: dispatches to the offline pass or the windowed loop.

        If the batch fits a single window (``num_chunks == 1``) this delegates straight
        to :meth:`_forward`, which is the same computation without any of the streaming
        bookkeeping -- see the comment at the branch. Otherwise it runs
        :meth:`_forward_windowed`.

        Reached only from :meth:`online_inference`, which only generation opens -- the
        window count depends on this rank's audio, so the loop must never run in a
        distributed training or validation step.

        Walks the recording in non-overlapping windows of ``online_inference_length``
        output frames, each extended by left/right context. Per window, one
        ``forward_packed`` call runs all three experts, with the speaker's streaming
        cache passed as a ``prefix`` so it attends over ``[spkcache | fifo | chunk]``
        while speech and sound are right-padded to that longer ``T`` and masked to
        their own length. Speech and sound use overlap-and-trim; the speaker's
        full-window predictions and the window's projected speaker embeddings are
        handed to ``streaming_update``, which trims context, updates the cache and
        returns the chunk-only predictions. This mirrors
        ``SortformerEncLabelModel.forward_streaming_step`` one-for-one, so the cache
        compression, silence profile and speaker permutation logic keep working.

        Args:
            audio_signal (Tensor): Un-normalised mel features; `per_feature` normalization
                is re-applied internally. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            spk_targets (Tensor, optional): ``(B, T, n_spk)`` override. Rows carrying the
                ``-1`` sentinel still get a streaming Sortformer prediction; the rest keep
                their targets and only the encoder is chunked for them.

        Returns:
            ``(outputs, encoded_lengths, experts)``; ``outputs`` has shape ``(B, D, T_asr)``.
        """
        total_feat_len = min(audio_signal.shape[-1], int(length.max().item()))
        num_chunks = max(1, math.ceil(total_feat_len / self.chunk_feat_len))

        if num_chunks == 1:
            # The whole batch fits one window, so every part of the streaming
            # apparatus is dead weight: the cache is empty (the prefix is a (B, 0, D)
            # no-op), there is no context to trim, and `streaming_update` degenerates
            # to `preds[:, 0:chunk_len]` -- a plain slice -- because spkcache, fifo and
            # lc are all zero. Take the offline path instead; same result, none of the
            # per-window bookkeeping. This is the common case for short-utterance
            # benchmarks (RTFx median ~7 s against a 30 s chunk).
            #
            # `num_chunks` follows `length.max()`, so this is a per-BATCH decision: one
            # long utterance keeps the whole batch on the windowed path. Length-bucket
            # upstream to get the win on mixed batches.
            #
            # Trim to `total_feat_len` first -- the windowed path never looks past it,
            # and carrying the batch's right padding into the encoder would both change
            # T and waste the compute this fast path exists to save.
            return self._forward(
                audio_signal=audio_signal[:, :, :total_feat_len],
                length=length.clamp(max=total_feat_len),
                spk_targets=spk_targets,
            )

        return self._forward_windowed(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
            total_feat_len=total_feat_len,
            num_chunks=num_chunks,
        )

    def _forward_windowed(self, audio_signal, length, spk_targets, total_feat_len, num_chunks):
        """The windowed long-form loop proper: one ``forward_packed`` per window with a
        live speaker cache. Split out from :meth:`_forward_online` so the single-window
        fast path and this path can each be exercised directly.
        """
        # Normalise the whole utterance once (not per chunk) to match offline stats.
        signal, signal_length = self._prepare_input(audio_signal, length)

        # Rows filled with the `-1` sentinel carry no RTTM and need a prediction, exactly as
        # in `_forward`. Reading the mask on the host is fine here: only generation reaches
        # this path (see `online_inference`), and it already syncs on `length.max()` above.
        self._check_spk_target_width(spk_targets)
        use_diarization = (
            None if spk_targets is None else (spk_targets <= self.missing_rttm_target).flatten(start_dim=1).any(dim=1)
        )
        run_streaming_diar = spk_targets is None or bool(use_diarization.any())

        streaming_state, stream_dtype = None, signal.dtype
        if run_streaming_diar:
            streaming_state, stream_dtype = self._init_streaming_diar(batch_size=signal.shape[0])

        asr_chunks: List[torch.Tensor] = []
        sound_chunks: List[torch.Tensor] = []
        diar_chunks: List[torch.Tensor] = []
        asr_encoded_len = torch.zeros_like(signal_length)

        for chunk_idx in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE online inference",
            disable=getattr(self, '_suppress_online_pbar', False),
        ):
            stt = chunk_idx * self.chunk_feat_len
            end = min(stt + self.chunk_feat_len, total_feat_len)

            # Shared context-extended window (input mel frames) for every expert.
            enc_stt = max(stt - self.left_ctx_feat_len, 0)
            enc_end = min(end + self.right_ctx_feat_len, total_feat_len)
            left_offset = stt - enc_stt
            right_offset = enc_end - end

            # The speaker attends over [cache | chunk]. Rather than let the cache
            # extend it past the others and pad them with zeros, walk the window's
            # start back by exactly the cache length and splice in 'replace' mode: the
            # cache stands in for the speaker's leading frames, while speech and sound
            # spend those same slots on REAL left context. Same T, same FLOPs, no zeros
            # -- and with every expert full-length the group stays FA-2 eligible.
            prefix, cache_len, extra = None, 0, 0
            if run_streaming_diar:
                cache = torch.cat([streaming_state.spkcache, streaming_state.fifo], dim=1)
                cache_len = cache.shape[1]
                prefix = {'speaker': cache.to(dtype=signal.dtype)}
                if self.online_prefix_mode == 'replace':
                    # Only as far back as the recording actually goes.
                    extra = min(cache_len * self.subsampling_factor, enc_stt) // self.subsampling_factor

            # 'replace' needs the window to carry at least `cache_len` leading frames to
            # give up. Near the start of a recording it cannot, so fall back to the
            # zero-padded 'extend' path for those windows only.
            prefix_mode = 'replace' if (self.online_prefix_mode == 'replace' and extra == cache_len) else 'extend'
            ext_stt = enc_stt - (extra * self.subsampling_factor if prefix_mode == 'replace' else 0)

            window = signal[:, :, ext_stt:enc_end]
            window_length = (signal_length - ext_stt).clamp(min=0, max=enc_end - ext_stt)

            with torch.set_grad_enabled(not (self.freeze_speech and self.freeze_speaker and self.freeze_sound)):
                packed, pre_encode = self.pee.forward_packed(
                    window,
                    window_length,
                    backend=self.ggemm_backend,
                    moe_mode=self.moe_mode,
                    prefix=prefix,
                    return_pre_encode=True,
                    prefix_mode=prefix_mode,
                )

            # Trim context off in output-frame space using rounded cumulative positions.
            # Speech/sound now also carry `extra` frames of extra left context up front.
            n_extra = extra if prefix_mode == 'replace' else 0
            left_drop = n_extra + left_offset // self.subsampling_factor
            right_drop = right_offset // self.subsampling_factor
            core_len = round(end / self.subsampling_factor) - round(stt / self.subsampling_factor)

            enc_ctx, _ = packed['speech']
            core = max(0, min(core_len, enc_ctx.shape[-1] - left_drop))
            asr_chunks.append(enc_ctx[:, :, left_drop : left_drop + core])
            snd_ctx, _ = packed['sound']
            sound_chunks.append(snd_ctx[:, :, left_drop : left_drop + core])
            asr_encoded_len += core
            align_target = core

            if run_streaming_diar:
                speaker_encoded, speaker_len = packed['speaker']
                with torch.set_grad_enabled(not self.freeze_speaker):
                    # Predictions span [spkcache | fifo | lc + chunk + rc], which is
                    # exactly the layout `streaming_update` documents.
                    preds = self._speaker_head(speaker_encoded, speaker_len)
                # The speaker's chunk excludes the frames the cache stood in for, so it
                # spans exactly [lc | chunk | rc] of the ORIGINAL window -- its lc is
                # `left_offset`, not the `left_drop` speech/sound use (which also has
                # to skip the extra left context those two received).
                chunk_embs, _ = pre_encode['speaker']  # (B, lc+chunk+rc, fc_d_model)
                with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
                    streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
                        streaming_state,
                        chunk=chunk_embs.to(stream_dtype),
                        preds=preds.to(stream_dtype),
                        lc=left_offset // self.subsampling_factor,
                        rc=right_drop,
                    )
                # Newly emitted frames, aligned to the speech chunk (frame-parallel).
                diar_chunks.append(self._align_diar_frames(chunk_preds, align_target))

        asr_encoded = torch.cat(asr_chunks, dim=2)  # (B, D, T_asr)
        sound_encoded = torch.cat(sound_chunks, dim=2)
        diarization_preds = torch.cat(diar_chunks, dim=1) if run_streaming_diar else None
        if spk_targets is None:
            spk_targets = diarization_preds

        # Same order as the offline path: sound joins the ASR backbone first, then the
        # speaker kernel is fused on top. The windowed sound chunks were trimmed with
        # the identical offsets, so they stay frame-aligned with the ASR states.
        asr_states = asr_encoded
        if self.merge_sound_expert_to_asr:
            asr_states = self._merge_sound_and_asr(asr_states, sound_encoded)
        else:
            asr_states = self._inject_sound_tokens(asr_states, sound_encoded)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(
                asr_states,
                spk_targets,
                diarization_preds=diarization_preds,
                use_diarization=use_diarization,
            )
        else:
            outputs = asr_states

        experts = {
            'speech': (asr_encoded, asr_encoded_len),
            'sound': (sound_encoded, asr_encoded_len),
            'speaker_preds': diarization_preds,
        }
        return outputs, asr_encoded_len, experts

    def _init_streaming_diar(self, batch_size: int) -> Tuple[object, torch.dtype]:
        """Configure the Sortformer streaming params and build the initial state.

        Args:
            batch_size (int): Batch size for the streaming state.

        Returns:
            ``(streaming_state, stream_dtype)`` on the speaker expert's device & dtype.
        """
        sm = self.sortformer_modules
        sm.chunk_len = self.online_inference_length
        sm.chunk_left_context = self.chunk_left_context
        sm.chunk_right_context = self.chunk_right_context
        sm.fifo_len = self.diar_fifo_len
        sm.spkcache_update_period = self.diar_spkcache_update_period
        sm.spkcache_len = self.diar_spkcache_len
        sm._check_streaming_parameters()

        param = next(self.pee.experts['speaker'].parameters(), None)
        if param is not None:
            device, stream_dtype = param.device, param.dtype
        else:
            device, stream_dtype = self.diar_kernel.device, torch.get_default_dtype()

        with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
            # Synchronous update: the cache starts empty and grows to `spkcache_len`,
            # so the prefix (and therefore T) is shorter on the first few windows.
            streaming_state = sm.init_streaming_state(batch_size=batch_size, async_streaming=False, device=device)
        return streaming_state, stream_dtype
