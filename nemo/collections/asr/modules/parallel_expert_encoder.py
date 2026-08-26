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

"""Parallel Expert Speech Encoder.

Runs a Sortformer speaker-diarization expert and an ASR Conformer encoder on the
same mel input, then fuses their outputs (LayerNorm + sinusoidal speaker-kernel +
ADD). Expects un-normalised mels; the ASR branch re-applies ``normalize_batch``
internally. I/O matches :class:`ConformerEncoder` (drop-in). Only self-contained PE
bundles (inline ``asr_encoder_cfg`` + ``diarization_model_cfg`` in
``model_config.yaml``) are supported.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import tarfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    'ParallelExpertEncoder',
    'ParallelExpertEncoderPT',
    'TransformerCTCDecoder',
    'PEETransformerCTCTimestampExtractor',
]


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default float dtype.

    Makes ``SortformerModules.init_streaming_state`` allocate its dtype-less
    speaker-cache / FIFO buffers in the diarizer's dtype, avoiding fp32/bf16 mismatch.
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

    Skips the cross-rank ``all_reduce`` in ``SortformerEncLabelModel.forward_streaming``,
    which is unnecessary and unsafe for single-recording inference (e.g. a vLLM worker).
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


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a ``DictConfig`` without resolving interpolations.

    ``from_config_dict`` mutates its input in place, so sub-target builders get a copy.
    """
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


@experimental
class ParallelExpertEncoderPT(ModelPT):
    """ModelPT shell so a :class:`ParallelExpertEncoder` can be saved/restored as a
    ``.nemo`` archive (inline ``asr_encoder_cfg`` + ``diarization_model_cfg``).
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = ParallelExpertEncoder(
            asr_encoder_cfg=self._cfg.get('asr_encoder_cfg', None),
            diarization_model_cfg=self._cfg.get('diarization_model_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', None),
            freeze_diar=self._cfg.get('freeze_diar', True),
            freeze_asr=self._cfg.get('freeze_asr', False),
            online_inference_length=self._cfg.get('online_inference_length', 500),
            chunk_left_context=self._cfg.get('chunk_left_context', 50),
            chunk_right_context=self._cfg.get('chunk_right_context', 50),
            diar_fifo_len=self._cfg.get('diar_fifo_len', 40),
            diar_spkcache_update_period=self._cfg.get('diar_spkcache_update_period', 300),
            diar_spkcache_len=self._cfg.get('diar_spkcache_len', 188),
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

        * a local ``.nemo`` file is restored with :meth:`ModelPT.restore_from`;
        * otherwise ``model_path_or_name`` is treated as a pretrained model
          identifier -- a HuggingFace Hub repo id (``{repo}/{name}``) or an NGC
          alias -- and resolved with :meth:`Model.from_pretrained`, which
          downloads/caches the ``.nemo`` (honouring the HuggingFace cache and
          ``HF_HUB_OFFLINE``, so a prefetched cache works on offline nodes).

        This mirrors ``speechlm2.parts.pretrained.load_pretrained_nemo`` so PE
        bundles load uniformly from local files or model cards.

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
            bundle = cls.restore_from(
                restore_path=model_path_or_name,
                map_location=map_location,
                strict=strict,
            )
        else:
            bundle = cls.from_pretrained(
                model_name=model_path_or_name,
                map_location=map_location,
                strict=strict,
            )
        return bundle.encoder

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

        The template must describe the same architecture (``d_model``, ``n_spk``);
        mismatches raise :class:`ValueError` fail-fast.

        Args:
            encoder (ParallelExpertEncoder): The encoder whose weights are persisted.
            output_nemo_path (str): Destination ``.nemo`` path.
            template_bundle_path (str): Existing PE ``.nemo`` whose ``model_config.yaml`` is reused.
        """
        if not isinstance(encoder, ParallelExpertEncoder):
            raise TypeError(f"save_to_nemo expects a ParallelExpertEncoder, " f"got {type(encoder).__name__}")
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

        tmpl_asr = template_cfg.get('asr_encoder_cfg', None)
        tmpl_diar = template_cfg.get('diarization_model_cfg', None)
        if tmpl_asr in (None, {}, '') or tmpl_diar in (None, {}, ''):
            raise ValueError(
                f"Template bundle {template_bundle_path} is not self-contained "
                "(asr_encoder_cfg / diarization_model_cfg missing); it cannot be "
                "used as a save template."
            )

        tmpl_d_model = int(tmpl_asr.get('d_model', -1))
        tmpl_n_spk = int(tmpl_diar.get('sortformer_modules', {}).get('num_spks', -1))
        enc_d_model = int(encoder.d_model)
        enc_n_spk = int(encoder.n_spk)
        if tmpl_d_model != enc_d_model:
            raise ValueError(
                f"Template asr_encoder_cfg.d_model={tmpl_d_model} does not match "
                f"encoder.d_model={enc_d_model}; the saved bundle would fail "
                "strict reload."
            )
        if tmpl_n_spk != enc_n_spk:
            raise ValueError(
                f"Template diarization_model_cfg.sortformer_modules.num_spks="
                f"{tmpl_n_spk} does not match encoder.n_spk={enc_n_spk}; the "
                "saved bundle would fail strict reload."
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
    """Sortformer-diarizer + ASR Conformer encoder; I/O identical to :class:`ConformerEncoder`.

    Reconstructed from inline configs in the PE bundle's ``model_config.yaml``.

    Args:
        asr_encoder_cfg (DictConfig): Inline config for the ASR-side :class:`ConformerEncoder`.
        diarization_model_cfg (DictConfig): Inline config for the :class:`SortformerEncLabelModel`.
        asr_normalize_type (str, optional): Normalization replayed on the ASR branch. Defaults to ``per_feature``.
        freeze_diar (bool): Freeze the Sortformer parameters. Defaults to ``True``.
        freeze_asr (bool): Freeze the wrapped ASR ConformerEncoder. Defaults to ``False``.
        online_inference_length (int): Online-inference window in encoder output frames
            (default ``500`` ~= 40s); ``<= 0`` disables it.
        chunk_left_context (int): Left context (output frames) per online window, shared by
            both branches. Default ``50``.
        chunk_right_context (int): Right context (output frames) per online window, shared by
            both branches. Default ``50``.
        diar_fifo_len (int): Sortformer streaming ``fifo_len``. Default ``40``.
        diar_spkcache_update_period (int): Sortformer streaming ``spkcache_update_period``. Default ``300``.
        diar_spkcache_len (int): Sortformer streaming ``spkcache_len``. Default ``188``.
    """

    supports_external_speaker_targets = True
    parallel_expert_encoder_kind = "two_branch"

    def __init__(
        self,
        asr_encoder_cfg: DictConfig,
        diarization_model_cfg: DictConfig,
        asr_normalize_type: Optional[str] = None,
        freeze_diar: bool = True,
        freeze_asr: bool = False,
        online_inference_length: int = 500,
        chunk_left_context: int = 50,
        chunk_right_context: int = 50,
        diar_fifo_len: int = 40,
        diar_spkcache_update_period: int = 300,
        diar_spkcache_len: int = 188,
    ):
        super().__init__()

        # Lazy import: SortformerEncLabelModel imports from asr.modules (circular).
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        if asr_encoder_cfg is None or diarization_model_cfg is None:
            raise ValueError(
                "ParallelExpertEncoder requires both `asr_encoder_cfg` and "
                "`diarization_model_cfg`; self-contained PE bundles supply "
                "these inline in their model_config.yaml."
            )

        self.asr_encoder = ConformerEncoder.from_config_dict(_clone_config(asr_encoder_cfg))
        if not isinstance(self.asr_encoder, ConformerEncoder):
            raise TypeError(
                f"Expected `asr_encoder_cfg._target_` to instantiate a "
                f"ConformerEncoder, got {type(self.asr_encoder).__name__} instead."
            )
        self.asr_normalize_type = asr_normalize_type or 'per_feature'
        self._feat_in = self.asr_encoder._feat_in

        diarization_model_cfg = _clone_config(diarization_model_cfg)
        diarization_model_cfg.output_subsampling_factor = self.asr_encoder.subsampling_factor
        self.diarization_model = SortformerEncLabelModel.from_config_dict(diarization_model_cfg)
        if self.diarization_model.output_subsampling_factor != self.asr_encoder.subsampling_factor:
            raise ValueError(
                "ParallelExpertEncoder requires the diarization output subsampling factor "
                f"({self.diarization_model.output_subsampling_factor}) to equal the ASR encoder subsampling factor "
                f"({self.asr_encoder.subsampling_factor})."
            )

        self.freeze_diar = freeze_diar
        self.freeze_asr = freeze_asr

        # Long-form / online inference configuration.
        self.online_inference_length = int(online_inference_length)
        # None preserves current-main direct use: long eval inputs stream automatically.
        # SALM mounts set this to False; generation enables it through a context.
        self.online_inference_enabled: Optional[bool] = None
        # Overlap-and-trim context (output frames) shared by both branches.
        self.chunk_left_context = max(0, int(chunk_left_context))
        self.chunk_right_context = max(0, int(chunk_right_context))
        # Online-inference window + context in input mel frames (constant per session).
        self.chunk_feat_len = self.online_inference_length * self.asr_encoder.subsampling_factor
        self.left_ctx_feat_len = self.chunk_left_context * self.asr_encoder.subsampling_factor
        self.right_ctx_feat_len = self.chunk_right_context * self.asr_encoder.subsampling_factor
        self.diar_fifo_len = int(diar_fifo_len)
        self.diar_spkcache_update_period = int(diar_spkcache_update_period)
        self.diar_spkcache_len = int(diar_spkcache_len)

        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.asr_d_model = self.asr_encoder.d_model

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.register_buffer(
            "diar_kernel",
            self._build_sinusoid_position_encoding(self.n_spk, self.asr_d_model),
            persistent=False,
        )

        if self.freeze_diar:
            self.diarization_model.eval()
            for p in self.diarization_model.parameters():
                p.requires_grad = False
        if self.freeze_asr:
            self.asr_encoder.eval()
            for p in self.asr_encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "ParallelExpertEncoder":
        """Set training mode, but keep frozen sub-branches in eval.

        The parent ``model.train()`` recurses into every sub-module, which would re-enable
        dropout / BatchNorm stat updates in a frozen branch. This re-asserts ``eval()`` on
        the frozen Sortformer (and ASR encoder) so their outputs stay deterministic.

        Args:
            mode (bool): Whether to set training mode (``True``) or eval mode (``False``).

        Returns:
            ParallelExpertEncoder: ``self``, matching ``nn.Module.train``d.
        """
        super().train(mode)
        if self.freeze_diar:
            self.diarization_model.eval()
        if self.freeze_asr:
            self.asr_encoder.eval()
        return self

    # ConformerEncoder-compatible properties (drop-in for SALM perception).
    @property
    def d_model(self) -> int:
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        return self.asr_encoder.subsampling_factor

    @property
    def pre_encode(self):
        return self.asr_encoder.pre_encode

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

    @staticmethod
    def _match_module_io(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Cast ``tensor`` to ``module``'s parameter device & dtype (mels arrive fp32, experts run bf16).

        Args:
            tensor (Tensor): Input to align (e.g. mel features).
            module (nn.Module): Module whose first parameter sets the target device/dtype.

        Returns:
            ``tensor`` moved to the module's device/dtype, or unchanged if it has no parameters.
        """
        param = next(module.parameters(), None)
        if param is None:
            return tensor
        return tensor.to(device=param.device, dtype=param.dtype)

    def _fuse_diar_and_asr(self, asr_encoded: torch.Tensor, spk_targets: torch.Tensor) -> torch.Tensor:
        """Fuse ASR states with speaker-activity preds (LayerNorm + sinusoidal kernel + ADD).

        Args:
            asr_encoded (Tensor): ASR encoder output. Shape ``(B, D, T_asr)``.
            spk_targets (Tensor): Speaker-activity predictions. Shape ``(B, T_diar, n_spk)``.

        Returns:
            Fused encoder output. Shape ``(B, D, T_asr)``.
        """
        asr_enc_states = asr_encoded.transpose(1, 2)  # (B, T, D)
        spk_targets = self._align_diar_frames(spk_targets, asr_enc_states.shape[1]).to(asr_enc_states.dtype)

        asr_enc_states = self.asr_norm(asr_enc_states)
        spk_targets = self.diar_norm(spk_targets)
        speaker_infusion = torch.matmul(spk_targets, self.diar_kernel.to(spk_targets.dtype))
        fused = speaker_infusion + asr_enc_states

        return fused.transpose(1, 2)  # (B, D, T)

    @contextlib.contextmanager
    def online_inference(self, enabled: bool = True):
        """Temporarily select the windowed inference path."""
        previous = getattr(self, "online_inference_enabled", None)
        self.online_inference_enabled = bool(enabled)
        try:
            yield
        finally:
            self.online_inference_enabled = previous

    # Forward — identical signature to ConformerEncoder.forward
    def forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ):
        """Encode ``audio_signal``, optionally fusing diarization.

        Dispatches to :meth:`_forward` (offline) or :meth:`_forward_online` (long-form,
        inference-only, when the input exceeds one window).

        Args:
            audio_signal (Tensor): Un-normalised mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            spk_targets (Tensor, optional): ``(B, T, n_spk)`` speaker-activity override (RTTM/oracle);
                when ``None`` the wrapped Sortformer is run.

        Returns:
            Tuple ``(outputs, encoded_lengths)`` with ``outputs`` of shape ``(B, D, T_asr)``.
        """
        if spk_targets is not None:
            use_online = False
        elif getattr(self, "online_inference_enabled", None) is not None:
            use_online = bool(self.online_inference_enabled) and self.online_inference_length > 0
        elif self.online_inference_length > 0 and not self.training:
            # Even if spk_targets is None, use offline if audio is short enough
            use_online = audio_signal.shape[-1] > self.chunk_feat_len
        else:
            use_online = False

        if use_online:
            return self._forward_online(audio_signal=audio_signal, length=length, spk_targets=spk_targets)

        return self._forward(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
        )

    def _forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ):
        """Offline (non-chunked) forward pass. See :meth:`forward` for argument semantics."""
        if spk_targets is None:
            # Cast fp32 mels to the diarizer's device/dtype before its conv subsampling.
            diar_signal = self._match_module_io(audio_signal, self.diarization_model)
            diar_length = length.to(device=diar_signal.device)
            with torch.set_grad_enabled(not self.freeze_diar):
                emb_seq, emb_seq_length = self.diarization_model.frontend_encoder(
                    processed_signal=diar_signal,
                    processed_signal_length=diar_length,
                    bypass_pre_encode=False,
                )
                spk_targets = self.diarization_model.forward_infer(
                    emb_seq=emb_seq,
                    emb_seq_length=emb_seq_length,
                )

        if self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal,
                length,
                normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal
        # Cast fp32 mels to the ASR encoder's device/dtype before its conv subsampling.
        asr_audio_signal = self._match_module_io(asr_audio_signal, self.asr_encoder)
        asr_length = length.to(device=asr_audio_signal.device)

        with torch.set_grad_enabled(not self.freeze_asr):
            asr_encoded, asr_encoded_len = self.asr_encoder(
                audio_signal=asr_audio_signal,
                length=asr_length,
            )

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(asr_encoded, spk_targets)
        else:
            outputs = asr_encoded

        return outputs, asr_encoded_len

    def _forward_online(self, audio_signal, length, spk_targets=None):
        """Long-form online inference: a lock-step loop over fixed windows.

        Walks the recording in non-overlapping windows of ``online_inference_length``
        output frames. Both experts run on the same context-extended slice
        ``[stt - left : end + right]`` (differing only in normalization): the ASR
        encoder uses overlap-and-trim, while the streaming Sortformer carries its
        speaker-cache / FIFO state across windows and trims context internally.
        Per-window diar outputs are aligned to the ASR frame count, then both buffers
        are concatenated and fused once.

        Args:
            audio_signal (Tensor): Un-normalised mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            spk_targets (Tensor, optional): ``(B, T, n_spk)`` override; when given, only ASR is chunked.

        Returns:
            Tuple ``(outputs, encoded_lengths)`` with ``outputs`` of shape ``(B, D, T_asr)``.
        """
        total_feat_len = min(audio_signal.shape[-1], int(length.max().item()))
        num_chunks = max(1, math.ceil(total_feat_len / self.chunk_feat_len))

        # Normalise the whole utterance once (not per chunk) to match offline stats.
        if self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal,
                length,
                normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal

        # Match the ASR encoder's device/dtype (mels arrive fp32, encoder runs bf16).
        asr_audio_signal = self._match_module_io(asr_audio_signal, self.asr_encoder)
        length = length.to(device=asr_audio_signal.device)

        run_streaming_diar = spk_targets is None
        if run_streaming_diar:
            streaming_state, stream_dtype, diar_audio_signal, diar_length = self._init_streaming_diar(
                audio_signal,
                length,
                batch_size=audio_signal.shape[0],
            )
            n_spk = self.diarization_model.sortformer_modules.n_spk
            total_preds = torch.zeros(
                (diar_audio_signal.shape[0], 0, n_spk),
                device=diar_audio_signal.device,
                dtype=stream_dtype,
            )

        asr_chunks: List[torch.Tensor] = []
        diar_chunks: List[torch.Tensor] = []
        asr_encoded_len = torch.zeros_like(length)

        for chunk_idx in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE online inference",
            disable=getattr(self, '_suppress_online_pbar', False),
        ):
            stt = chunk_idx * self.chunk_feat_len
            end = min(stt + self.chunk_feat_len, total_feat_len)

            # Shared context-extended window (input mel frames) for both branches.
            enc_stt = max(stt - self.left_ctx_feat_len, 0)
            enc_end = min(end + self.right_ctx_feat_len, total_feat_len)
            left_offset = stt - enc_stt
            right_offset = enc_end - end

            asr_chunk = asr_audio_signal[:, :, enc_stt:enc_end]
            chunk_length = (length - enc_stt).clamp(min=0, max=enc_end - enc_stt)
            with torch.set_grad_enabled(not self.freeze_asr):
                enc_ctx, _ = self.asr_encoder(audio_signal=asr_chunk, length=chunk_length)
            # Trim context off in output-frame space using rounded cumulative positions.
            left_drop = left_offset // self.subsampling_factor
            core_len = round(end / self.subsampling_factor) - round(stt / self.subsampling_factor)
            core_len = max(0, min(core_len, enc_ctx.shape[-1] - left_drop))
            enc_chunk = enc_ctx[:, :, left_drop : left_drop + core_len]
            asr_chunks.append(enc_chunk)
            asr_encoded_len += core_len
            align_target = enc_chunk.shape[-1]

            # Diar branch: stream the same window; Sortformer trims context internally.
            if run_streaming_diar:
                prev_len = total_preds.shape[1]
                diar_chunk = diar_audio_signal[:, :, enc_stt:enc_end].transpose(1, 2)  # (B, t, feat_in)
                diar_chunk_length = (diar_length - enc_stt).clamp(min=0, max=enc_end - enc_stt)
                with (
                    torch.set_grad_enabled(not self.freeze_diar),
                    _disable_dist_feature_sync(),
                    _default_dtype(stream_dtype),
                ):
                    streaming_state, total_preds = self.diarization_model.forward_streaming_step(
                        processed_signal=diar_chunk,
                        processed_signal_length=diar_chunk_length,
                        streaming_state=streaming_state,
                        total_preds=total_preds,
                        left_offset=left_offset,
                        right_offset=right_offset,
                    )
                diar_raw = total_preds[:, prev_len:]
                # Newly emitted frames, aligned to the ASR chunk (frame-parallel).
                new_preds = self._align_diar_frames(diar_raw, align_target)
                diar_chunks.append(new_preds)

        asr_encoded = torch.cat(asr_chunks, dim=2)  # (B, D, T_asr)
        if run_streaming_diar:
            spk_targets = torch.cat(diar_chunks, dim=1)  # (B, T_asr, n_spk)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(asr_encoded, spk_targets)
        else:
            outputs = asr_encoded

        return outputs, asr_encoded_len

    def _init_streaming_diar(self, audio_signal: torch.Tensor, length: torch.Tensor, batch_size: int):
        """Configure the wrapped Sortformer for streaming and build its initial state.

        Args:
            audio_signal (Tensor): Input mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            batch_size (int): Batch size for the streaming state.

        Returns:
            ``(streaming_state, stream_dtype, diar_audio_signal, diar_length)`` cast onto
            the diarizer's device & dtype.
        """
        sm = self.diarization_model.sortformer_modules
        sm.chunk_len = self.online_inference_length
        sm.fifo_len = self.diar_fifo_len
        sm.spkcache_update_period = self.diar_spkcache_update_period
        sm.spkcache_len = self.diar_spkcache_len
        self.diarization_model._check_streaming_parameters()

        diar_param = next(self.diarization_model.parameters(), None)
        if diar_param is not None:
            self.diarization_model.to(diar_param.device)
            diar_device, stream_dtype = diar_param.device, diar_param.dtype
        else:
            diar_device, stream_dtype = audio_signal.device, torch.get_default_dtype()

        diar_audio_signal = audio_signal.to(device=diar_device, dtype=stream_dtype)
        diar_length = length.to(device=diar_device)

        with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
            streaming_state = sm.init_streaming_state(
                batch_size=batch_size,
                async_streaming=self.diarization_model.async_streaming,
                device=diar_device,
            )
        return streaming_state, stream_dtype, diar_audio_signal, diar_length


class TransformerCTCDecoder(ConvASRDecoder):
    """Temporary CTC head with an optional Transformer bridge.

    The standard ConvASRDecoder 1x1 Conv1d is always the CTC classifier.
    use_transformer controls whether a length-aware Transformer bridge is inserted
    before that classifier, allowing direct comparisons of Transformer+Conv and
    Conv-only timestamp heads.

    The module is colocated here temporarily while the final ownership of the timestamp
    head is decided. Transformer padding masks require encoded_lengths only when
    use_transformer=True.
    """

    requires_encoded_lengths = False

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        init_mode: str = "xavier_uniform",
        vocabulary: Optional[List[str]] = None,
        add_blank: bool = True,
        use_transformer: bool = True,
        d_model: Optional[int] = None,
        n_heads: int = 8,
        n_layers: int = 2,
        drop_rate: float = 0.1,
        dropout_pre_encoder: Optional[float] = None,
        dropout_emb: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        self_attention_model: Optional[str] = "rope",
        rope_base: float = 10000.0,
        rotary_fraction: float = 1.0,
        pos_emb_max_len: int = 5000,
        xscaling: bool = False,
        attn_mode: str = "full",
        sync_max_audio_length: bool = True,
        residual: bool = False,
        residual_scale: float = 1.0,
        learnable_residual_scale: bool = False,
    ):
        if residual and not use_transformer:
            raise ValueError("TransformerCTCDecoder residual connections require use_transformer=True.")

        super().__init__(
            feat_in=feat_in,
            num_classes=num_classes,
            init_mode=init_mode,
            vocabulary=vocabulary,
            add_blank=add_blank,
        )

        self.use_transformer = use_transformer
        self.requires_encoded_lengths = use_transformer

        self.input_projection = None
        self.transformer = None
        self.output_projection = None
        self.residual = residual
        if self.use_transformer:
            transformer_d_model = int(feat_in if d_model is None else d_model)
            self.input_projection = (
                nn.Identity() if transformer_d_model == feat_in else nn.Linear(feat_in, transformer_d_model)
            )
            self.transformer = TransformerEncoder(
                feat_in=transformer_d_model,
                d_model=transformer_d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                subsampling=None,
                subsampling_factor=1,
                drop_rate=drop_rate,
                dropout_pre_encoder=dropout_pre_encoder,
                dropout_emb=dropout_emb,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                ff_expansion=ff_expansion,
                pre_block_norm=pre_block_norm,
                self_attention_model=self_attention_model,
                rope_base=rope_base,
                rotary_fraction=rotary_fraction,
                pos_emb_max_len=pos_emb_max_len,
                xscaling=xscaling,
                attn_mode=attn_mode,
                sync_max_audio_length=sync_max_audio_length,
            )
            # This head uses TransformerEncoder's pre-encoded input path. Remove its
            # otherwise-unused pre-encoder so it cannot be optimized or checkpointed.
            self.transformer.pre_encode = nn.Identity()
            self.output_projection = (
                nn.Identity() if transformer_d_model == feat_in else nn.Linear(transformer_d_model, feat_in)
            )

            if residual:
                if learnable_residual_scale:
                    self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
                else:
                    self.register_buffer('residual_scale', torch.tensor(float(residual_scale)), persistent=True)

    def forward(
        self, encoder_output: torch.Tensor, encoded_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Produce CTC log probabilities from channels-first acoustic states.

        Args:
            encoder_output: PEE/encoder states with shape (B, D, T).
            encoded_lengths: Valid-frame counts, shape (B,). Required only when
                use_transformer=True.

        Returns:
            CTC log probabilities with shape (B, T, num_classes_with_blank).
        """
        if encoder_output.ndim != 3:
            raise ValueError(
                "TransformerCTCDecoder expects encoder_output with shape (B, D, T), "
                f"but got {tuple(encoder_output.shape)}."
            )
        if encoder_output.shape[1] != self._feat_in:
            raise ValueError(
                f"TransformerCTCDecoder expected {self._feat_in} encoder features, "
                f"but got {encoder_output.shape[1]}."
            )

        encoded = encoder_output
        if self.use_transformer:
            if encoded_lengths is None:
                raise ValueError("TransformerCTCDecoder requires encoded_lengths when use_transformer=True.")

            residual_input = encoded
            encoded = self.input_projection(encoded.transpose(1, 2))
            encoded, _ = self.transformer(
                audio_signal=encoded,
                length=encoded_lengths,
                bypass_pre_encode=True,
            )
            encoded = self.output_projection(encoded.transpose(1, 2)).transpose(1, 2)

            if self.residual:
                encoded = residual_input + self.residual_scale.to(dtype=encoded.dtype) * encoded

        return super().forward(encoder_output=encoded)


class PEETransformerCTCTimestampExtractor:
    """Extract speaker-specific word timestamps from PEE and a CTC timestamp head.

    This is intentionally an inference helper, not a new PEE encoder.  It keeps the
    supplied :class:`ParallelExpertEncoder` and :class:`TransformerCTCDecoder`
    untouched, runs the PEE ASR expert, and combines three already synchronized
    signals:

    * CTC log-probabilities from ``TransformerCTCDecoder``;
    * raw (pre-threshold) Sortformer speaker sigmoids; and
    * a Nemotron-Transcribe t-SOT string containing ``<spk:N>`` tags.

    CTC alignment follows the usual blank-expanded Viterbi dynamic program used by
    NeMo Forced Aligner. ``serialized`` mode is the default: it aligns the original
    t-SOT word order as one monotonic stream, then returns per-speaker word lists.
    This is the stable choice for a serialized generation. ``parallel`` mode instead
    independently aligns each tagged stream against the same CTC frames with a soft
    Sortformer activity prior. It can emit overlap, but is less constrained for a
    low-evidence or hallucinated t-SOT word.

    Args:
        encoder: Archive-compatible PEE module, or its ``ParallelExpertEncoderPT``
            wrapper.  It must support ``return_experts=True`` and return raw
            ``experts['speech']`` plus ``experts['speaker_preds']``.
        ctc_decoder: The separately loaded :class:`TransformerCTCDecoder` head.
        tokenizer: The BPE tokenizer used to train the CTC head.  It must expose
            ``text_to_ids(str)``.
        blank_id: CTC blank index.  Defaults to the decoder's last class.
        input_frame_seconds: Input mel frame shift, used with the PEE subsampling
            factor when an audio duration is not supplied.  PEE normally uses 0.01.
        ctc_frame_seconds: Optional explicit CTC frame shift.
        sortformer_frame_seconds: Optional explicit Sortformer frame shift.
        speaker_activity_threshold: Threshold used only to expose optional speaker
            activity subspans; raw sigmoid values remain in the DP.
        speaker_logprob_weight: Weight of ``log(sigmoid)`` added to CTC token-state
            emissions.  Set to zero for pure CTC alignment.
        alignment_mode: ``'serialized'`` (default) or ``'parallel'``.
        speaker_assignment_mode: ``'optimal'`` learns a per-audio one-to-one mapping
            from t-SOT tags to Sortformer columns using an initial CTC alignment;
            ``'identity'`` uses tag ``N`` as Sortformer column ``N``.
    """

    _SPEAKER_TAG_RE = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        ctc_decoder: Optional[TransformerCTCDecoder] = None,
        tokenizer: Optional[Any] = None,
        *,
        blank_id: Optional[int] = None,
        input_frame_seconds: float = 0.01,
        ctc_frame_seconds: Optional[float] = None,
        sortformer_frame_seconds: Optional[float] = None,
        speaker_activity_threshold: float = 0.5,
        speaker_logprob_weight: float = 0.25,
        alignment_mode: str = 'serialized',
        speaker_assignment_mode: str = 'optimal',
        epsilon: float = 1.0e-6,
    ):
        if input_frame_seconds <= 0:
            raise ValueError(f"input_frame_seconds must be positive, got {input_frame_seconds}.")
        if ctc_frame_seconds is not None and ctc_frame_seconds <= 0:
            raise ValueError(f"ctc_frame_seconds must be positive, got {ctc_frame_seconds}.")
        if sortformer_frame_seconds is not None and sortformer_frame_seconds <= 0:
            raise ValueError(f"sortformer_frame_seconds must be positive, got {sortformer_frame_seconds}.")
        if not 0.0 <= speaker_activity_threshold <= 1.0:
            raise ValueError("speaker_activity_threshold must be between zero and one.")
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.encoder = encoder
        self.ctc_decoder = ctc_decoder
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.input_frame_seconds = float(input_frame_seconds)
        self.ctc_frame_seconds = ctc_frame_seconds
        self.sortformer_frame_seconds = sortformer_frame_seconds
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.speaker_logprob_weight = float(speaker_logprob_weight)
        self.alignment_mode = self._validate_alignment_mode(alignment_mode)
        self.speaker_assignment_mode = self._validate_assignment_mode(speaker_assignment_mode)
        self.epsilon = float(epsilon)

    @classmethod
    def parse_sot_words(cls, sot_transcript: str) -> List[Dict[str, Any]]:
        """Parse a t-SOT transcript without sending speaker tags to the tokenizer.

        A tag remains active until the next tag.  Text that precedes the first tag
        is retained with ``speaker_tag=None`` rather than being silently assigned to
        speaker zero.
        """
        if not isinstance(sot_transcript, str):
            raise TypeError(f"sot_transcript must be a string, got {type(sot_transcript).__name__}.")

        words: List[Dict[str, Any]] = []
        active_speaker: Optional[int] = None
        cursor = 0

        def append_words(text: str, speaker_tag: Optional[int]) -> None:
            for word in re.findall(r"\S+", text):
                words.append(
                    {
                        'word': word,
                        'speaker_tag': speaker_tag,
                        'word_index': len(words),
                    }
                )

        for match in cls._SPEAKER_TAG_RE.finditer(sot_transcript):
            append_words(sot_transcript[cursor : match.start()], active_speaker)
            active_speaker = int(match.group(1))
            cursor = match.end()
        append_words(sot_transcript[cursor:], active_speaker)
        return words

    def extract_ctc_and_sortformer(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the PEE ASR expert and CTC head on already preprocessed mel features.

        This mirrors the preprocessor -> encoder -> decoder portion of
        ``examples/asr/transcribe_speech.py``.  The legacy PEE timestamp head was
        trained on raw ``experts['speech']`` states, not on PEE's fused top-level
        encoder output, so this method deliberately uses the former.
        """
        if self.encoder is None:
            raise ValueError("encoder is required to run PEE inference.")
        if self.ctc_decoder is None:
            raise ValueError("ctc_decoder is required to run PEE inference.")

        # ``ParallelExpertEncoderPT`` stores the callable module at ``.encoder``;
        # a direct PEE module is already the callable object we want.
        pee_encoder = getattr(self.encoder, 'encoder', self.encoder)
        if not isinstance(pee_encoder, nn.Module):
            raise TypeError(
                "encoder must be a PEE module or ParallelExpertEncoderPT wrapper; "
                f"got {type(pee_encoder).__name__}."
            )

        modules: List[nn.Module] = [pee_encoder, self.ctc_decoder]
        previous_modes = [(module, module.training) for module in modules]
        try:
            for module in modules:
                module.eval()
            with torch.inference_mode():
                try:
                    encoder_result = pee_encoder(
                        audio_signal=processed_signal,
                        length=processed_signal_length,
                        return_experts=True,
                    )
                except TypeError as error:
                    raise RuntimeError(
                        "This PEE does not expose return_experts=True. Use the archive-compatible "
                        "grouped-GEMM ParallelExpertEncoder, or call extract_from_outputs() with "
                        "precomputed CTC log-probs and Sortformer sigmoids."
                    ) from error

                if not isinstance(encoder_result, tuple) or len(encoder_result) != 3:
                    raise RuntimeError(
                        "PEE return_experts=True must return (encoded, encoded_lengths, experts)."
                    )
                _, _, experts = encoder_result
                if not isinstance(experts, dict) or 'speech' not in experts:
                    raise RuntimeError("PEE experts must include raw ASR states under experts['speech'].")
                if 'speaker_preds' not in experts:
                    raise RuntimeError("PEE experts must include Sortformer sigmoids under experts['speaker_preds'].")

                speech_expert = experts['speech']
                if not isinstance(speech_expert, tuple) or len(speech_expert) != 2:
                    raise RuntimeError("experts['speech'] must be a (states, lengths) tuple.")
                speech_states, speech_lengths = speech_expert
                ctc_log_probs = self.ctc_decoder(speech_states, encoded_lengths=speech_lengths)
                sortformer_sigmoids = experts['speaker_preds']
        finally:
            for module, was_training in previous_modes:
                module.train(was_training)

        if sortformer_sigmoids is None:
            raise RuntimeError("PEE did not return Sortformer sigmoid predictions for this audio.")
        return {
            'ctc_log_probs': ctc_log_probs,
            'ctc_lengths': speech_lengths,
            'sortformer_sigmoids': sortformer_sigmoids,
            # In the archive-compatible PEE these share the CTC timeline.  The
            # core extractor also handles a different Sortformer length explicitly.
            'sortformer_lengths': speech_lengths,
        }

    def extract_from_audio(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
        preprocessor: nn.Module,
        sot_transcript: str,
        *,
        audio_duration: Optional[float] = None,
        time_offset: float = 0.0,
        **alignment_kwargs: Any,
    ) -> Dict[str, Any]:
        """Preprocess one waveform, run PEE + CTC, and return word timestamps.

        ``time_offset`` should be the JSONL record's ``offset`` when timestamps are
        required on the original recording timeline; the default returns chunk-local
        seconds.
        """
        if not isinstance(preprocessor, nn.Module):
            raise TypeError(f"preprocessor must be an nn.Module, got {type(preprocessor).__name__}.")

        was_training = preprocessor.training
        try:
            preprocessor.eval()
            with torch.inference_mode():
                preprocessor_result = preprocessor(input_signal=input_signal, length=input_signal_length)
        finally:
            preprocessor.train(was_training)

        if not isinstance(preprocessor_result, tuple) or len(preprocessor_result) != 2:
            raise RuntimeError("preprocessor must return (processed_signal, processed_signal_length).")
        processed_signal, processed_signal_length = preprocessor_result
        model_outputs = self.extract_ctc_and_sortformer(processed_signal, processed_signal_length)

        if audio_duration is None:
            sample_rate = getattr(preprocessor, '_sample_rate', getattr(preprocessor, 'sample_rate', None))
            if sample_rate is not None:
                audio_duration = self._scalar_length(input_signal_length, 'input_signal_length') / float(sample_rate)

        return self.extract_from_outputs(
            sot_transcript=sot_transcript,
            audio_duration=audio_duration,
            time_offset=time_offset,
            **model_outputs,
            **alignment_kwargs,
        )

    def extract_from_outputs(
        self,
        ctc_log_probs: torch.Tensor,
        sortformer_sigmoids: Optional[torch.Tensor],
        sot_transcript: str,
        *,
        ctc_lengths: Optional[torch.Tensor] = None,
        sortformer_lengths: Optional[torch.Tensor] = None,
        audio_duration: Optional[float] = None,
        time_offset: float = 0.0,
        alignment_mode: Optional[str] = None,
        speaker_assignment_mode: Optional[str] = None,
        speaker_logprob_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Force-align a single t-SOT transcript from precomputed model outputs.

        Args:
            ctc_log_probs: CTC ``log_softmax`` output with shape ``(T, V)`` or
                ``(1, T, V)``.  It is cast to CPU fp32 for numerically stable DP.
            sortformer_sigmoids: Raw Sortformer sigmoid output with shape ``(T, S)``
                or ``(1, T, S)``.  It may have a different frame count from CTC;
                it is interpolated onto the CTC timeline for alignment.
            sot_transcript: Nemotron-Transcribe output containing ``<spk:N>`` tags.
            ctc_lengths: Valid CTC frames.  Batch size greater than one is rejected
                deliberately so every result is tied to one audio record.
            sortformer_lengths: Valid Sortformer frames.
            audio_duration: Chunk duration in seconds.  When supplied, it controls
                the timestamp grid exactly; otherwise the PEE frame shift is used.
            time_offset: Added to output seconds, e.g. JSONL ``offset``.
            alignment_mode: Per-call ``'parallel'`` or ``'serialized'`` override.
            speaker_assignment_mode: Per-call ``'optimal'`` or ``'identity'``
                Sortformer-column mapping override.
            speaker_logprob_weight: Per-call Sortformer DP-prior weight override.

        Returns:
            A dictionary whose ``speaker_word_timestamps`` contains one ordered list
            per t-SOT speaker tag.  Word ``start`` / ``end`` are CTC-derived seconds;
            Sortformer details are supplied as activity/confidence metadata.
        """
        mode = self._validate_alignment_mode(alignment_mode or self.alignment_mode)
        assignment_mode = self._validate_assignment_mode(
            speaker_assignment_mode or self.speaker_assignment_mode
        )
        speaker_weight = self.speaker_logprob_weight if speaker_logprob_weight is None else speaker_logprob_weight
        if speaker_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")

        ctc = self._single_recording_tensor(ctc_log_probs, 'ctc_log_probs', expected_ndim=2)
        if ctc.shape[0] == 0 or ctc.shape[1] < 2:
            raise ValueError(f"ctc_log_probs must have at least one frame and two classes, got {tuple(ctc.shape)}.")
        ctc_length = self._select_length(ctc_lengths, ctc.shape[0], 'ctc_lengths')
        ctc = ctc[:ctc_length].detach().to(device='cpu', dtype=torch.float32)
        if torch.isnan(ctc).any():
            raise ValueError("ctc_log_probs contains NaN values.")
        ctc_log_normalizer_error = float(torch.logsumexp(ctc, dim=-1).abs().max().item())
        if ctc_log_normalizer_error > 0.05:
            raise ValueError(
                "ctc_log_probs does not appear to be log-softmax output: maximum "
                f"log-normalization error is {ctc_log_normalizer_error:.4f}."
            )
        blank_id = self._resolve_blank_id(ctc.shape[1])

        sortformer: Optional[torch.Tensor] = None
        sortformer_length: Optional[int] = None
        if sortformer_sigmoids is not None:
            sortformer = self._single_recording_tensor(
                sortformer_sigmoids,
                'sortformer_sigmoids',
                expected_ndim=2,
            )
            if sortformer.shape[1] == 0:
                raise ValueError("sortformer_sigmoids must contain at least one speaker column.")
            sortformer_length = self._select_length(
                sortformer_lengths,
                sortformer.shape[0],
                'sortformer_lengths',
            )
            sortformer = sortformer[:sortformer_length].detach().to(device='cpu', dtype=torch.float32)
            if torch.isnan(sortformer).any():
                raise ValueError("sortformer_sigmoids contains NaN values.")
            if float(sortformer.min().item()) < -1.0e-3 or float(sortformer.max().item()) > 1.001:
                raise ValueError("sortformer_sigmoids must contain sigmoid probabilities in [0, 1].")
            sortformer = sortformer.clamp(min=0.0, max=1.0)
            sortformer_on_ctc = self._resample_speaker_probs(sortformer, ctc_length)
        else:
            sortformer_on_ctc = None

        if audio_duration is not None:
            audio_duration = float(audio_duration)
            if audio_duration <= 0:
                raise ValueError(f"audio_duration must be positive, got {audio_duration}.")
        time_offset = float(time_offset)
        ctc_step_seconds = self._resolve_ctc_frame_seconds(ctc_length, audio_duration)
        sortformer_step_seconds = self._resolve_sortformer_frame_seconds(
            sortformer_length,
            audio_duration,
            ctc_step_seconds,
        )

        tokenized_words = self._tokenize_words(self.parse_sot_words(sot_transcript), blank_id)
        speaker_tags = self._speaker_tags_in_order(tokenized_words)
        if not tokenized_words:
            return {
                'speaker_word_timestamps': {},
                'speaker_tag_to_sortformer_column': {},
                'alignment_mode': mode,
                'speaker_assignment_mode': assignment_mode,
                'ctc_frame_seconds': ctc_step_seconds,
                'sortformer_frame_seconds': sortformer_step_seconds,
                'time_offset': time_offset,
                'num_ctc_frames': ctc_length,
                'num_sortformer_frames': sortformer_length,
                'ctc_log_normalizer_error': ctc_log_normalizer_error,
                'alignment_diagnostics': {},
            }

        # First align the serialized word order using CTC alone.  It gives a stable
        # tag-to-Sortformer assignment even when the final output mode is parallel.
        preliminary_rows, preliminary_path_score = self._align_word_sequence(
            tokenized_words=tokenized_words,
            ctc_log_probs=ctc,
            blank_id=blank_id,
            speaker_probs=None,
            speaker_mapping={},
            ctc_step_seconds=ctc_step_seconds,
            time_offset=time_offset,
            speaker_logprob_weight=0.0,
        )
        speaker_mapping, assignment_scores = self._resolve_speaker_mapping(
            speaker_tags=speaker_tags,
            preliminary_rows=preliminary_rows,
            speaker_probs=sortformer_on_ctc,
            assignment_mode=assignment_mode,
        )

        alignment_scores: Dict[Optional[int], float] = {}
        if mode == 'serialized':
            rows, path_score = self._align_word_sequence(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=sortformer_on_ctc,
                speaker_mapping=speaker_mapping,
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=float(speaker_weight),
            )
            alignment_scores[None] = path_score
        else:
            rows = []
            for speaker_tag, speaker_words in self._group_words_by_speaker(tokenized_words).items():
                speaker_rows, path_score = self._align_word_sequence(
                    tokenized_words=speaker_words,
                    ctc_log_probs=ctc,
                    blank_id=blank_id,
                    speaker_probs=sortformer_on_ctc,
                    speaker_mapping=speaker_mapping,
                    ctc_step_seconds=ctc_step_seconds,
                    time_offset=time_offset,
                    speaker_logprob_weight=float(speaker_weight),
                )
                rows.extend(speaker_rows)
                alignment_scores[speaker_tag] = path_score

        speaker_word_timestamps: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for row in rows:
            speaker_word_timestamps.setdefault(row['speaker_tag'], []).append(row)

        return {
            'speaker_word_timestamps': speaker_word_timestamps,
            'speaker_tag_to_sortformer_column': speaker_mapping,
            'alignment_mode': mode,
            'speaker_assignment_mode': assignment_mode,
            'ctc_frame_seconds': ctc_step_seconds,
            'sortformer_frame_seconds': sortformer_step_seconds,
            'time_offset': time_offset,
            'num_ctc_frames': ctc_length,
            'num_sortformer_frames': sortformer_length,
            'ctc_log_normalizer_error': ctc_log_normalizer_error,
            'alignment_diagnostics': {
                'serialized_ctc_path_score': preliminary_path_score,
                'final_path_scores': alignment_scores,
                'speaker_assignment_scores': assignment_scores,
            },
        }

    def _tokenize_words(self, words: Sequence[Dict[str, Any]], blank_id: int) -> List[Dict[str, Any]]:
        """Attach per-word BPE IDs while refusing CTC-invalid tokenizer output."""
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to force-align t-SOT words.")
        if not hasattr(self.tokenizer, 'text_to_ids'):
            raise TypeError("tokenizer must expose a text_to_ids(str) method.")

        tokenized_words: List[Dict[str, Any]] = []
        for word_record in words:
            word = word_record['word']
            try:
                token_ids = self.tokenizer.text_to_ids(word)
            except TypeError as error:
                raise TypeError(
                    "tokenizer.text_to_ids(word) failed. Supply the single-language BPE tokenizer "
                    "used by the CTC head, rather than an aggregate tokenizer that requires a language ID."
                ) from error
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().tolist()
            token_ids = [int(token_id) for token_id in token_ids]
            if not token_ids:
                raise ValueError(f"Tokenizer produced no CTC tokens for word {word!r}.")
            invalid_ids = [token_id for token_id in token_ids if token_id < 0 or token_id >= blank_id]
            if invalid_ids:
                raise ValueError(
                    f"Tokenizer produced CTC-invalid IDs {invalid_ids} for word {word!r}; "
                    f"valid non-blank IDs are [0, {blank_id})."
                )
            item = dict(word_record)
            item['token_ids'] = token_ids
            tokenized_words.append(item)
        return tokenized_words

    @staticmethod
    def _speaker_tags_in_order(tokenized_words: Sequence[Dict[str, Any]]) -> List[int]:
        """Return distinct explicit t-SOT tags in first-occurrence order."""
        tags: List[int] = []
        for word in tokenized_words:
            tag = word['speaker_tag']
            if tag is not None and tag not in tags:
                tags.append(tag)
        return tags

    @staticmethod
    def _group_words_by_speaker(
        tokenized_words: Sequence[Dict[str, Any]],
    ) -> Dict[Optional[int], List[Dict[str, Any]]]:
        """Preserve t-SOT stream order within each speaker-specific transcript."""
        grouped: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for word in tokenized_words:
            grouped.setdefault(word['speaker_tag'], []).append(word)
        return grouped

    @staticmethod
    def _build_ctc_target(
        tokenized_words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> Tuple[List[int], List[Optional[int]], List[int]]:
        """Build ``[blank, token, blank, ...]`` labels and token-state ownership."""
        labels: List[int] = [blank_id]
        state_to_word: List[Optional[int]] = [None]
        flat_tokens: List[int] = []
        for word_index, word in enumerate(tokenized_words):
            for token_id in word['token_ids']:
                labels.append(token_id)
                state_to_word.append(word_index)
                labels.append(blank_id)
                state_to_word.append(None)
                flat_tokens.append(token_id)
        return labels, state_to_word, flat_tokens

    @staticmethod
    def _minimum_ctc_frames(token_ids: Sequence[int]) -> int:
        """Minimum CTC frames, including mandatory blanks for equal neighbours."""
        if not token_ids:
            return 0
        repeated_neighbours = sum(
            previous == current for previous, current in zip(token_ids[:-1], token_ids[1:])
        )
        return len(token_ids) + repeated_neighbours

    def _align_word_sequence(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        ctc_log_probs: torch.Tensor,
        blank_id: int,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step_seconds: float,
        time_offset: float,
        speaker_logprob_weight: float,
    ) -> Tuple[List[Dict[str, Any]], float]:
        labels, state_to_word, flat_tokens = self._build_ctc_target(tokenized_words, blank_id)
        minimum_frames = self._minimum_ctc_frames(flat_tokens)
        if minimum_frames > ctc_log_probs.shape[0]:
            raise ValueError(
                "CTC target is infeasible: it needs at least "
                f"{minimum_frames} frames (including repeated-token blanks), but only "
                f"{ctc_log_probs.shape[0]} valid CTC frames are available."
            )

        state_speaker_columns: List[Optional[int]] = [None] * len(labels)
        for state_index, local_word_index in enumerate(state_to_word):
            if local_word_index is None:
                continue
            speaker_tag = tokenized_words[local_word_index]['speaker_tag']
            state_speaker_columns[state_index] = speaker_mapping.get(speaker_tag)

        path, path_score = self._ctc_viterbi_align(
            ctc_log_probs=ctc_log_probs,
            labels=labels,
            blank_id=blank_id,
            state_speaker_columns=state_speaker_columns,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
        )
        rows = self._word_rows_from_path(
            tokenized_words=tokenized_words,
            labels=labels,
            state_to_word=state_to_word,
            path=path,
            ctc_log_probs=ctc_log_probs,
            speaker_probs=speaker_probs,
            speaker_mapping=speaker_mapping,
            ctc_step_seconds=ctc_step_seconds,
            time_offset=time_offset,
        )
        return rows, path_score

    def _ctc_viterbi_align(
        self,
        *,
        ctc_log_probs: torch.Tensor,
        labels: Sequence[int],
        blank_id: int,
        state_speaker_columns: Sequence[Optional[int]],
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
    ) -> Tuple[torch.Tensor, float]:
        """Run blank-expanded CTC Viterbi DP with an optional Sortformer prior.

        The recurrence is the one used by NeMo Forced Aligner: each state can stay,
        advance one state, or skip a blank when the two surrounding non-blank labels
        differ.  All arithmetic is fp32 on CPU because the PEE/CTC inference path is
        commonly bf16.
        """
        if ctc_log_probs.ndim != 2:
            raise ValueError(f"ctc_log_probs must have shape (T, V), got {tuple(ctc_log_probs.shape)}.")
        if len(labels) != len(state_speaker_columns):
            raise ValueError("labels and state_speaker_columns must have the same length.")
        if not labels:
            raise ValueError("Cannot align an empty CTC target.")

        log_probs = ctc_log_probs.detach().to(device='cpu', dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if int(labels_tensor.max().item()) >= log_probs.shape[1] or int(labels_tensor.min().item()) < 0:
            raise ValueError("CTC target contains labels outside the CTC vocabulary.")
        emissions = log_probs.index_select(dim=1, index=labels_tensor)

        if speaker_probs is not None and speaker_logprob_weight > 0.0:
            if speaker_probs.ndim != 2 or speaker_probs.shape[0] != log_probs.shape[0]:
                raise ValueError("speaker_probs must have shape (T_ctc, num_speakers).")
            state_columns = torch.tensor(
                [-1 if column is None else int(column) for column in state_speaker_columns],
                dtype=torch.long,
            )
            valid_states = torch.nonzero(state_columns >= 0, as_tuple=False).flatten()
            if valid_states.numel() > 0:
                speaker_columns = state_columns.index_select(0, valid_states)
                if int(speaker_columns.max().item()) >= speaker_probs.shape[1]:
                    raise ValueError("speaker mapping references a missing Sortformer column.")
                speaker_log_probs = torch.log(
                    speaker_probs[:, speaker_columns].to(dtype=torch.float32).clamp_min(self.epsilon)
                )
                emissions[:, valid_states] += float(speaker_logprob_weight) * speaker_log_probs

        num_frames, num_states = emissions.shape
        if num_states < 2:
            raise ValueError("CTC target must contain at least blank and one token state.")
        neg_inf = -float('inf')
        previous_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
        previous_scores[0] = emissions[0, 0]
        previous_scores[1] = emissions[0, 1]
        backpointers = torch.full((num_frames, num_states), -1, dtype=torch.long)
        backpointers[0, 0] = 0
        backpointers[0, 1] = 1
        state_indices = torch.arange(num_states, dtype=torch.long)

        for frame_index in range(1, num_frames):
            best_scores = previous_scores.clone()
            best_previous_states = state_indices.clone()

            advance_one_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
            advance_one_scores[1:] = previous_scores[:-1]
            take_advance_one = advance_one_scores > best_scores
            best_scores = torch.where(take_advance_one, advance_one_scores, best_scores)
            best_previous_states = torch.where(take_advance_one, state_indices - 1, best_previous_states)

            if num_states > 2:
                skip_positions = torch.arange(2, num_states, dtype=torch.long)
                can_skip = (labels_tensor[skip_positions] != blank_id) & (
                    labels_tensor[skip_positions] != labels_tensor[skip_positions - 2]
                )
                if can_skip.any():
                    allowed_positions = skip_positions[can_skip]
                    skip_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
                    skip_scores[allowed_positions] = previous_scores[allowed_positions - 2]
                    take_skip = skip_scores > best_scores
                    best_scores = torch.where(take_skip, skip_scores, best_scores)
                    best_previous_states = torch.where(take_skip, state_indices - 2, best_previous_states)

            previous_scores = best_scores + emissions[frame_index]
            backpointers[frame_index] = best_previous_states

        final_state = num_states - 1
        if previous_scores[num_states - 2] > previous_scores[final_state]:
            final_state = num_states - 2
        final_score = previous_scores[final_state]
        if not torch.isfinite(final_score):
            raise ValueError("No valid CTC Viterbi path exists for this transcript and audio.")

        path = torch.empty((num_frames,), dtype=torch.long)
        state = int(final_state)
        for frame_index in range(num_frames - 1, -1, -1):
            path[frame_index] = state
            if frame_index > 0:
                state = int(backpointers[frame_index, state].item())
                if state < 0:
                    raise RuntimeError("CTC Viterbi backtrace reached an invalid state.")
        return path, float(final_score.item())

    def _word_rows_from_path(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        labels: Sequence[int],
        state_to_word: Sequence[Optional[int]],
        path: torch.Tensor,
        ctc_log_probs: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step_seconds: float,
        time_offset: float,
    ) -> List[Dict[str, Any]]:
        """Convert a Viterbi state path into CTC word intervals and speaker metadata."""
        frames_by_word: List[List[int]] = [[] for _ in tokenized_words]
        for frame_index, state_index in enumerate(path.tolist()):
            word_index = state_to_word[state_index]
            if word_index is not None:
                frames_by_word[word_index].append(frame_index)

        rows: List[Dict[str, Any]] = []
        for word, frames in zip(tokenized_words, frames_by_word):
            if not frames:
                raise RuntimeError(f"CTC path did not visit any token state for word {word['word']!r}.")
            start_frame, end_frame = frames[0], frames[-1]
            selected_log_probs = torch.tensor(
                [ctc_log_probs[frame, labels[int(path[frame].item())]].item() for frame in frames],
                dtype=torch.float32,
            )
            speaker_tag = word['speaker_tag']
            sortformer_column = speaker_mapping.get(speaker_tag)
            speaker_confidence: Optional[float] = None
            speaker_activity_start: Optional[float] = None
            speaker_activity_end: Optional[float] = None
            if speaker_probs is not None and sortformer_column is not None:
                activity = speaker_probs[start_frame : end_frame + 1, sortformer_column]
                speaker_confidence = float(activity.mean().item())
                active_indices = torch.nonzero(activity >= self.speaker_activity_threshold, as_tuple=False).flatten()
                if active_indices.numel() > 0:
                    activity_start = start_frame + int(active_indices[0].item())
                    activity_end = start_frame + int(active_indices[-1].item())
                    speaker_activity_start = time_offset + activity_start * ctc_step_seconds
                    speaker_activity_end = time_offset + (activity_end + 1) * ctc_step_seconds

            rows.append(
                {
                    'word': word['word'],
                    'word_index': word['word_index'],
                    'speaker_tag': speaker_tag,
                    'start': time_offset + start_frame * ctc_step_seconds,
                    'end': time_offset + (end_frame + 1) * ctc_step_seconds,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'ctc_confidence': float(torch.exp(selected_log_probs.mean()).item()),
                    'sortformer_column': sortformer_column,
                    'speaker_confidence': speaker_confidence,
                    'speaker_activity_start': speaker_activity_start,
                    'speaker_activity_end': speaker_activity_end,
                }
            )
        return rows

    def _resolve_speaker_mapping(
        self,
        *,
        speaker_tags: Sequence[int],
        preliminary_rows: Sequence[Dict[str, Any]],
        speaker_probs: Optional[torch.Tensor],
        assignment_mode: str,
    ) -> Tuple[Dict[int, Optional[int]], Dict[int, List[float]]]:
        """Map t-SOT tags to raw Sortformer columns using preliminary CTC spans."""
        mapping: Dict[int, Optional[int]] = {tag: None for tag in speaker_tags}
        assignment_scores: Dict[int, List[float]] = {}
        if speaker_probs is None:
            return mapping, assignment_scores

        num_columns = speaker_probs.shape[1]
        valid_tags = [tag for tag in speaker_tags if 0 <= tag < num_columns]
        rows_by_tag: Dict[int, List[Dict[str, Any]]] = {tag: [] for tag in valid_tags}
        for row in preliminary_rows:
            tag = row['speaker_tag']
            if tag in rows_by_tag:
                rows_by_tag[tag].append(row)

        score_matrix: List[List[float]] = []
        for tag in valid_tags:
            column_scores: List[float] = []
            for column in range(num_columns):
                frame_scores: List[torch.Tensor] = []
                for row in rows_by_tag[tag]:
                    frame_scores.append(
                        torch.log(
                            speaker_probs[row['start_frame'] : row['end_frame'] + 1, column].clamp_min(self.epsilon)
                        )
                    )
                score = torch.cat(frame_scores).mean() if frame_scores else torch.tensor(-float('inf'))
                column_scores.append(float(score.item()))
            assignment_scores[tag] = column_scores
            score_matrix.append(column_scores)

        if assignment_mode == 'identity':
            for tag in valid_tags:
                mapping[tag] = tag
            return mapping, assignment_scores

        for tag, column in zip(valid_tags, self._maximum_weight_assignment(score_matrix)):
            mapping[tag] = column
        return mapping, assignment_scores

    @staticmethod
    def _maximum_weight_assignment(score_matrix: Sequence[Sequence[float]]) -> List[int]:
        """Solve a small rectangular one-to-one maximum-weight assignment exactly."""
        if not score_matrix:
            return []
        num_rows = len(score_matrix)
        num_columns = len(score_matrix[0])
        if num_rows > num_columns:
            raise ValueError("Cannot assign more t-SOT speakers than Sortformer columns.")
        if any(len(row) != num_columns for row in score_matrix):
            raise ValueError("speaker assignment score rows must have equal width.")

        # Sortformer has at most eight slots here, so an exact bitmask DP is clearer
        # and avoids adding a SciPy dependency just for this tiny assignment problem.
        states: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
        for row in score_matrix:
            next_states: Dict[int, Tuple[float, List[int]]] = {}
            for used_columns, (score_so_far, columns) in states.items():
                for column, score in enumerate(row):
                    if used_columns & (1 << column):
                        continue
                    next_mask = used_columns | (1 << column)
                    candidate = (score_so_far + score, columns + [column])
                    current = next_states.get(next_mask)
                    if current is None or candidate[0] > current[0]:
                        next_states[next_mask] = candidate
            states = next_states
        if not states:
            raise ValueError("No valid Sortformer speaker assignment exists.")
        return max(states.values(), key=lambda item: item[0])[1]

    @staticmethod
    def _resample_speaker_probs(speaker_probs: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Linearly resample raw Sortformer probabilities onto the CTC frame grid."""
        if speaker_probs.ndim != 2:
            raise ValueError(f"speaker_probs must have shape (T, S), got {tuple(speaker_probs.shape)}.")
        source_frames = speaker_probs.shape[0]
        if source_frames <= 0 or target_frames <= 0:
            raise ValueError("speaker and CTC frame counts must be positive.")
        if source_frames == target_frames:
            return speaker_probs
        if source_frames == 1:
            return speaker_probs.expand(target_frames, -1)

        positions = torch.linspace(0, source_frames - 1, target_frames, dtype=torch.float32)
        lower = positions.floor().to(dtype=torch.long)
        upper = positions.ceil().to(dtype=torch.long)
        fraction = (positions - lower.to(dtype=torch.float32)).unsqueeze(-1)
        return speaker_probs[lower] * (1.0 - fraction) + speaker_probs[upper] * fraction

    @staticmethod
    def _validate_alignment_mode(mode: str) -> str:
        mode = str(mode).lower()
        if mode not in {'parallel', 'serialized'}:
            raise ValueError(f"alignment_mode must be 'parallel' or 'serialized', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_assignment_mode(mode: str) -> str:
        mode = str(mode).lower()
        if mode not in {'optimal', 'identity'}:
            raise ValueError(f"speaker_assignment_mode must be 'optimal' or 'identity', got {mode!r}.")
        return mode

    @staticmethod
    def _single_recording_tensor(tensor: torch.Tensor, name: str, expected_ndim: int) -> torch.Tensor:
        """Accept ``(T, D)`` or ``(1, T, D)``, rejecting ambiguous batched input."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.ndim == expected_ndim:
            return tensor
        if tensor.ndim == expected_ndim + 1:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{name} has batch size {tensor.shape[0]}; align one recording per call so "
                    "timestamps cannot be associated with the wrong transcript."
                )
            return tensor[0]
        raise ValueError(
            f"{name} must have shape (T, D) or (1, T, D), but got {tuple(tensor.shape)}."
        )

    @staticmethod
    def _scalar_length(value: Any, name: str) -> int:
        """Read an integral single-item tensor / scalar length without hidden batching."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain exactly one length, got shape {tuple(value.shape)}.")
            value = value.detach().cpu().item()
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a scalar length, got {type(value).__name__}.") from error
        if not math.isfinite(numeric_value) or numeric_value != int(numeric_value):
            raise ValueError(f"{name} must be a finite integer, got {value!r}.")
        return int(numeric_value)

    @classmethod
    def _select_length(cls, lengths: Optional[Any], maximum: int, name: str) -> int:
        if maximum <= 0:
            raise ValueError(f"{name} maximum must be positive, got {maximum}.")
        if lengths is None:
            return maximum
        length = cls._scalar_length(lengths, name)
        if not 0 < length <= maximum:
            raise ValueError(f"{name} must be in [1, {maximum}], got {length}.")
        return length

    def _resolve_blank_id(self, ctc_vocab_size: int) -> int:
        """Validate the decoder/head blank convention against the supplied log-probs."""
        if self.blank_id is not None:
            blank_id = int(self.blank_id)
        elif self.ctc_decoder is not None:
            blank_id = int(self.ctc_decoder.num_classes_with_blank) - 1
        else:
            blank_id = ctc_vocab_size - 1
        if not 0 <= blank_id < ctc_vocab_size:
            raise ValueError(
                f"blank_id={blank_id} is outside the CTC class range [0, {ctc_vocab_size})."
            )
        if self.ctc_decoder is not None and self.ctc_decoder.num_classes_with_blank != ctc_vocab_size:
            raise ValueError(
                "CTC log-probability class count does not match the TransformerCTCDecoder: "
                f"{ctc_vocab_size} vs {self.ctc_decoder.num_classes_with_blank}."
            )
        tokenizer_vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        if callable(tokenizer_vocab_size):
            tokenizer_vocab_size = tokenizer_vocab_size()
        if tokenizer_vocab_size is not None and int(tokenizer_vocab_size) > blank_id:
            raise ValueError(
                "Tokenizer vocabulary is larger than the non-blank CTC vocabulary: "
                f"{tokenizer_vocab_size} vs blank_id={blank_id}."
            )
        return blank_id

    def _resolve_ctc_frame_seconds(self, ctc_length: int, audio_duration: Optional[float]) -> float:
        if audio_duration is not None:
            return float(audio_duration) / ctc_length
        if self.ctc_frame_seconds is not None:
            return float(self.ctc_frame_seconds)

        pee_encoder = getattr(self.encoder, 'encoder', self.encoder)
        subsampling_factor = getattr(pee_encoder, 'subsampling_factor', None)
        if subsampling_factor is None and hasattr(pee_encoder, 'pee'):
            speech_expert = getattr(pee_encoder.pee, 'experts', {}).get('speech', None)
            subsampling_factor = getattr(speech_expert, 'subsampling_factor', None)
        if subsampling_factor is None:
            subsampling_factor = 1
        return self.input_frame_seconds * float(subsampling_factor)

    def _resolve_sortformer_frame_seconds(
        self,
        sortformer_length: Optional[int],
        audio_duration: Optional[float],
        ctc_step_seconds: float,
    ) -> Optional[float]:
        if sortformer_length is None:
            return None
        if audio_duration is not None:
            return float(audio_duration) / sortformer_length
        if self.sortformer_frame_seconds is not None:
            return float(self.sortformer_frame_seconds)
        return ctc_step_seconds
