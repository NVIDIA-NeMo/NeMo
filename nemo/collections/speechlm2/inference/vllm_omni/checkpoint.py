# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Building and reading the vLLM-Omni wrapper checkpoint.

``AsyncOmni(model=...)`` wants a directory laid out as::

    <wrapper>/
        config.json         # {"model_type": "nemotron_voicechat"}
        nemotron/           # converted NemotronDuplexH checkpoint
        eartts/             # converted EarTTS checkpoint + speaker_latents/

Everything that reads or writes that layout lives here: conversion, the source
fingerprint that makes incremental builds safe, the small config patches
applied before an engine starts, and the two loaders that read values back out
of a built wrapper. Nothing here imports vLLM, so it can be exercised without
an engine.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from nemo.utils import logging

NEMOTRON_SUBDIR = "nemotron"
EARTTS_SUBDIR = "eartts"
_WRAPPER_CONFIG = {"model_type": "nemotron_voicechat"}
_SOURCE_MANIFEST = ".nemo_source.json"


def _checkpoint_fingerprint(model_path: str) -> dict[str, Any]:
    """Cheap content identity for safe incremental wrapper construction."""
    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No safetensors weights found in checkpoint: {root}")
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "weights": [{"name": path.name, "size": path.stat().st_size} for path in weights],
    }


def _read_source_manifest(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def build_wrapper_checkpoint(
    model_path: str,
    wrapper_dir: str | None = None,
    *,
    nemotron_dtype: str = "float32",
    eartts_precompute_batch_size: int = 256,
    include_nemotron: bool = True,
    include_eartts: bool = True,
) -> str:
    """Build a wrapper checkpoint directory consumed by ``AsyncOmni(model=...)``.

    Layout::

        <wrapper>/
            config.json         # {"model_type": "nemotron_voicechat"}
            nemotron/           # converted NemotronDuplexH checkpoint
            eartts/             # converted EarTTS checkpoint + speaker_latents/

    Args:
        model_path: Path to the source NemotronVoiceChat HF-format checkpoint
            directory (``config.json`` + ``model.safetensors``).
        wrapper_dir: Where to put the wrapper directory. Defaults to
            ``<tempdir>/<basename>_vllm_omni_wrapper``, where ``<tempdir>`` is
            :func:`tempfile.gettempdir` and so honours ``$TMPDIR``.
        nemotron_dtype: dtype for the converted Nemotron checkpoint.
        eartts_precompute_batch_size: batch size used when baking out the
            EarTTS subword-encoder lookup table.

    Returns:
        Absolute path to the wrapper directory. If the wrapper directory
        already exists and looks complete, the existing one is returned and
        nothing is re-converted.
    """
    src = os.path.normpath(model_path)
    if wrapper_dir is None:
        wrapper_dir = os.path.join(tempfile.gettempdir(), os.path.basename(src) + "_vllm_omni_wrapper")
    wrapper_dir = os.path.abspath(wrapper_dir)

    nemotron_dir = os.path.join(wrapper_dir, NEMOTRON_SUBDIR)
    eartts_dir = os.path.join(wrapper_dir, EARTTS_SUBDIR)
    config_path = os.path.join(wrapper_dir, "config.json")
    manifest_path = os.path.join(wrapper_dir, _SOURCE_MANIFEST)
    source_fingerprint = _checkpoint_fingerprint(src)

    nemotron_ready = (
        os.path.isdir(nemotron_dir)
        and os.path.isfile(os.path.join(nemotron_dir, "config.json"))
        and os.path.isfile(os.path.join(nemotron_dir, "model.safetensors"))
    )
    eartts_ready = (
        os.path.isdir(eartts_dir)
        and os.path.isfile(os.path.join(eartts_dir, "config.json"))
        and os.path.isfile(os.path.join(eartts_dir, "model.safetensors"))
    )
    config_ready = os.path.isfile(config_path)

    if not include_nemotron and not include_eartts:
        raise ValueError("At least one vLLM-Omni component must be requested")

    manifest = _read_source_manifest(manifest_path)
    if manifest is None:
        adding_to_unverified_partial_wrapper = (include_nemotron and not nemotron_ready and eartts_ready) or (
            include_eartts and not eartts_ready and nemotron_ready
        )
        if adding_to_unverified_partial_wrapper:
            raise ValueError(
                f"Cannot safely add a component to wrapper {wrapper_dir}: "
                f"{_SOURCE_MANIFEST} is missing, so the existing component's "
                "source checkpoint cannot be verified. Use a fresh wrapper_dir."
            )
    elif manifest.get("source") != source_fingerprint:
        logging.warning(
            "Wrapper source checkpoint changed; rebuilding converted components in %s",
            wrapper_dir,
        )
        for component_dir in (nemotron_dir, eartts_dir):
            if os.path.isdir(component_dir):
                shutil.rmtree(component_dir)
        nemotron_ready = False
        eartts_ready = False
        manifest = None
    else:
        if include_nemotron and nemotron_ready and manifest.get("nemotron", {}).get("dtype") != nemotron_dtype:
            shutil.rmtree(nemotron_dir)
            nemotron_ready = False
        if (
            include_eartts
            and eartts_ready
            and manifest.get("eartts", {}).get("precompute_batch_size") != eartts_precompute_batch_size
        ):
            shutil.rmtree(eartts_dir)
            eartts_ready = False

    if (not include_nemotron or nemotron_ready) and (not include_eartts or eartts_ready) and config_ready:
        if manifest is None:
            # Wrapper is complete but carries no source manifest. Stamp one
            # now: no component is being added, so this cannot mix checkpoints.
            logging.warning(
                "Adopting vLLM-Omni wrapper without source manifest: %s",
                wrapper_dir,
            )
            adopted_manifest: dict[str, Any] = {"source": source_fingerprint}
            if nemotron_ready:
                adopted_manifest["nemotron"] = {"dtype": nemotron_dtype}
            if eartts_ready:
                adopted_manifest["eartts"] = {"precompute_batch_size": eartts_precompute_batch_size}
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(adopted_manifest, fh, indent=2, sort_keys=True)
        logging.info(f"Reusing existing vllm-omni wrapper checkpoint at {wrapper_dir}")
        return wrapper_dir

    os.makedirs(wrapper_dir, exist_ok=True)

    if include_nemotron and not nemotron_ready:
        # Convert the Nemotron LLM with the existing DuplexSTT converter.
        # That converter's output (HF NemotronH config + filtered weights)
        # is consumed directly by NemotronDuplexHForCausalLM's WeightsMapper.
        if os.path.isdir(nemotron_dir):
            shutil.rmtree(nemotron_dir)
        logging.info(f"Converting Nemotron LLM into {nemotron_dir} ...")
        from nemo.collections.speechlm2.inference.vllm_omni.scripts.convert_duplex_stt_checkpoint import (
            convert_to_vllm_format as convert_nemotron,
        )

        convert_nemotron(
            checkpoint_path=src,
            output_dir=nemotron_dir,
            dtype=nemotron_dtype,
        )

    if include_eartts and not eartts_ready:
        if os.path.isdir(eartts_dir):
            shutil.rmtree(eartts_dir)
        logging.info(f"Converting EarTTS into {eartts_dir} ...")
        from nemo.collections.speechlm2.inference.vllm_omni.scripts.convert_duplex_eartts_checkpoint import (
            convert_to_vllm_format as convert_eartts,
        )

        convert_eartts(
            outdir=eartts_dir,
            config=os.path.join(src, "config.json"),
            model_path=os.path.join(src, "model.safetensors"),
            precompute_batch_size=eartts_precompute_batch_size,
        )

    if not config_ready:
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(_WRAPPER_CONFIG, fh, indent=2)

    completed_manifest: dict[str, Any] = {"source": source_fingerprint}
    if os.path.isfile(os.path.join(nemotron_dir, "model.safetensors")):
        completed_manifest["nemotron"] = {"dtype": nemotron_dtype}
    if os.path.isfile(os.path.join(eartts_dir, "model.safetensors")):
        completed_manifest["eartts"] = {"precompute_batch_size": eartts_precompute_batch_size}
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(completed_manifest, fh, indent=2, sort_keys=True)

    return wrapper_dir


def write_nemotron_inference_overrides(wrapper_dir: str, overrides: dict[str, Any]) -> None:
    """Update inference settings in the converted Nemotron ``config.json``.

    ``NemotronDuplexHForCausalLM`` reads some settings off its HF config at
    load time -- the user-channel logit boosts, which cannot be delivered per
    request because the ASR head's logits never reach vLLM's sampler. Those are
    still chosen per run in the inference yaml, so the small JSON is rewritten
    here before the stage child starts, rather than re-converting weights.

    Keys whose value is ``None`` are removed, so clearing a boost in the config
    clears it in the engine too.
    """
    config_path = os.path.join(wrapper_dir, "nemotron", "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)

    changed = False
    for key, value in overrides.items():
        if value is None:
            if config.pop(key, None) is not None:
                changed = True
        elif config.get(key) != value:
            config[key] = value
            changed = True
    if not changed:
        return

    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    logging.info(f"Updated Nemotron inference overrides in {config_path}: {overrides}")


def load_speaker_latent(eartts_dir: str, speaker_name: str) -> torch.Tensor:
    """Load ``<eartts_dir>/speaker_latents/<speaker_name>.pt`` (saved by the
    EarTTS converter) and return a contiguous CPU tensor of shape
    ``(Tref, hidden_size)``.
    """
    latents_dir = os.path.join(eartts_dir, "speaker_latents")
    latent_path = os.path.join(latents_dir, f"{speaker_name}.pt")
    if not os.path.isfile(latent_path):
        available = []
        if os.path.isdir(latents_dir):
            available = sorted(
                os.path.splitext(name)[0] for name in os.listdir(latents_dir) if name.endswith(".pt")
            )
        raise FileNotFoundError(
            f"Speaker latent for '{speaker_name}' not found at {latent_path}. "
            f"Registered speakers: {available or '(none)'}. "
            "Pick a speaker_name present in the EarTTS checkpoint, or re-run the "
            "EarTTS converter on a checkpoint that contains the requested "
            "audio_prompt_latents."
        )
    latent = torch.load(latent_path, weights_only=False)
    if isinstance(latent, torch.Tensor) and latent.dim() == 3:
        latent = latent[0]
    if not isinstance(latent, torch.Tensor) or latent.dim() != 2:
        raise ValueError(
            f"Expected speaker latent at {latent_path} to be a 2-D tensor [Tref, hidden], "
            f"got {type(latent).__name__} with shape "
            f"{tuple(latent.shape) if isinstance(latent, torch.Tensor) else 'n/a'}"
        )
    return latent.detach().to(torch.float32).cpu().contiguous()


def compute_prefill_len(model_dir: str, system_prompt: str) -> int:
    """Length of the prefill chunk fed to NemotronDuplexH for a
    given system prompt. Mirrors the in-model tokenization:
    ``[BOS] + tokenizer.encode(prompt) + [EOS]``.
    """
    from transformers import AutoTokenizer

    from nemo.collections.speechlm2.inference.vllm_omni.nemotron_duplex_h.nemotron_duplex_h import (
        NemotronDuplexHForCausalLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return NemotronDuplexHForCausalLM.compute_prefix_len(tokenizer, system_prompt)
