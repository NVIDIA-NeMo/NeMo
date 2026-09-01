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

"""What the ``s2s`` config resolves to, and the process state it implies.

Kept out of the wrapper so that answering "which engines did this config
select?" does not require constructing a wrapper, loading a checkpoint or
importing torch's model stack. The builder, the wrapper and the tests all read
their answers from here.
"""

from collections.abc import Mapping

from nemo.collections.speechlm2.parts.precision import inference_precision, inference_precision_in_effect

NATIVE = "native"
VLLM_OMNI = "vllm_omni"

SUPPORTED_S2S_ENGINE_TYPES = frozenset({NATIVE, VLLM_OMNI})


def _component_engine(model_cfg: Mapping, key: str) -> str:
    value = model_cfg.get(key, NATIVE)
    if value is None:
        return NATIVE
    return str(value).lower()


def resolve_engine_types(model_cfg: Mapping) -> tuple[str, str]:
    """Read independent LLM and TTS engines from *model_cfg*.

    Missing or null keys default to ``native``. ``engine_type`` is not a
    config key: if it is set, this raises rather than treating it as a
    shared default.
    """
    leftover = model_cfg.get("engine_type", None)
    if leftover is not None:
        raise ValueError(
            "`engine_type` is not a config key. Set `llm_engine_type` and "
            "`tts_engine_type` independently; each must be one of "
            f"{sorted(SUPPORTED_S2S_ENGINE_TYPES)}."
        )
    llm = _component_engine(model_cfg, "llm_engine_type")
    tts = _component_engine(model_cfg, "tts_engine_type")
    invalid = {
        name: value
        for name, value in (("llm_engine_type", llm), ("tts_engine_type", tts))
        if value not in SUPPORTED_S2S_ENGINE_TYPES
    }
    if invalid:
        values = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise ValueError(
            f"Unsupported S2S engine selection ({values}); expected one of " f"{sorted(SUPPORTED_S2S_ENGINE_TYPES)}."
        )
    return llm, tts


def reject_unsupported_determinism(llm_engine_type: str, tts_engine_type: str, deterministic: bool) -> None:
    """Raise if ``deterministic`` was asked for alongside a vLLM component.

    vLLM's custom kernels (PagedAttention, FlashAttention) have no
    deterministic mode, so this cannot be honoured rather than merely being
    slower. Same no-silent-no-op contract as the rest of the config checks.
    """
    if not deterministic:
        return
    vllm_components = [
        name
        for name, value in (("llm_engine_type", llm_engine_type), ("tts_engine_type", tts_engine_type))
        if value == VLLM_OMNI
    ]
    if vllm_components:
        raise ValueError(
            "`deterministic` is not compatible with vLLM engines because vLLM uses custom "
            "CUDA kernels (PagedAttention, FlashAttention) that do not support deterministic mode. "
            f"Selected vLLM components: {', '.join(vllm_components)}. "
            "Use native engines for deterministic inference."
        )


def native_weight_skip_prefixes(llm_engine_type: str, tts_engine_type: str) -> set[str]:
    """Checkpoint prefixes not needed by the selected component backends."""
    prefixes = {"stt_model.rnnt_decoder.", "stt_model.rnnt_joint."}
    if llm_engine_type == VLLM_OMNI:
        prefixes.add("stt_model.llm.")
    if tts_engine_type == VLLM_OMNI:
        prefixes.add("tts_model.tts_model.")
    return prefixes


def _precision_settings(model_cfg: Mapping) -> dict:
    """The three torch precision switches *model_cfg* asks for."""
    return {
        "allow_tf32": bool(model_cfg.get("allow_tf32", True)),
        "matmul_precision": str(model_cfg.get("matmul_precision", "medium")),
        "deterministic": bool(model_cfg.get("deterministic", False)),
    }


def precision_matches_cfg(model_cfg: Mapping) -> bool:
    """Whether the process is already configured the way *model_cfg* asks.

    What entry points check before loading weights, so a forgotten
    :func:`inference_precision_from_cfg` is reported rather than silently
    changing the numbers.
    """
    return inference_precision_in_effect(**_precision_settings(model_cfg))


def inference_precision_from_cfg(model_cfg: Mapping):
    """Scope the process-wide torch settings *model_cfg* implies.

    The switches must be in effect before any weights load and stay on for the
    whole run, but they are process globals, so they are restored on exit
    rather than left set. That keeps a deterministic run from changing every
    later computation in the process.

    The builder requires this scope and does not enter it. Typical caller::

        with inference_precision_from_cfg(cfg.s2s):
            pipeline = S2SPipelineBuilder.build_pipeline(cfg)
            try:
                pipeline.run(...)
            finally:
                pipeline.shutdown()
    """
    llm_engine_type, tts_engine_type = resolve_engine_types(model_cfg)
    settings = _precision_settings(model_cfg)
    reject_unsupported_determinism(llm_engine_type, tts_engine_type, settings["deterministic"])
    return inference_precision(**settings)
