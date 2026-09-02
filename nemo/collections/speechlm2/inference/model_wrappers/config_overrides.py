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

"""Which ``s2s`` config keys each backend honours, and where they land.

Two kinds of settings arrive in the ``s2s`` config block:

* **Model-consumed keys** are read by shared model code off its own ``cfg``
  (the logit boosts in ``DuplexSTTModel``, ``force_turn_taking``, the EarTTS
  generation config), so the wrapper cannot pass them as arguments. They are
  bridged into the relevant model config by :func:`apply_model_cfg_overrides`,
  which is also how the streaming path inherits the offline path's behaviour
  for the same key.
* **Wrapper-consumed knobs** (``decode_audio``, ``top_p``, ``use_llm_cache``,
  ...) are read straight into wrapper attributes. Nothing is bridged for
  them, but ignored keys are still reported.

This module is the single answer to "can I set this, where does it land, and
does the selected backend honour it". :func:`apply_model_cfg_overrides` bridges
the model-consumed keys and warns about everything the selected backends will
ignore.

``deterministic`` is deliberately not here: it is a process-global torch
setting rather than a model or wrapper key, and has to be settled before any
weights load. See
:func:`~nemo.collections.speechlm2.inference.model_wrappers.engine_selection.reject_unsupported_determinism`.
"""

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import VLLM_OMNI
from nemo.utils import logging

# Component that owns the config object each key is read from. The component
# also selects the ``*_engine_type`` that decides whether the key takes effect.
LLM = "llm"  # -> model.stt_model.cfg  (DuplexSTTModel)
TTS = "tts"  # -> model.tts_model.cfg  (DuplexEARTTS)

LLM_KEYS = (
    # Agent (text) and user (ASR) side logit boosts, read by the DuplexSTTModel
    # heads; then forced turn-taking, read by _maybe_apply_forced_turn_taking.
    "inference_pad_boost",
    "inference_bos_boost",
    "inference_eos_boost",
    "inference_user_pad_boost",
    "inference_user_bos_boost",
    "inference_user_eos_boost",
    "force_turn_taking",
    "force_turn_taking_threshold",
    "force_turn_taking_pad_window",
)

TTS_KEYS = (
    # Both backends implement EOS -> codec silence inside the model:
    # DuplexEARTTS.infer_codes_one_step (flag-gated, defaults to True) and the
    # vLLM EarTTS preprocess (unconditional). See VLLM_FORCES_TRUE.
    "inference_force_speech_silence_on_eos",
)

COMPONENT_OF = {**{key: LLM for key in LLM_KEYS}, **{key: TTS for key in TTS_KEYS}}

# --- Support tables -------------------------------------------------------
# Every key not listed below works on both backends. Each entry is
# ``key -> (component, why)``; the component picks which ``*_engine_type``
# decides, and the reason is quoted verbatim to the user.

# The run is correct, but the setting does nothing: warn.
VLLM_IGNORES = {
    "use_llm_cache": (LLM, "vLLM always keeps a paged KV cache"),
    "use_tts_torch_compile": (TTS, "vLLM compiles inside the engine"),
    "use_tts_subword_cache": (
        TTS,
        "the subword table is baked in at checkpoint conversion, so it is always in effect",
    ),
}

# These boolean flags request an optimization only when enabled. Their false
# values are no-ops.
_IGNORED_ENABLE_FLAGS = frozenset({"use_llm_cache", "use_tts_torch_compile", "use_tts_subword_cache"})

# vLLM does this unconditionally: it can honour True but not False. Warn only
# when False was asked for.
VLLM_FORCES_TRUE = {
    "inference_force_speech_silence_on_eos": (
        TTS,
        "the converted EarTTS always substitutes codec silence when the incoming text token is EOS",
    ),
}


def _selected(component: str, llm_engine_type: str, tts_engine_type: str) -> str:
    return str(llm_engine_type if component == LLM else tts_engine_type).lower()


def _target_cfg(model, component: str):
    """Model config that *component*'s code reads its settings from."""
    if component == LLM:
        submodel = model.stt_model
    elif component == TTS:
        submodel = model.tts_model
    else:
        raise ValueError(f"Unknown component {component!r}; expected {LLM!r} or {TTS!r}")
    return None if submodel is None else submodel.cfg


def apply_model_cfg_overrides(
    model,
    model_cfg: Mapping,
    *,
    llm_engine_type: str,
    tts_engine_type: str,
) -> dict[str, Any]:
    """Bridge the model-consumed keys into *model*, then report what is ignored.

    Keys absent from *model_cfg* are left alone, so whatever the checkpoint
    carries stays in effect. Keys the selected backend cannot honour are
    reported once per component instead of silently doing nothing.

    Returns:
        The effective value of every model-consumed key after bridging, for
        logging.
    """
    for key, component in COMPONENT_OF.items():
        value = model_cfg.get(key, None)
        if value is None:
            continue
        target = _target_cfg(model, component)
        if target is None:
            logging.warning(f"Ignoring `{key}`: this checkpoint has no {component} component to apply it to.")
            continue
        OmegaConf.update(target, key, value, force_add=True)

    effective: dict[str, Any] = {}
    for key, component in COMPONENT_OF.items():
        target = _target_cfg(model, component)
        effective[key] = None if target is None else target.get(key, None)

    # Model-consumed keys are checked post-bridge, so the reported value is the
    # one the model will actually read; wrapper knobs come straight from config.
    def value_of(key: str) -> Any:
        return effective[key] if key in COMPONENT_OF else model_cfg.get(key, None)

    _warn_unsupported(value_of, llm_engine_type=llm_engine_type, tts_engine_type=tts_engine_type)
    return effective


def _warn_unsupported(value_of, *, llm_engine_type: str, tts_engine_type: str) -> None:
    """Report set keys the selected backends ignore, grouped by component."""
    unsupported: dict[str, list[str]] = {}

    for key, (component, why) in VLLM_IGNORES.items():
        if _selected(component, llm_engine_type, tts_engine_type) != VLLM_OMNI:
            continue
        value = value_of(key)
        if value is None or (key in _IGNORED_ENABLE_FLAGS and not value):
            continue
        unsupported.setdefault(f"{component}_engine_type={VLLM_OMNI}", []).append(f"{key} ({why})")

    for selection, keys in unsupported.items():
        listed = ", ".join(sorted(keys))
        logging.warning(
            f"These settings have no effect with {selection}: {listed}. "
            "Select the native engine for that component, or remove them from the config "
            "so it reflects what the run actually does."
        )

    for key, (component, why) in VLLM_FORCES_TRUE.items():
        if _selected(component, llm_engine_type, tts_engine_type) != VLLM_OMNI:
            continue
        if value_of(key) is not False:
            continue
        logging.warning(
            f"`{key}=False` is not supported by {component}_engine_type={VLLM_OMNI}: {why}. "
            "Expect it to behave as True."
        )
