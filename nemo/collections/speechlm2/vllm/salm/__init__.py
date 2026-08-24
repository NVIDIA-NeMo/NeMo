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

"""vLLM plugin registration for NeMo Speech LM (SALM) models.

Registers ``NeMoSpeechLMConfig`` and the single
``NeMoSpeechLMForConditionalGeneration`` model class with vLLM via the
``vllm.general_plugins`` entry point.

A single model class covers every supported backbone family (standard
decoder-only LLMs like Qwen3, hybrid Mamba+MoE like NemotronH).
Backbone-specific behavior is selected at instantiation time.
"""

import logging

_PKG = "nemo.collections.speechlm2.vllm.salm"
_LOG = logging.getLogger(__name__)


def _patch_vllm_for_nemo_speechlm_mtp() -> None:
    """Extend vLLM's speculative-decoding framework to support nemo_speechlm MTP.

    Releases without ``MTPModelTypes`` return without patching so the ordinary
    SpeechLM target model remains usable. Otherwise, three patches are applied:

    1. ``MTPModelTypes`` — the Literal type that guards the MTP detection
       branch in ``SpeculativeConfig.__post_init__`` is extended to include
       ``"nemo_speechlm_mtp"``.

    2. ``SpeculativeConfig.hf_config_override`` — the static method that
       rewrites the draft-model HF config is wrapped to detect
       ``nemo_speechlm`` checkpoints that carry MTP heads (``mtp.enabled``
       and ``mtp.num_nextn_predict_layers > 0``) and redirect them to the
       ``NeMoSpeechLMMTPModel`` architecture with the right ``n_predict``.

    3. ``ModelRegistry`` — ``NeMoSpeechLMMTPModel`` is registered so that
       vLLM can resolve and instantiate it as the draft model.
    """
    from typing import Literal, get_args

    import vllm.config.speculative as _spec_mod
    from vllm.config.speculative import SpeculativeConfig

    # MTP support was added incrementally across vLLM releases. Keep the
    # ordinary SpeechLM target-model plugin usable on releases that do not yet
    # expose this type guard; speculative decoding will remain unavailable and
    # vLLM will report that if the user tries to enable it.
    if not hasattr(_spec_mod, "MTPModelTypes"):
        _LOG.warning(
            "This vLLM release does not expose MTPModelTypes; NeMo SpeechLM "
            "will be registered without MTP speculative-decoding support."
        )
        return

    # Extend vLLM's recognized MTP model types.
    old_args = get_args(_spec_mod.MTPModelTypes)
    if "nemo_speechlm_mtp" not in old_args:
        _spec_mod.MTPModelTypes = Literal[old_args + ("nemo_speechlm_mtp",)]

    # Route SpeechLM MTP checkpoints through SpeculativeConfig.hf_config_override.
    current_override = SpeculativeConfig.hf_config_override
    if not getattr(current_override, "_nemo_speechlm_mtp_override", False):
        original_override = current_override

        def _patched_override(hf_config):
            if hf_config.model_type == "nemo_speechlm":
                mtp_cfg = getattr(hf_config, "mtp", None)
                if not isinstance(mtp_cfg, dict):
                    mtp_cfg = {}
                # Match SALMAutomodel's training defaults exactly: merely
                # retaining a recipe depth does not enable MTP, while an enabled
                # block with no explicit depth constructs one logical head.
                mtp_enabled = bool(mtp_cfg.get("enabled", False))
                n_predict = mtp_cfg.get("num_nextn_predict_layers", 1 if mtp_enabled else 0)
                if mtp_enabled and n_predict > 0:
                    use_repeated_layer = bool(mtp_cfg.get("use_repeated_layer", False))
                    if n_predict > 1 and not use_repeated_layer:
                        raise ValueError(
                            f"NeMo SpeechLM MTP with {n_predict} distinct head layers is not "
                            f"supported: vLLM's NemotronHMultiTokenPredictor builds a single "
                            f"physical MTP layer and reuses it every speculative step. Only "
                            f"checkpoints trained with mtp.use_repeated_layer=true match that "
                            f"execution model."
                        )
                    hf_config.model_type = "nemo_speechlm_mtp"
                    hf_config.update(
                        {
                            # Size of the physical MTP block that vLLM reuses. A
                            # repeated-layer checkpoint ships one shared head even
                            # when it was trained for multiple next-token positions,
                            # so arbitrary inference K values must be multiples of 1.
                            # Consequently vLLM defaults to K=1 when K is omitted;
                            # callers should set num_speculative_tokens explicitly.
                            "n_predict": 1,
                            # Physical MTP prediction steps to instantiate. Repeated-layer checkpoints
                            # ship one shared step (one mtp.layers.* module per hybrid-pattern character)
                            # that is reapplied every speculative iteration, exactly as vLLM drives its
                            # MTP draft. This also shadows the backbone text_config's
                            # num_nextn_predict_layers (e.g. 4), which would otherwise trip the
                            # single-step assert in NemotronHMultiTokenPredictor.
                            "num_nextn_predict_layers": 1,
                            "architectures": ["NeMoSpeechLMMTPModel"],
                        }
                    )
                    return hf_config
            return original_override(hf_config)

        _patched_override._nemo_speechlm_mtp_override = True
        SpeculativeConfig.hf_config_override = staticmethod(_patched_override)

    # Register the SpeechLM MTP draft architecture with vLLM.
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMMTPModel",
        f"{_PKG}.mtp:NeMoSpeechLMMTP",
    )


def register():
    """Register the NeMo Speech LM model and config with vLLM."""
    from transformers import AutoConfig

    from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig

    AutoConfig.register("nemo_speechlm", NeMoSpeechLMConfig)

    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY["nemo_speechlm"] = NeMoSpeechLMConfig

    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "NeMoSpeechLMForConditionalGeneration",
        f"{_PKG}.model:NeMoSpeechLMForConditionalGeneration",
    )

    _patch_vllm_for_nemo_speechlm_mtp()
