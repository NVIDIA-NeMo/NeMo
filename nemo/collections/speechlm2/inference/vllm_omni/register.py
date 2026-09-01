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

"""Runtime registration of the NemotronDuplexH + EarTTS models and the
``nemotron_voicechat`` pipeline with vLLM and vLLM-Omni.

Call :func:`register_nemo_voicechat` once, before constructing
``vllm_omni.AsyncOmni`` / ``vllm_omni.Omni``. It is idempotent.

NeMo's ``pyproject.toml`` exposes this function as an entry point in **both**
the ``vllm.general_plugins`` and ``vllm_omni.general_plugins`` groups, so a
plugin loader invokes it automatically in every process (orchestrator + each
spawned ``StageEngineCoreProc`` child + workers). vllm-omni uses
``multiprocessing`` with start method ``spawn`` for stage children, so spawned
processes do NOT inherit Python state from the parent — registering in the
parent alone is not enough, which is why a plugin entry point is the correct
hook. vLLM's group is the one that matters in the parent: vllm-omni resolves
the pipeline for ``model_type`` while constructing ``AsyncOmniEngine``, before
it loads its own plugin group.

Being in vLLM's group means this also runs in processes that have no vllm-omni
installed at all (an ordinary ``vllm serve``), so the vllm-omni half is
optional and skipped when the import fails.

Three things get registered:

1. ``EarTTSConfig`` with ``transformers.AutoConfig`` (so ``AutoConfig.from_pretrained``
   resolves ``model_type = "eartts"``) and with vLLM's ``_CONFIG_REGISTRY``.
2. Model architectures (``NemotronDuplexHForCausalLM``, ``EarTTSForCausalLM``)
   with both ``vllm.model_executor.models.ModelRegistry`` and
   ``vllm_omni.model_executor.models.OmniModelRegistry``. The two registries
   serve different lookups inside vLLM-Omni so both have to know about the
   new arches.
3. The one-stage :data:`NEMOTRON_VOICECHAT_PIPELINE` and
   :data:`EARTTS_PIPELINE` with ``vllm_omni.config.register_pipeline``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_PKG = "nemo.collections.speechlm2.inference.vllm_omni"


_ARCH_MAP: dict[str, tuple[str, str]] = {
    # arch_name -> (module_path, class_name)
    "NemotronDuplexHForCausalLM": (
        f"{_PKG}.nemotron_duplex_h.nemotron_duplex_h",
        "NemotronDuplexHForCausalLM",
    ),
    "EarTTSForCausalLM": (
        f"{_PKG}.eartts.eartts",
        "EarTTSForCausalLM",
    ),
}


_registered = False


def register_nemo_voicechat() -> None:
    """Register the NemotronDuplexH + EarTTS models, ``EarTTSConfig``,
    and the ``nemotron_voicechat`` pipeline with vLLM / vLLM-Omni.

    Safe to call multiple times.
    """
    global _registered
    if _registered:
        return

    _register_hf_configs()
    _register_model_archs()
    omni = _register_omni()

    _registered = True
    logger.info(
        "nemo_voicechat: registered NemotronDuplexH + EarTTS%s.",
        " + split nemotron_voicechat/eartts pipelines"
        if omni
        else " (vllm-omni absent, pipelines not registered)",
    )


def _register_hf_configs() -> None:
    from nemo.collections.speechlm2.inference.vllm_omni.eartts.configuration_eartts import (
        EarTTSConfig,
        register_eartts_config,
    )

    register_eartts_config()

    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except ImportError:
        _CONFIG_REGISTRY = None

    if _CONFIG_REGISTRY is not None and EarTTSConfig.model_type not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY[EarTTSConfig.model_type] = EarTTSConfig


def _register_model_archs() -> None:
    # vLLM's public model registry — needed for ``ModelRegistry.is_*``
    # checks and for the arch → module resolution used outside of
    # OmniModelConfig.
    from vllm.model_executor.models import ModelRegistry

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (module_path, class_name) in _ARCH_MAP.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, f"{module_path}:{class_name}")


def _register_omni() -> bool:
    """Register with vLLM-Omni, if it is installed. Returns whether it was."""
    try:
        # OmniModelRegistry is the mirror registry ``OmniModelConfig.registry``
        # returns, used to load the model class for each pipeline stage.
        from vllm_omni.config import register_pipeline
        from vllm_omni.model_executor.models import OmniModelRegistry
    except ImportError:
        return False

    omni_supported = OmniModelRegistry.get_supported_archs()
    for arch, (module_path, class_name) in _ARCH_MAP.items():
        if arch not in omni_supported:
            OmniModelRegistry.register_model(arch, f"{module_path}:{class_name}")

    from nemo.collections.speechlm2.inference.vllm_omni.nemotron_voicechat.pipeline import (
        NEMOTRON_VOICECHAT_PIPELINE,
    )
    from nemo.collections.speechlm2.inference.vllm_omni.eartts.pipeline import (
        EARTTS_PIPELINE,
    )

    register_pipeline(NEMOTRON_VOICECHAT_PIPELINE)
    register_pipeline(EARTTS_PIPELINE)
    return True


__all__ = ["register_nemo_voicechat"]
