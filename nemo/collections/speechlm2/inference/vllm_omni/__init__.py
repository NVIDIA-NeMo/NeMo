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

"""NemotronDuplexH + EarTTS model/pipeline package for vLLM-Omni.

This package lives inside NeMo and plugs into ``vllm-omni`` at runtime
through :func:`register_nemo_voicechat`. The function is auto-invoked in
every vllm-omni subprocess via the ``vllm_omni.general_plugins`` entry
point declared in NeMo's ``pyproject.toml`` (vllm-omni uses ``spawn`` for
stage children, so the entry-point hook is required — PYTHONPATH alone
is not enough).

The plugin registers three things:

* HF config ``"eartts"`` → :class:`EarTTSConfig`
* Model arch ``"NemotronDuplexHForCausalLM"`` →
  :class:`nemo.collections.speechlm2.inference.vllm_omni.nemotron_duplex_h.nemotron_duplex_h.NemotronDuplexHForCausalLM`
* Model arch ``"EarTTSForCausalLM"`` →
  :class:`nemo.collections.speechlm2.inference.vllm_omni.eartts.eartts.EarTTSForCausalLM`
* One-stage pipelines ``model_type = "nemotron_voicechat"`` and ``"eartts"``.

Bundled deploy YAMLs for the independent engines live under ``deploy/``.
"""

from __future__ import annotations

from pathlib import Path


def default_deploy_yaml() -> Path:
    """Return the absolute path to the bundled ``nemotron_voicechat.yaml``."""
    return Path(__file__).resolve().parent / "deploy" / "nemotron_voicechat.yaml"


def default_eartts_deploy_yaml() -> Path:
    """Return the absolute path to the bundled single-stage ``eartts.yaml``."""
    return Path(__file__).resolve().parent / "deploy" / "eartts.yaml"


from nemo.collections.speechlm2.inference.vllm_omni.register import (
    register_nemo_voicechat,
)

__all__ = [
    "default_deploy_yaml",
    "default_eartts_deploy_yaml",
    "register_nemo_voicechat",
]
