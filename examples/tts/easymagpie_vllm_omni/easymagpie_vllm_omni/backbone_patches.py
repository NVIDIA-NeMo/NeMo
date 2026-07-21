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
"""Runtime fixes for the EasyMagpie Nemotron-H backbone."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.v1.attention.backends.mamba_attn as _mamba_attn
from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_mamba_streaming_decode() -> None:
    """Classify one-token stream extensions as decodes.

    Decode CUDA graphs require decode metadata and a refreshed Mamba state
    index. Multi-token context inputs remain prefills. Streaming callers must
    generate exactly one new token per step.
    """
    orig = _mamba_attn.split_decodes_and_prefills
    if getattr(orig, "_easymagpie_patched", False):
        return

    def patched(
        common_attn_metadata,
        decode_threshold: int = 1,
        require_uniform: bool = False,
        treat_short_extends_as_decodes: bool = True,
    ):
        return orig(
            common_attn_metadata,
            decode_threshold=decode_threshold,
            require_uniform=require_uniform,
            treat_short_extends_as_decodes=True,
        )

    patched._easymagpie_patched = True
    _mamba_attn.split_decodes_and_prefills = patched
    logger.info("Mamba streaming-decode classification patch installed")


class _SiluActivation(nn.Module):
    """Module wrapper for ``F.silu``."""

    def forward(self, x):
        return F.silu(x)


def patch_silu_shared_experts(backbone) -> int:
    """Use the checkpoint's SiLU activation on every shared expert."""
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        se = getattr(mixer, "shared_experts", None)
        if se is None:
            continue
        se.act_fn = _SiluActivation()
        patched += 1
    logger.info("SiLU shared_experts fix installed on %d layers", patched)
    return patched


def patch_moe_routed_scale(backbone) -> int:
    """Apply the routed scaling factor omitted from Nemotron-H FP16 outputs."""
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        scale = float(getattr(mixer, "routed_scaling_factor", 1.0))
        if scale == 1.0:
            continue

        def _scale_output(_mod, _inp, out, _scale=scale):
            # FusedMoE only defers the scale in FP16; leave other dtypes alone.
            if isinstance(out, torch.Tensor) and out.dtype == torch.float16:
                return out * _scale
            return out

        mixer.register_forward_hook(_scale_output)
        patched += 1
    logger.info("FP16 MoE routed-scale fix installed on %d layers", patched)
    return patched
