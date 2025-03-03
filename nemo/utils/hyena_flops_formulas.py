# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

import math
from typing import Optional

# TODO(@cye): Merge MCore HyenaConfig with NeMo HyenaConfig to have all model params in 1 config.
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_config import HyenaConfig
from nemo.utils.flops_formulas import FLOPSConfig


def hyena(config: FLOPSConfig):
    """Model FLOPs for Hyena family. FPL = 'flops per layer'."""

    # TODO(@cye): For now, pull the Hyena defaults directly from a constant dataclass. Merge this config with the NeMo
    #   model config.
    hyena_config = HyenaConfig()
    # Hyena Parameters
    hyena_short_conv_L = hyena_config.short_conv_L
    hyena_short_conv_len = hyena_config.hyena_short_conv_len
    hyena_medium_conv_len = hyena_config.hyena_medium_conv_len

    def _hyena_layer_count(model_pattern: Optional[str]):
        """Count how many small, medium, and large Hyena layers there are in the model. Also, count the
        number of Attention layers.
        """
        S, D, H, A = 0, 0, 0, 0
        if model_pattern is None:
            return 0, 0, 0, 0
        for layer in model_pattern:
            if layer == "S":
                S += 1
            elif layer == "D":
                D += 1
            elif layer == "H":
                H += 1
            elif layer == "*":
                A += 1
        return S, D, H, A

    # Count S, D, H, and * layers in HyenaModel.
    S, D, H, A = _hyena_layer_count(config.model_pattern)
    # Logits FLOPs per batch for a flattened L x H -> V GEMM.
    logits_fpl = 2 * config.gbs * config.enc_seq_len * config.hs * config.vocab_size
    # Hyena Mixer Common FLOPs - Pre-Attention QKV Projections, Post-Attention Projections, and
    #   GLU FFN FLOPs per layer.
    pre_attn_qkv_proj_fpl = 2 * 3 * config.gbs * config.enc_seq_len * config.hs**2
    post_attn_proj_fpl = 2 * config.gbs * config.enc_seq_len * config.hs**2
    # 3 Batched GEMMs: y = A(gelu(Bx) * Cx) where B,C: H -> F and A: F -> H.
    glu_ffn_fpl = 2 * 3 * config.gbs * config.enc_seq_len * config.ffn_hs * config.hs
    # Transformer (Self) Attention FLOPs - QK Attention Logits ((L, D) x (D, L)) & Attention-Weighted
    #   Values FLOPs ((L, L) x (L, D))
    attn_fpl = 2 * 2 * config.gbs * config.hs * config.enc_seq_len**2
    # Hyena Projection
    hyena_proj_fpl = 2 * 3 * config.gbs * config.enc_seq_len * hyena_short_conv_L * config.hs
    # Hyena Short Conv
    hyena_short_conv_fpl = 2 * config.gbs * config.enc_seq_len * hyena_short_conv_len * config.hs
    # Hyena Medium Conv
    hyena_medium_conv_fpl = 2 * config.gbs * config.enc_seq_len * hyena_medium_conv_len * config.hs
    # Hyena Long Conv (FFT)
    hyena_long_conv_fft_fpl = config.gbs * 10 * config.enc_seq_len * math.log2(config.enc_seq_len) * config.hs
    # Based off of https://gitlab-master.nvidia.com/clara-discovery/savanna/-/blob/main/savanna/mfu.py#L182
    # Assumption: 1x Backwards Pass FLOPS = 2x Forward Pass FLOPS
    return 3 * (
        logits_fpl
        + config.layers * (pre_attn_qkv_proj_fpl + post_attn_proj_fpl + glu_ffn_fpl)
        + A * attn_fpl
        + (S + D + H) * hyena_proj_fpl
        + S * hyena_short_conv_fpl
        + D * hyena_medium_conv_fpl
        + H * hyena_long_conv_fft_fpl
    )
