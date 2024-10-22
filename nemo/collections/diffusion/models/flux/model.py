# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Callable

import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import openai_gelu
from torch import nn

from nemo.collections.diffusion.models.dit.dit_layer_spec import (
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from nemo.collections.diffusion.models.flux.layers import EmbedND, MLPEmbedder, TimeStepEmbedder


@dataclass
class FluxParams:
    num_joint_layers: int = 19
    num_single_layers: int = 38
    hidden_size: int = 3072
    num_attention_heads: int = 24
    activation_func: Callable = openai_gelu
    add_qkv_bias: bool = True
    ffn_hidden_size: int = 16384
    in_channels: int = 64
    context_dim: int = 4096
    model_channels: int = 256
    patch_size: int = 1
    guidance_embed: bool = False
    vec_in_dim: int = 768


class Flux(VisionModule):
    def __init__(self, config: FluxParams):

        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.guidance_embed = config.guidance_embed
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            use_cpu_initialization=True,
            activation_func=config.activation_func,
            hidden_dropout=0,
            attention_dropout=0,
            layernorm_epsilon=1e-6,
            add_qkv_bias=config.add_qkv_bias,
            rotary_interleaved=True,
        )
        super().__init__(transformer_config)

        self.pos_embed = EmbedND(dim=self.hidden_size, theta=10000, axes_dim=[16, 56, 56])
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)
        self.timestep_embedding = TimeStepEmbedder(config.model_channels, self.hidden_size)
        self.vector_embedding = MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)
        if config.guidance_embed:
            self.guidance_embedding = (
                MLPEmbedder(in_dim=config.model_channels, hidden_dim=self.hidden_size)
                if config.guidance_embed
                else nn.Identity()
            )

        self.double_blocks = nn.ModuleList(
            [
                MMDiTLayer(
                    config=transformer_config,
                    submodules=get_flux_double_transformer_engine_spec().submodules,
                    layer_number=i,
                    context_pre_only=False,
                )
                for i in range(config.num_joint_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    config=transformer_config,
                    submodules=get_flux_single_transformer_engine_spec().submodules,
                    layer_number=i,
                )
                for i in range(config.num_single_layers)
            ]
        )

        self.norm_out = AdaLNContinuous(config=transformer_config, conditioning_embedding_dim=self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=True)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
    ):
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)

        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))
        vec_emb = vec_emb + self.vector_embedding(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)
        for id_block, block in enumerate(self.double_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        for id_block, block in enumerate(self.single_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                emb=vec_emb,
            )

        hidden_states = hidden_states[encoder_hidden_states.shape[0] :, ...]

        hidden_states = self.norm_out(hidden_states, vec_emb)
        output = self.proj_out(hidden_states)

        return output
