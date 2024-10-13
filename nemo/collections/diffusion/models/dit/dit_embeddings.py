# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, get_3d_sincos_pos_embed
from einops import rearrange
from einops.layers.torch import Rearrange
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank
from megatron.core.transformer.module import MegatronModule
from torch import nn


class ParallelTimestepEmbedding(TimestepEmbedding):
    """
    ParallelTimestepEmbedding is a subclass of TimestepEmbedding that initializes
    the embedding layers with an optional random seed for syncronization.

    Args:
        in_channels (int): Number of input channels.
        time_embed_dim (int): Dimension of the time embedding.
        seed (int, optional): Random seed for initializing the embedding layers.
                              If None, no specific seed is set.

    Attributes:
        linear_1 (nn.Module): First linear layer for the embedding.
        linear_2 (nn.Module): Second linear layer for the embedding.

    Methods:
        __init__(in_channels, time_embed_dim, seed=None): Initializes the embedding layers.
    """

    def __init__(self, in_channels: int, time_embed_dim: int, seed=None):
        super().__init__(in_channels=in_channels, time_embed_dim=time_embed_dim)
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.linear_1.reset_parameters()
                self.linear_2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the positional embeddings for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C).

        Returns:
            torch.Tensor: Positional embeddings of shape (B, T, H, W, C).
        """
        return super().forward(x.to(torch.bfloat16, non_blocking=True))


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    """
    Adjusts the positional embeddings tensor to the current context parallel rank.

    Args:
        pos_emb (torch.Tensor): The positional embeddings tensor.
        seq_dim (int): The sequence dimension index in the positional embeddings tensor.

    Returns:
        torch.Tensor: The adjusted positional embeddings tensor for the current context parallel rank.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor([cp_rank], device="cpu", pin_memory=True).cuda(non_blocking=True)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], cp_size, -1, *pos_emb.shape[(seq_dim + 1) :])
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


class SinCosPosEmb3D(MegatronModule):
    """
    SinCosPosEmb3D is a 3D sine-cosine positional embedding module.

    Args:
        model_channels (int): Number of channels in the model.
        h (int): Length of the height dimension.
        w (int): Length of the width dimension.
        t (int): Length of the temporal dimension.
        spatial_interpolation_scale (float, optional): Scale factor for spatial interpolation. Default is 1.0.
        temporal_interpolation_scale (float, optional): Scale factor for temporal interpolation. Default is 1.0.

    Methods:
        forward(pos_ids: torch.Tensor) -> torch.Tensor:
            Computes the positional embeddings for the input tensor.

            Args:
                pos_ids (torch.Tensor): Input tensor of shape (B S 3).

            Returns:
                torch.Tensor: Positional embeddings of shape (B S D).
    """

    def __init__(
        self,
        config,
        h: int,
        w: int,
        t: int,
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
    ):
        super().__init__(config=config)
        self.h = h
        self.w = w
        self.t = t
        # h w t
        param = get_3d_sincos_pos_embed(
            config.hidden_size, [h, w], t, spatial_interpolation_scale, temporal_interpolation_scale
        )
        param = rearrange(param, "t hw c -> (t hw) c")
        self.pos_embedding = torch.nn.Embedding(param.shape[0], config.hidden_size)
        self.pos_embedding.weight = torch.nn.Parameter(torch.tensor(param), requires_grad=False)

    def forward(self, pos_ids: torch.Tensor):
        # pos_ids: t h w
        pos_id = pos_ids[..., 0] * self.h * self.w + pos_ids[..., 1] * self.w + pos_ids[..., 2]
        return self.pos_embedding(pos_id)


class FactorizedLearnable3DEmbedding(MegatronModule):
    def __init__(
        self,
        config,
        t: int,
        h: int,
        w: int,
        **kwargs,
    ):
        super().__init__(config=config)
        self.emb_t = torch.nn.Embedding(t, config.hidden_size)
        self.emb_h = torch.nn.Embedding(h, config.hidden_size)
        self.emb_w = torch.nn.Embedding(w, config.hidden_size)

        if config.perform_initialization:
            config.init_method(self.emb_t.weight)
            config.init_method(self.emb_h.weight)
            config.init_method(self.emb_w.weight)

    def forward(self, pos_ids: torch.Tensor):
        return self.emb_t(pos_ids[..., 0]) + self.emb_h(pos_ids[..., 1]) + self.emb_w(pos_ids[..., 2])
