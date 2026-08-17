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

"""Four-token transformer estimator for per-frame EasyMagpie flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn

from nemo.collections.tts.modules import transformer_2501


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar flow times with fixed log-spaced sinusoidal features."""

    def __init__(self, embedding_dim: int, max_period: float = 1000.0):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError(f"time embedding_dim must be at least two, got {embedding_dim}")
        half_dim = embedding_dim // 2
        frequencies = torch.exp(torch.linspace(0.0, math.log(max_period), half_dim))
        self.embedding_dim = embedding_dim
        self.register_buffer("frequencies", 2.0 * math.pi * frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        arguments = time.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat([arguments.sin(), arguments.cos()], dim=1)
        if embedding.size(1) < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.embedding_dim - embedding.size(1)))
        return embedding


class EasyMagpieFlowMatchingTransformerEstimator(nn.Module):
    """Predict velocity from four fully-attending tokens at each codec frame.

    The tokens are the decoder-backbone hidden state, semantic codec token,
    sinusoidal flow time, and current noisy acoustic state.  Frames are folded
    into the batch dimension, so attention is bidirectional within the four
    tokens but never crosses codec-frame boundaries.
    """

    requires_semantic_codes = True

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        semantic_vocab_size: int,
        semantic_channels: int,
        hidden_channels: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        time_embedding_dim: int,
        ffn_multiplier: float = 4.0,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_channels < 1 or n_layers < 1 or n_heads < 1:
            raise ValueError("hidden_channels, n_layers, and n_heads must be positive")
        if hidden_channels % n_heads:
            raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by n_heads ({n_heads})")
        if semantic_vocab_size < 1 or semantic_channels < 1:
            raise ValueError("semantic_vocab_size and semantic_channels must be positive")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= condition_dropout < 1.0:
            raise ValueError("dropout and condition_dropout must be in [0, 1)")
        if ffn_multiplier <= 0.0:
            raise ValueError(f"ffn_multiplier must be positive, got {ffn_multiplier}")

        self.acoustic_channels = acoustic_channels
        self.condition_channels = condition_channels
        self.semantic_channels = semantic_channels
        self.hidden_channels = hidden_channels
        self.condition_dropout = condition_dropout
        self.condition_projection = nn.Linear(condition_channels, hidden_channels, bias=False)
        self.semantic_embeddings = nn.ModuleList(
            [nn.Embedding(semantic_vocab_size, hidden_channels) for _ in range(semantic_channels)]
        )
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_projection = nn.Linear(time_embedding_dim, hidden_channels)
        self.state_projection = nn.Linear(acoustic_channels, hidden_channels)
        self.transformer = transformer_2501.Transformer(
            n_layers=n_layers,
            d_model=hidden_channels,
            d_ffn=int(round(ffn_multiplier * hidden_channels)),
            sa_n_heads=n_heads,
            kernel_size=1,
            p_dropout=dropout,
            is_causal=False,
            apply_norm_out=True,
            max_length_causal_mask=4,
            use_learnable_pos_emb=True,
        )
        self.output_projection = nn.Linear(hidden_channels, acoustic_channels)

    def _embed_semantic_codes(self, semantic_codes: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(semantic_codes[:, channel].long()) for channel, embedding in enumerate(self.semantic_embeddings)
        ]
        return torch.stack(embeddings, dim=0).sum(dim=0)

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        semantic_codes: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_frames = state.shape
        expected_condition = (batch_size, self.condition_channels, num_frames)
        expected_semantic = (batch_size, self.semantic_channels, num_frames)
        if tuple(condition.shape) != expected_condition:
            raise ValueError(f"Expected condition {expected_condition}, got {tuple(condition.shape)}.")
        if tuple(semantic_codes.shape) != expected_semantic:
            raise ValueError(f"Expected semantic codes {expected_semantic}, got {tuple(semantic_codes.shape)}.")

        condition_token = self.condition_projection(condition.transpose(1, 2))
        if self.training and self.condition_dropout > 0.0:
            keep_condition = torch.rand(batch_size, 1, 1, device=condition.device) >= self.condition_dropout
            condition_token = condition_token * keep_condition.to(condition_token.dtype)
        semantic_token = self._embed_semantic_codes(semantic_codes)
        time_token = self.time_projection(self.time_embedding(time)).unsqueeze(1).expand(-1, num_frames, -1)
        state_token = self.state_projection(state.transpose(1, 2))

        tokens = torch.stack(
            [condition_token, semantic_token, time_token, state_token],
            dim=2,
        ).reshape(batch_size * num_frames, 4, self.hidden_channels)
        token_mask = mask[:, 0].bool().reshape(batch_size * num_frames, 1).expand(-1, 4)
        transformed = self.transformer(tokens, token_mask)['output']
        velocity = self.output_projection(transformed[:, -1]).reshape(batch_size, num_frames, self.acoustic_channels)
        return velocity.transpose(1, 2) * mask
