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

"""Interfaces and construction helpers for one-shot EasyMagpie acoustic predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


NORMALIZING_FLOW = "normalizing_flow"
FLOW_MATCHING = "flow_matching"
DIFFUSION = "diffusion"


class OneShotLocalPredictor(ABC, nn.Module):
    """Predict a complete continuous acoustic codec frame from backbone conditioning.

    Implementations own both their training objective and inference procedure. This keeps
    EasyMagpie independent of the continuous generative model used for acoustic prediction.
    """

    def __init__(self, acoustic_channels: int):
        super().__init__()
        if acoustic_channels < 1:
            raise ValueError(f"acoustic_channels must be positive, got {acoustic_channels}")
        self.acoustic_channels = acoustic_channels

    @staticmethod
    def length_mask(lengths: torch.Tensor, max_length: int, dtype: torch.dtype) -> torch.Tensor:
        """Return a broadcastable ``(batch, 1, time)`` mask."""
        positions = torch.arange(max_length, device=lengths.device)
        return (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).to(dtype)

    @abstractmethod
    def compute_loss(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the scalar training loss for a batch of target acoustic embeddings."""

    @abstractmethod
    def compute_diagnostics(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return compact diagnostics for the worst sample in a batch."""

    @abstractmethod
    def predict(
        self,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate acoustic embeddings from the conditioning sequence."""


def create_oneshot_local_predictor(
    predictor_type: str,
    *,
    acoustic_channels: int,
    condition_channels: int,
    cfg,
    semantic_vocab_size: int | None = None,
    semantic_channels: int | None = None,
) -> OneShotLocalPredictor:
    """Construct the configured one-shot predictor behind the common interface."""
    predictor_type = str(predictor_type).lower()
    if predictor_type == NORMALIZING_FLOW:
        from nemo.collections.tts.modules.magpietts_flow import OneShotLocalFlow

        return OneShotLocalFlow(
            acoustic_channels=acoustic_channels,
            condition_channels=condition_channels,
            hidden_channels=int(cfg.get("local_flow_hidden_dim", 1536)),
            n_layers=int(cfg.get("local_flow_n_layers", 3)),
            n_flows=int(cfg.get("local_flow_n_flows", 4)),
            dropout=float(cfg.get("local_flow_dropout", 0.0)),
            coupling_type=str(cfg.get("local_flow_coupling_type", "affine")),
            spline_num_bins=int(cfg.get("local_flow_spline_num_bins", 8)),
            spline_tail_bound=float(cfg.get("local_flow_spline_tail_bound", 5.0)),
            spline_min_bin_width=float(cfg.get("local_flow_spline_min_bin_width", 1e-3)),
            spline_min_bin_height=float(cfg.get("local_flow_spline_min_bin_height", 1e-3)),
            spline_min_derivative=float(cfg.get("local_flow_spline_min_derivative", 1e-3)),
            match_affine_parameter_count=bool(cfg.get("local_flow_match_affine_parameter_count", True)),
        )

    if predictor_type == FLOW_MATCHING:
        from nemo.collections.tts.modules.magpietts_flow_matching import (
            OneShotLocalFlowMatching,
        )

        return OneShotLocalFlowMatching(
            acoustic_channels=acoustic_channels,
            condition_channels=condition_channels,
            hidden_channels=int(cfg.get("local_flow_matching_hidden_dim", 1536)),
            n_layers=int(cfg.get("local_flow_matching_n_layers", 3)),
            dropout=float(cfg.get("local_flow_matching_dropout", 0.0)),
            time_embedding_dim=int(cfg.get("local_flow_matching_time_embedding_dim", 128)),
            inference_steps=int(cfg.get("local_flow_matching_inference_steps", 8)),
            solver=str(cfg.get("local_flow_matching_solver", "midpoint")),
            num_noise_samples=int(cfg.get("local_flow_matching_train_num_noise_samples", 1)),
            estimator_type=str(cfg.get("local_flow_matching_estimator_type", "pointwise")),
            semantic_vocab_size=semantic_vocab_size,
            semantic_channels=semantic_channels,
            transformer_n_heads=int(cfg.get("local_flow_matching_transformer_n_heads", 12)),
            transformer_ffn_multiplier=float(cfg.get("local_flow_matching_transformer_ffn_multiplier", 4.0)),
            transformer_condition_dropout=float(cfg.get("local_flow_matching_transformer_condition_dropout", 0.1)),
        )

    if predictor_type == DIFFUSION:
        from nemo.collections.tts.modules.magpietts_diffusion import (
            OneShotLocalDiffusion,
        )

        return OneShotLocalDiffusion(
            acoustic_channels=acoustic_channels,
            condition_channels=condition_channels,
            hidden_channels=int(cfg.get("local_diffusion_hidden_dim", 1536)),
            n_layers=int(cfg.get("local_diffusion_n_layers", 3)),
            dropout=float(cfg.get("local_diffusion_dropout", 0.0)),
            time_embedding_dim=int(cfg.get("local_diffusion_time_embedding_dim", 128)),
            training_timesteps=int(cfg.get("local_diffusion_training_timesteps", 1000)),
            inference_steps=int(cfg.get("local_diffusion_inference_steps", 16)),
            beta_schedule=str(cfg.get("local_diffusion_beta_schedule", "linear")),
            ddim_eta=float(cfg.get("local_diffusion_ddim_eta", 0.0)),
        )

    raise ValueError(f"Unsupported one-shot local predictor type {predictor_type!r}.")
