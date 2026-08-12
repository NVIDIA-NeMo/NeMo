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

"""Conditional flow matching for one-shot EasyMagpie acoustic prediction.

The straight conditional optimal-transport path follows the formulation demonstrated by
https://github.com/facebookresearch/flow_matching. The implementation here is native to
NeMo and uses the repository's existing tensor conventions and masking behavior.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_oneshot import OneShotLocalPredictor


EULER_SOLVER = "euler"
MIDPOINT_SOLVER = "midpoint"


class SinusoidalTimeEmbedding(nn.Module):
    """Embed scalar flow times with fixed log-spaced sinusoidal features."""

    def __init__(self, embedding_dim: int, max_period: float = 1000.0):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError(f"time embedding_dim must be at least two, got {embedding_dim}")
        if max_period <= 1.0:
            raise ValueError(f"max_period must be greater than one, got {max_period}")

        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        frequencies = torch.exp(torch.linspace(0.0, math.log(max_period), half_dim))
        self.register_buffer("frequencies", 2.0 * math.pi * frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        arguments = time.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat([arguments.sin(), arguments.cos()], dim=1)
        if embedding.size(1) < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.embedding_dim - embedding.size(1)))
        return embedding


class PointwiseFlowMatchingResidualBlock(nn.Module):
    """Pre-normalized per-frame residual MLP."""

    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)
        self.layers = nn.Sequential(
            nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(2 * hidden_channels, hidden_channels, 1),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        return (hidden + self.layers(normalized)) * mask


class PointwiseFlowMatchingEstimator(nn.Module):
    """Estimate the conditional acoustic velocity independently at every codec frame."""

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        hidden_channels: int,
        n_layers: int,
        dropout: float,
        time_embedding_dim: int,
    ):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.state_projection = nn.Conv1d(acoustic_channels, hidden_channels, 1)
        self.condition_projection = nn.Conv1d(condition_channels, hidden_channels, 1)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.residual_layers = nn.ModuleList(
            [PointwiseFlowMatchingResidualBlock(hidden_channels, dropout) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_channels)
        self.output_projection = nn.Conv1d(hidden_channels, acoustic_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_projection(self.time_embedding(time).to(state.dtype)).unsqueeze(2)
        hidden = (self.state_projection(state) + self.condition_projection(condition) + time_embedding) * mask
        for residual_layer in self.residual_layers:
            hidden = residual_layer(hidden, mask)
        hidden = self.output_norm(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(torch.nn.functional.silu(hidden)) * mask


class OneShotLocalFlowMatching(OneShotLocalPredictor):
    """Conditional straight-path flow matching over continuous acoustic codec embeddings."""

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        hidden_channels: int,
        n_layers: int = 3,
        dropout: float = 0.0,
        time_embedding_dim: int = 128,
        inference_steps: int = 8,
        solver: str = MIDPOINT_SOLVER,
    ):
        super().__init__(acoustic_channels=acoustic_channels)
        if inference_steps < 1:
            raise ValueError(f"inference_steps must be positive, got {inference_steps}")
        solver = solver.lower()
        if solver not in (EULER_SOLVER, MIDPOINT_SOLVER):
            raise ValueError(
                f"Unsupported flow-matching solver {solver!r}; expected {EULER_SOLVER!r} or {MIDPOINT_SOLVER!r}."
            )

        self.inference_steps = inference_steps
        self.solver = solver
        self.estimator = PointwiseFlowMatchingEstimator(
            acoustic_channels=acoustic_channels,
            condition_channels=condition_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim,
        )

    def _mask(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if acoustic_embedding.shape[0] != condition.shape[0] or acoustic_embedding.shape[2] != condition.shape[2]:
            raise ValueError(
                "Acoustic target and condition must share batch/time dimensions, got "
                f"{acoustic_embedding.shape} and {condition.shape}"
            )
        if acoustic_embedding.size(1) != self.acoustic_channels:
            raise ValueError(f"Expected {self.acoustic_channels} acoustic channels, got {acoustic_embedding.size(1)}.")

        mask = self.length_mask(lengths, acoustic_embedding.size(2), acoustic_embedding.dtype)
        if frame_mask is not None:
            if frame_mask.shape != (acoustic_embedding.size(0), acoustic_embedding.size(2)):
                raise ValueError(
                    f"frame_mask must have shape {(acoustic_embedding.size(0), acoustic_embedding.size(2))}, "
                    f"got {frame_mask.shape}."
                )
            mask = mask * frame_mask.unsqueeze(1).to(mask.dtype)
        return mask

    @staticmethod
    def sample_path(
        target: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the straight conditional OT path and its constant conditional velocity."""
        broadcast_time = time.view(time.size(0), *([1] * (target.ndim - 1))).to(target.dtype)
        state = (1.0 - broadcast_time) * noise + broadcast_time * target
        velocity = target - noise
        return state, velocity

    def _loss_tensors(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self._mask(acoustic_embedding, condition, lengths, frame_mask)
        target = acoustic_embedding * mask
        noise = torch.randn_like(target) * mask
        time = torch.rand(target.size(0), device=target.device)
        state, target_velocity = self.sample_path(target, noise, time)
        predicted_velocity = self.estimator(state, condition * mask, time, mask)
        squared_error = (predicted_velocity.float() - target_velocity.float()).square() * mask.float()
        return squared_error, mask, state, target_velocity, predicted_velocity

    def compute_loss(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squared_error, mask, _, _, _ = self._loss_tensors(acoustic_embedding, condition, lengths, frame_mask)
        denominator = (mask.float().sum() * self.acoustic_channels).clamp_min(1.0)
        return squared_error.sum() / denominator

    @torch.no_grad()
    def compute_diagnostics(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        squared_error, mask, state, target_velocity, predicted_velocity = self._loss_tensors(
            acoustic_embedding, condition, lengths, frame_mask
        )
        float_mask = mask.float()
        valid_frames = float_mask.sum(dim=(1, 2))
        denominator = (valid_frames * self.acoustic_channels).clamp_min(1.0)
        per_sample_loss = squared_error.sum(dim=(1, 2)) / denominator
        worst_sample_index = per_sample_loss.argmax()

        def _masked_abs_max(values: torch.Tensor) -> torch.Tensor:
            return (values.float().abs() * float_mask).amax(dim=(1, 2))

        def _masked_rms(values: torch.Tensor) -> torch.Tensor:
            value_denominator = (valid_frames * values.size(1)).clamp_min(1.0)
            return ((values.float().square() * float_mask).sum(dim=(1, 2)) / value_denominator).sqrt()

        return {
            "sample_index": worst_sample_index,
            "sample_loss": per_sample_loss[worst_sample_index],
            "valid_frames": valid_frames[worst_sample_index],
            "target_abs_max": _masked_abs_max(acoustic_embedding)[worst_sample_index],
            "target_rms": _masked_rms(acoustic_embedding)[worst_sample_index],
            "condition_abs_max": _masked_abs_max(condition)[worst_sample_index],
            "condition_rms": _masked_rms(condition)[worst_sample_index],
            "latent_abs_max": _masked_abs_max(state)[worst_sample_index],
            "latent_rms": _masked_rms(state)[worst_sample_index],
            "normalized_log_determinant": acoustic_embedding.new_zeros(()),
            "target_velocity_abs_max": _masked_abs_max(target_velocity)[worst_sample_index],
            "predicted_velocity_abs_max": _masked_abs_max(predicted_velocity)[worst_sample_index],
        }

    def _guided_velocity(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor,
        unconditional_condition: torch.Tensor | None,
        cfg_scale: float,
    ) -> torch.Tensor:
        conditional_velocity = self.estimator(state, condition, time, mask)
        if unconditional_condition is None or cfg_scale == 1.0:
            return conditional_velocity

        unconditional_velocity = self.estimator(state, unconditional_condition, time, mask)
        return cfg_scale * conditional_velocity + (1.0 - cfg_scale) * unconditional_velocity

    @torch.no_grad()
    def predict(
        self,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        noise_scale: float = 1.0,
        unconditional_condition: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
        if unconditional_condition is not None and unconditional_condition.shape != condition.shape:
            raise ValueError(
                "Conditional and unconditional flow inputs must have the same shape, got "
                f"{condition.shape} and {unconditional_condition.shape}."
            )
        mask = self.length_mask(lengths, condition.size(2), condition.dtype)
        state = torch.randn(
            condition.size(0),
            self.acoustic_channels,
            condition.size(2),
            device=condition.device,
            dtype=condition.dtype,
        )
        state = state * noise_scale * mask
        condition = condition * mask
        if unconditional_condition is not None:
            unconditional_condition = unconditional_condition * mask
        step_size = 1.0 / self.inference_steps

        for step in range(self.inference_steps):
            start_time = step * step_size
            time = torch.full((state.size(0),), start_time, device=state.device)
            velocity = self._guided_velocity(state, condition, time, mask, unconditional_condition, cfg_scale)
            if self.solver == EULER_SOLVER:
                state = state + step_size * velocity
            else:
                midpoint_state = state + 0.5 * step_size * velocity
                midpoint_time = time + 0.5 * step_size
                midpoint_velocity = self._guided_velocity(
                    midpoint_state,
                    condition,
                    midpoint_time,
                    mask,
                    unconditional_condition,
                    cfg_scale,
                )
                state = state + step_size * midpoint_velocity
            state = state * mask

        return state
