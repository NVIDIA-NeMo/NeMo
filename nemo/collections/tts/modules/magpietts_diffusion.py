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

"""Conditional DDPM training and iterative DDIM sampling for acoustic embeddings.

The noise-prediction objective follows the original DDPM formulation. Inference uses
DDIM so a validation sample does not require all 1,000 training diffusion steps. The
implementation follows the equations used by the widely adopted reference at
https://github.com/lucidrains/denoising-diffusion-pytorch while retaining NeMo's
one-shot predictor interface and tensor conventions.
"""

from __future__ import annotations

import math

import torch

from nemo.collections.tts.modules.magpietts_flow_matching import (
    PointwiseFlowMatchingEstimator,
)
from nemo.collections.tts.modules.magpietts_oneshot import OneShotLocalPredictor


LINEAR_SCHEDULE = "linear"
COSINE_SCHEDULE = "cosine"


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    """Create a numerically valid DDPM variance schedule in float32."""
    if timesteps < 2:
        raise ValueError(f"training_timesteps must be at least two, got {timesteps}")

    schedule = schedule.lower()
    if schedule == LINEAR_SCHEDULE:
        return torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32)
    if schedule == COSINE_SCHEDULE:
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        offset = 0.008
        alpha_bar = torch.cos(
            ((steps / timesteps + offset) / (1.0 + offset)) * math.pi / 2.0
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-4, 0.999).float()
    raise ValueError(
        f"Unsupported diffusion beta schedule {schedule!r}; expected {LINEAR_SCHEDULE!r} or {COSINE_SCHEDULE!r}."
    )


class OneShotLocalDiffusion(OneShotLocalPredictor):
    """Conditional epsilon-prediction DDPM with iterative DDIM inference."""

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        hidden_channels: int,
        n_layers: int = 3,
        dropout: float = 0.0,
        time_embedding_dim: int = 128,
        training_timesteps: int = 1000,
        inference_steps: int = 16,
        beta_schedule: str = LINEAR_SCHEDULE,
        ddim_eta: float = 0.0,
    ):
        super().__init__(acoustic_channels=acoustic_channels)
        if inference_steps < 1 or inference_steps > training_timesteps:
            raise ValueError(
                f"inference_steps must be in [1, {training_timesteps}], got {inference_steps}"
            )
        if ddim_eta < 0.0:
            raise ValueError(f"ddim_eta must be non-negative, got {ddim_eta}")

        betas = make_beta_schedule(beta_schedule, training_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alpha_cumprod", alpha_cumprod, persistent=False)
        self.register_buffer(
            "sqrt_alpha_cumprod", alpha_cumprod.sqrt(), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            (1.0 - alpha_cumprod).sqrt(),
            persistent=False,
        )

        self.training_timesteps = training_timesteps
        self.inference_steps = inference_steps
        self.beta_schedule = beta_schedule.lower()
        self.ddim_eta = ddim_eta
        self.estimator = PointwiseFlowMatchingEstimator(
            acoustic_channels=acoustic_channels,
            condition_channels=condition_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim,
        )

    @staticmethod
    def _extract(
        values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        return values[timesteps].view(timesteps.size(0), 1, 1).to(reference.dtype)

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / float(self.training_timesteps - 1)

    def _mask(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            acoustic_embedding.shape[0] != condition.shape[0]
            or acoustic_embedding.shape[2] != condition.shape[2]
        ):
            raise ValueError(
                "Acoustic target and condition must share batch/time dimensions, got "
                f"{acoustic_embedding.shape} and {condition.shape}"
            )
        if acoustic_embedding.size(1) != self.acoustic_channels:
            raise ValueError(
                f"Expected {self.acoustic_channels} acoustic channels, got {acoustic_embedding.size(1)}."
            )

        mask = self.length_mask(
            lengths, acoustic_embedding.size(2), acoustic_embedding.dtype
        )
        if frame_mask is not None:
            if frame_mask.shape != (
                acoustic_embedding.size(0),
                acoustic_embedding.size(2),
            ):
                raise ValueError(
                    f"frame_mask must have shape {(acoustic_embedding.size(0), acoustic_embedding.size(2))}, "
                    f"got {frame_mask.shape}."
                )
            mask = mask * frame_mask.unsqueeze(1).to(mask.dtype)
        return mask

    def sample_noisy_state(
        self,
        target: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the configured forward diffusion process."""
        signal_scale = self._extract(self.sqrt_alpha_cumprod, timesteps, target)
        noise_scale = self._extract(
            self.sqrt_one_minus_alpha_cumprod, timesteps, target
        )
        return signal_scale * target + noise_scale * noise

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
        timesteps = torch.randint(
            self.training_timesteps, (target.size(0),), device=target.device
        )
        noisy_state = self.sample_noisy_state(target, noise, timesteps) * mask
        predicted_noise = self.estimator(
            noisy_state,
            condition * mask,
            self._normalized_time(timesteps),
            mask,
        )
        squared_error = (
            predicted_noise.float() - noise.float()
        ).square() * mask.float()
        return squared_error, mask, noisy_state, noise, predicted_noise

    def compute_loss(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squared_error, mask, _, _, _ = self._loss_tensors(
            acoustic_embedding, condition, lengths, frame_mask
        )
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
        squared_error, mask, state, target_noise, predicted_noise = self._loss_tensors(
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
            return (
                (values.float().square() * float_mask).sum(dim=(1, 2))
                / value_denominator
            ).sqrt()

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
            "target_noise_abs_max": _masked_abs_max(target_noise)[worst_sample_index],
            "predicted_noise_abs_max": _masked_abs_max(predicted_noise)[
                worst_sample_index
            ],
        }

    @torch.no_grad()
    def predict(
        self,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate acoustic embeddings with deterministic DDIM when ddim_eta is zero."""
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")

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

        current_steps = (
            torch.linspace(
                self.training_timesteps - 1,
                0,
                self.inference_steps,
                device=condition.device,
            )
            .round()
            .long()
        )
        next_steps = torch.cat([current_steps[1:], current_steps.new_tensor([-1])])

        for current_step, next_step in zip(current_steps, next_steps):
            timesteps = current_step.expand(state.size(0))
            predicted_noise = self.estimator(
                state,
                condition,
                self._normalized_time(timesteps),
                mask,
            )
            alpha = self.alpha_cumprod[current_step].to(state.dtype)
            next_alpha = (
                self.alpha_cumprod[next_step].to(state.dtype)
                if int(next_step.item()) >= 0
                else state.new_ones(())
            )
            predicted_start = (
                state - (1.0 - alpha).sqrt() * predicted_noise
            ) / alpha.sqrt()

            sigma = (
                self.ddim_eta
                * (
                    ((1.0 - next_alpha) / (1.0 - alpha)).clamp_min(0.0)
                    * (1.0 - alpha / next_alpha).clamp_min(0.0)
                ).sqrt()
            )
            residual_scale = (1.0 - next_alpha - sigma.square()).clamp_min(0.0).sqrt()
            state = (
                next_alpha.sqrt() * predicted_start + residual_scale * predicted_noise
            )
            if self.ddim_eta > 0.0 and int(next_step.item()) >= 0:
                state = state + sigma * torch.randn_like(state)
            state = state * mask

        return state
