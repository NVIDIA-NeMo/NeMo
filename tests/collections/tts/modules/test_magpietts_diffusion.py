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

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_diffusion import (
    OneShotLocalDiffusion,
    make_beta_schedule,
)


pytestmark = pytest.mark.unit


def _make_predictor(**kwargs) -> OneShotLocalDiffusion:
    return OneShotLocalDiffusion(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        n_layers=2,
        time_embedding_dim=8,
        training_timesteps=20,
        inference_steps=4,
        **kwargs,
    )


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_diffusion_schedule_is_finite_and_monotonic(schedule):
    betas = make_beta_schedule(schedule, timesteps=1000)
    alpha_cumprod = torch.cumprod(1.0 - betas, dim=0)

    assert betas.shape == (1000,)
    assert torch.isfinite(betas).all()
    assert torch.all((betas > 0.0) & (betas < 1.0))
    assert torch.all(alpha_cumprod[1:] < alpha_cumprod[:-1])


def test_diffusion_forward_process_uses_configured_signal_to_noise_ratio():
    predictor = _make_predictor()
    target = torch.full((2, 12, 3), 2.0)
    noise = torch.full_like(target, -0.5)
    timesteps = torch.tensor([0, 19])

    state = predictor.sample_noisy_state(target, noise, timesteps)

    signal = predictor.sqrt_alpha_cumprod[timesteps].view(2, 1, 1)
    noise_scale = predictor.sqrt_one_minus_alpha_cumprod[timesteps].view(2, 1, 1)
    torch.testing.assert_close(state, signal * target + noise_scale * noise)


def test_diffusion_loss_is_finite_and_backpropagates():
    predictor = _make_predictor()
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5, requires_grad=True)
    lengths = torch.tensor([5, 4])

    loss = predictor.compute_loss(acoustic, condition, lengths)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert condition.grad is not None
    assert torch.isfinite(condition.grad).all()
    assert predictor.estimator.output_projection.weight.grad is not None
    assert torch.count_nonzero(predictor.estimator.output_projection.weight.grad) > 0


def test_diffusion_loss_ignores_padding_and_explicitly_masked_frames():
    predictor = _make_predictor()
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 4])
    frame_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]]
    )
    changed_acoustic = acoustic.masked_fill(~frame_mask.unsqueeze(1), 1000.0)
    changed_condition = condition.masked_fill(~frame_mask.unsqueeze(1), -1000.0)

    torch.manual_seed(42)
    loss = predictor.compute_loss(acoustic, condition, lengths, frame_mask=frame_mask)
    torch.manual_seed(42)
    changed_loss = predictor.compute_loss(
        changed_acoustic, changed_condition, lengths, frame_mask=frame_mask
    )

    assert torch.equal(loss, changed_loss)


class _ZeroNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, condition, time, mask):
        del condition, time
        self.calls += 1
        return torch.zeros_like(state) * mask


class _OracleNoise(nn.Module):
    def __init__(self, predictor: OneShotLocalDiffusion, target: torch.Tensor):
        super().__init__()
        self.predictor = predictor
        self.target = target

    def forward(self, state, condition, time, mask):
        del condition
        timesteps = (time * (self.predictor.training_timesteps - 1)).round().long()
        alpha = self.predictor._extract(self.predictor.alpha_cumprod, timesteps, state)
        return ((state - alpha.sqrt() * self.target) / (1.0 - alpha).sqrt()) * mask


def test_diffusion_predict_iterates_inside_one_call_and_masks_padding():
    predictor = _make_predictor()
    estimator = _ZeroNoise()
    predictor.estimator = estimator
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])

    prediction = predictor.predict(condition, lengths)

    assert prediction.shape == (2, 12, 5)
    assert torch.isfinite(prediction).all()
    assert estimator.calls == predictor.inference_steps
    assert torch.count_nonzero(prediction[1, :, 3:]) == 0


def test_ddim_sampler_recovers_target_with_exact_noise_estimator():
    predictor = _make_predictor()
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])
    mask = predictor.length_mask(lengths, condition.size(2), condition.dtype)
    target = torch.randn(2, 12, 5) * mask
    predictor.estimator = _OracleNoise(predictor, target)

    prediction = predictor.predict(condition, lengths)

    torch.testing.assert_close(prediction, target, atol=2e-5, rtol=2e-5)


def test_diffusion_diagnostics_are_finite_and_select_worst_sample():
    predictor = _make_predictor()
    acoustic = torch.randn(2, 12, 5)
    acoustic[1] *= 100.0
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 4])

    diagnostics = predictor.compute_diagnostics(acoustic, condition, lengths)

    assert diagnostics["valid_frames"].item() in (4, 5)
    assert all(torch.isfinite(value) for value in diagnostics.values())
