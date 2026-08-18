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

from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_flow_matching import OneShotLocalFlowMatching


pytestmark = pytest.mark.unit


def _make_predictor(**kwargs) -> OneShotLocalFlowMatching:
    return OneShotLocalFlowMatching(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        n_layers=2,
        time_embedding_dim=8,
        inference_steps=4,
        **kwargs,
    )


def test_flow_matching_straight_path_has_expected_state_and_velocity():
    target = torch.tensor([[[3.0]], [[-1.0]]])
    noise = torch.tensor([[[-1.0]], [[3.0]]])
    time = torch.tensor([0.25, 0.75])

    state, velocity = OneShotLocalFlowMatching.sample_path(target, noise, time)

    assert torch.equal(state, torch.tensor([[[0.0]], [[0.0]]]))
    assert torch.equal(velocity, torch.tensor([[[4.0]], [[-4.0]]]))


def test_flow_matching_loss_is_finite_and_backpropagates():
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


def test_flow_matching_loss_ignores_padding_and_explicitly_masked_frames():
    predictor = _make_predictor()
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 4])
    frame_mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    changed_acoustic = acoustic.masked_fill(~frame_mask.unsqueeze(1), 1000.0)
    changed_condition = condition.masked_fill(~frame_mask.unsqueeze(1), -1000.0)

    torch.manual_seed(42)
    loss = predictor.compute_loss(acoustic, condition, lengths, frame_mask=frame_mask)
    torch.manual_seed(42)
    changed_loss = predictor.compute_loss(changed_acoustic, changed_condition, lengths, frame_mask=frame_mask)

    assert torch.equal(loss, changed_loss)


class _RecordingZeroVelocity(nn.Module):
    def forward(self, state, condition, time, mask):
        self.state = state.detach().clone()
        self.condition = condition.detach().clone()
        self.time = time.detach().clone()
        self.mask = mask.detach().clone()
        return condition[:, : state.size(1)] * 0.0


def test_flow_matching_loss_reuses_conditions_with_independent_noise_samples():
    num_noise_samples = 4
    predictor = _make_predictor(num_noise_samples=num_noise_samples)
    estimator = _RecordingZeroVelocity()
    predictor.estimator = estimator
    acoustic = torch.zeros(2, 12, 3)
    condition = torch.arange(2 * 20 * 3, dtype=torch.float32).view(2, 20, 3).requires_grad_()
    lengths = torch.tensor([3, 2])
    frame_mask = torch.tensor([[True, True, False], [True, False, False]])
    expanded_batch_size = acoustic.size(0) * num_noise_samples

    def deterministic_noise(value):
        return torch.arange(value.numel(), device=value.device, dtype=value.dtype).view_as(value) / value.numel()

    def deterministic_time(size, *, device):
        return torch.arange(1, size + 1, device=device, dtype=torch.float32) / (size + 1)

    with (
        patch.object(torch, "randn_like", side_effect=deterministic_noise) as noise_mock,
        patch.object(torch, "rand", side_effect=deterministic_time) as time_mock,
    ):
        loss = predictor.compute_loss(acoustic, condition, lengths, frame_mask=frame_mask)
    loss.backward()

    expanded_condition = condition.detach().repeat_interleave(num_noise_samples, dim=0)
    expanded_lengths = lengths.repeat_interleave(num_noise_samples, dim=0)
    expanded_frame_mask = frame_mask.repeat_interleave(num_noise_samples, dim=0)
    expected_mask = predictor.length_mask(
        expanded_lengths, acoustic.size(2), acoustic.dtype
    ) * expanded_frame_mask.unsqueeze(1)

    assert estimator.state.shape == (expanded_batch_size, 12, 3)
    torch.testing.assert_close(estimator.condition, expanded_condition * expected_mask)
    torch.testing.assert_close(estimator.mask, expected_mask)
    torch.testing.assert_close(
        estimator.time,
        torch.arange(1, expanded_batch_size + 1, dtype=torch.float32) / (expanded_batch_size + 1),
    )
    assert not torch.equal(estimator.state[0], estimator.state[1])
    assert condition.grad is not None
    assert noise_mock.call_args.args[0].size(0) == expanded_batch_size
    time_mock.assert_called_once_with(expanded_batch_size, device=acoustic.device)


def test_flow_matching_validation_does_not_expand_noise_samples():
    predictor = _make_predictor(num_noise_samples=4)
    estimator = _RecordingZeroVelocity()
    predictor.estimator = estimator
    predictor.eval()
    acoustic = torch.zeros(2, 12, 3)
    condition = torch.zeros(2, 20, 3)
    lengths = torch.tensor([3, 2])

    predictor.compute_loss(acoustic, condition, lengths)

    assert estimator.state.size(0) == acoustic.size(0)


class _ConditionVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, condition, time, mask):
        del time
        self.calls += 1
        return condition[:, : state.size(1)] * mask


class _SquaredConditionVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, condition, time, mask):
        del state, time
        self.calls += 1
        return condition[:, :12].square() * mask


@pytest.mark.parametrize("solver, expected_calls", [("euler", 4), ("midpoint", 8)])
def test_flow_matching_predict_integrates_inside_one_call(solver, expected_calls):
    predictor = _make_predictor(solver=solver)
    estimator = _ConditionVelocity()
    predictor.estimator = estimator
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])
    mask = predictor.length_mask(lengths, condition.size(2), condition.dtype)

    torch.manual_seed(42)
    initial_state = torch.randn(2, 12, 5) * mask
    torch.manual_seed(42)
    prediction = predictor.predict(condition, lengths)

    expected = (initial_state + condition[:, :12] * mask) * mask
    assert torch.allclose(prediction, expected, atol=1e-6, rtol=1e-6)
    assert estimator.calls == expected_calls
    assert torch.count_nonzero(prediction[1, :, 3:]) == 0


def test_flow_matching_predict_applies_cfg_to_nonlinear_velocity_fields():
    predictor = _make_predictor(solver="euler")
    predictor.inference_steps = 1
    estimator = _SquaredConditionVelocity()
    predictor.estimator = estimator
    condition = torch.full((1, 20, 2), 2.0)
    unconditional_condition = torch.ones_like(condition)
    lengths = torch.tensor([2])
    cfg_scale = 2.5

    torch.manual_seed(42)
    initial_state = torch.randn(1, 12, 2)
    torch.manual_seed(42)
    prediction = predictor.predict(
        condition,
        lengths,
        unconditional_condition=unconditional_condition,
        cfg_scale=cfg_scale,
    )

    conditional_velocity = condition[:, :12].square()
    unconditional_velocity = unconditional_condition[:, :12].square()
    expected_velocity = cfg_scale * conditional_velocity + (1.0 - cfg_scale) * unconditional_velocity
    torch.testing.assert_close(prediction, initial_state + expected_velocity)
    assert estimator.calls == 2


def test_flow_matching_diagnostics_are_finite_and_select_worst_sample():
    predictor = _make_predictor()
    acoustic = torch.randn(2, 12, 5)
    acoustic[1] *= 100.0
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 4])

    diagnostics = predictor.compute_diagnostics(acoustic, condition, lengths)

    assert diagnostics["sample_index"].item() == 1
    assert diagnostics["valid_frames"].item() == 4
    assert diagnostics["target_abs_max"] > diagnostics["condition_abs_max"]
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_transformer_flow_matching_uses_semantic_tokens_and_backpropagates():
    predictor = _make_predictor(
        estimator_type="transformer",
        semantic_vocab_size=32,
        semantic_channels=1,
        transformer_n_heads=4,
        transformer_ffn_multiplier=2.0,
        transformer_condition_dropout=0.0,
    )
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5, requires_grad=True)
    semantic_codes = torch.randint(0, 32, (2, 1, 5))
    lengths = torch.tensor([5, 3])

    torch.testing.assert_close(
        predictor.estimator.output_projection.weight,
        torch.zeros_like(predictor.estimator.output_projection.weight),
    )
    torch.testing.assert_close(
        predictor.estimator.output_projection.bias,
        torch.zeros_like(predictor.estimator.output_projection.bias),
    )

    loss = predictor.compute_loss(acoustic, condition, lengths, semantic_codes=semantic_codes)
    loss.backward()

    assert predictor.requires_semantic_codes
    assert torch.isfinite(loss)
    assert condition.grad is not None
    assert predictor.estimator.semantic_embeddings[0].weight.grad is not None
    assert predictor.estimator.output_projection.weight.grad is not None

    predictor.eval()
    prediction = predictor.predict(
        condition.detach(),
        lengths,
        semantic_codes=semantic_codes,
        unconditional_condition=torch.zeros_like(condition),
        cfg_scale=1.3,
    )
    assert prediction.shape == acoustic.shape
    assert torch.isfinite(prediction).all()
    assert torch.count_nonzero(prediction[1, :, 3:]) == 0
