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


class _ConditionVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, condition, time, mask):
        del time
        self.calls += 1
        return condition[:, : state.size(1)] * mask


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
