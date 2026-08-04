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

from nemo.collections.tts.modules.magpietts_flow import OneShotLocalFlow


pytestmark = pytest.mark.unit


def test_one_shot_local_flow_is_invertible_on_valid_frames():
    torch.manual_seed(42)
    flow = OneShotLocalFlow(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        kernel_size=3,
        dilation_rate=1,
        n_layers=2,
        n_flows=2,
    )
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])

    latent, mask = flow.encode(acoustic, condition, lengths)
    restored = flow.decode(latent, condition, mask)

    assert torch.allclose(restored * mask, acoustic * mask, atol=1e-5, rtol=1e-5)


def test_one_shot_local_flow_loss_is_finite_and_backpropagates():
    torch.manual_seed(42)
    flow = OneShotLocalFlow(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        kernel_size=3,
        dilation_rate=1,
        n_layers=2,
        n_flows=2,
    )
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5, requires_grad=True)
    lengths = torch.tensor([5, 4])

    loss = flow.compute_loss(acoustic, condition, lengths)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert condition.grad is not None
    assert torch.isfinite(condition.grad).all()
    assert any(parameter.grad is not None for parameter in flow.parameters())
