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

from nemo.collections.tts.modules.magpietts_flow import (
    OneShotLocalFlow,
    PointwiseAffineCoupling,
    PointwiseRationalQuadraticSplineCoupling,
)


pytestmark = pytest.mark.unit


def _make_non_identity_flow() -> OneShotLocalFlow:
    torch.manual_seed(42)
    flow = OneShotLocalFlow(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        n_layers=2,
        n_flows=2,
    )
    with torch.no_grad():
        for module in flow.modules():
            if isinstance(module, PointwiseAffineCoupling):
                torch.nn.init.normal_(module.output_projection.weight, std=0.01)
                torch.nn.init.normal_(module.output_projection.bias, std=0.01)
    return flow


def _make_non_identity_spline_flow() -> OneShotLocalFlow:
    torch.manual_seed(42)
    flow = OneShotLocalFlow(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        n_layers=2,
        n_flows=2,
        coupling_type="rational_quadratic_spline",
        spline_num_bins=4,
    )
    with torch.no_grad():
        for module in flow.modules():
            if isinstance(module, PointwiseRationalQuadraticSplineCoupling):
                torch.nn.init.normal_(module.output_projection.weight, std=0.01)
    return flow


@pytest.mark.parametrize("flow_factory", [_make_non_identity_flow, _make_non_identity_spline_flow])
def test_one_shot_local_flow_couplings_are_invertible_on_valid_frames(flow_factory):
    flow = flow_factory()
    acoustic = torch.randn(2, 12, 5).clamp(-2.0, 2.0)
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])

    latent, mask, log_determinant = flow.encode(acoustic, condition, lengths)
    restored = flow.decode(latent, condition, mask)

    assert torch.allclose(restored * mask, acoustic * mask, atol=2e-5, rtol=2e-5)
    assert torch.any(log_determinant != 0.0)


def test_one_shot_local_flow_is_invertible_on_valid_frames():
    flow = _make_non_identity_flow()
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])

    latent, mask, log_determinant = flow.encode(acoustic, condition, lengths)
    restored = flow.decode(latent, condition, mask)

    assert torch.allclose(restored * mask, acoustic * mask, atol=1e-5, rtol=1e-5)
    assert torch.any(log_determinant != 0.0)


def test_one_shot_local_flow_is_frame_permutation_equivariant():
    flow = _make_non_identity_flow().eval()
    acoustic = torch.randn(2, 12, 5)
    condition = torch.randn(2, 20, 5)
    lengths = torch.full((2,), 5)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    latent, _, log_determinant = flow.encode(acoustic, condition, lengths)
    permuted_latent, _, permuted_log_determinant = flow.encode(
        acoustic[:, :, permutation], condition[:, :, permutation], lengths
    )

    assert torch.allclose(permuted_latent, latent[:, :, permutation], atol=1e-6, rtol=1e-6)
    assert torch.allclose(permuted_log_determinant, log_determinant, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("flow_factory", [_make_non_identity_flow, _make_non_identity_spline_flow])
def test_one_shot_local_flow_loss_is_finite_and_backpropagates(flow_factory):
    flow = flow_factory()
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


def test_one_shot_local_flow_diagnostics_select_worst_sample():
    flow = _make_non_identity_flow()
    acoustic = torch.randn(2, 12, 5)
    acoustic[1] *= 100.0
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 4])

    diagnostics = flow.compute_diagnostics(acoustic, condition, lengths)

    assert set(diagnostics) == {
        "sample_index",
        "sample_loss",
        "valid_frames",
        "target_abs_max",
        "target_rms",
        "condition_abs_max",
        "condition_rms",
        "latent_abs_max",
        "latent_rms",
        "normalized_log_determinant",
    }
    assert diagnostics["sample_index"].item() == 1
    assert diagnostics["valid_frames"].item() == 4
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert diagnostics["target_abs_max"] > diagnostics["condition_abs_max"]


def test_spline_flow_matches_affine_parameter_budget():
    common_kwargs = {
        "acoustic_channels": 12,
        "condition_channels": 20,
        "hidden_channels": 64,
        "n_layers": 2,
        "n_flows": 2,
    }
    affine_flow = OneShotLocalFlow(**common_kwargs)
    spline_flow = OneShotLocalFlow(
        **common_kwargs,
        coupling_type="spline",
        spline_num_bins=4,
        match_affine_parameter_count=True,
    )

    affine_parameters = sum(parameter.numel() for parameter in affine_flow.parameters())
    spline_parameters = sum(parameter.numel() for parameter in spline_flow.parameters())

    assert spline_flow.coupling_type == "rational_quadratic_spline"
    assert spline_flow.requested_hidden_channels == 64
    assert spline_flow.hidden_channels < spline_flow.requested_hidden_channels
    assert abs(spline_parameters - affine_parameters) / affine_parameters < 0.01


def test_spline_flow_is_identity_initialized_and_has_linear_tails():
    flow = OneShotLocalFlow(
        acoustic_channels=12,
        condition_channels=20,
        hidden_channels=16,
        n_layers=2,
        n_flows=2,
        coupling_type="spline",
        spline_num_bins=4,
        spline_tail_bound=3.0,
    )
    acoustic = torch.full((2, 12, 5), 4.0)
    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])

    latent, mask, log_determinant = flow.encode(acoustic, condition, lengths)

    assert torch.equal(latent * mask, acoustic * mask)
    assert torch.equal(log_determinant, torch.zeros_like(log_determinant))


def test_spline_coupling_log_determinant_matches_autograd_jacobian():
    torch.manual_seed(42)
    coupling = PointwiseRationalQuadraticSplineCoupling(
        channels=2,
        condition_channels=3,
        hidden_channels=4,
        n_layers=1,
        num_bins=4,
        tail_bound=3.0,
    ).double()
    with torch.no_grad():
        torch.nn.init.normal_(coupling.output_projection.weight, std=0.1)

    inputs = torch.tensor([[[0.2], [-0.4]]], dtype=torch.double)
    condition = torch.randn(1, 3, 1, dtype=torch.double)
    mask = torch.ones(1, 1, 1, dtype=torch.double)

    def _transform(flattened_inputs):
        outputs, _ = coupling(flattened_inputs.view_as(inputs), mask, condition)
        return outputs.flatten()

    jacobian = torch.autograd.functional.jacobian(_transform, inputs.flatten())
    _, expected_log_determinant = torch.linalg.slogdet(jacobian)
    _, log_determinant = coupling(inputs, mask, condition)

    assert torch.allclose(log_determinant.squeeze(0), expected_log_determinant, atol=1e-8, rtol=1e-8)
