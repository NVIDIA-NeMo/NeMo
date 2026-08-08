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

# MIT License
#
# Copyright (c) 2021 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Conditional one-shot local flow for EasyMagpie.

The residual coupling definition is adapted from the VITS implementation:
https://github.com/jaywalnut310/vits
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from nemo.collections.tts.modules.magpietts_oneshot import OneShotLocalPredictor


AFFINE_COUPLING = "affine"
SPLINE_COUPLING = "rational_quadratic_spline"


def _canonical_coupling_type(coupling_type: str) -> str:
    coupling_type = coupling_type.lower()
    if coupling_type == "spline":
        return SPLINE_COUPLING
    if coupling_type not in (AFFINE_COUPLING, SPLINE_COUPLING):
        raise ValueError(
            f"Unsupported flow coupling type {coupling_type!r}; expected "
            f"{AFFINE_COUPLING!r}, {SPLINE_COUPLING!r}, or 'spline'."
        )
    return coupling_type


def _coupling_parameter_count(
    channels: int,
    condition_channels: int,
    hidden_channels: int,
    n_layers: int,
    output_parameters_per_channel: int,
) -> int:
    """Return the trainable parameter count of one pointwise coupling layer."""
    half_channels = channels // 2
    output_channels = output_parameters_per_channel * half_channels
    return (
        4 * n_layers * hidden_channels**2
        + hidden_channels * (half_channels + condition_channels + 2 + 3 * n_layers + output_channels)
        + output_channels
    )


def _parameter_matched_spline_hidden_channels(
    channels: int,
    condition_channels: int,
    affine_hidden_channels: int,
    n_layers: int,
    num_bins: int,
) -> int:
    """Choose the spline conditioner width closest to the affine parameter budget."""
    target = _coupling_parameter_count(
        channels=channels,
        condition_channels=condition_channels,
        hidden_channels=affine_hidden_channels,
        n_layers=n_layers,
        output_parameters_per_channel=2,
    )
    half_channels = channels // 2
    output_channels = (3 * num_bins - 1) * half_channels
    quadratic_coefficient = 4 * n_layers
    linear_coefficient = half_channels + condition_channels + 2 + 3 * n_layers + output_channels
    discriminant = linear_coefficient**2 + 4 * quadratic_coefficient * (target - output_channels)
    positive_root = (-linear_coefficient + math.sqrt(discriminant)) / (2 * quadratic_coefficient)
    candidates = {max(1, math.floor(positive_root)), max(1, math.ceil(positive_root))}
    return min(
        candidates,
        key=lambda hidden_channels: abs(
            _coupling_parameter_count(
                channels=channels,
                condition_channels=condition_channels,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
                output_parameters_per_channel=3 * num_bins - 1,
            )
            - target
        ),
    )


class PointwiseResidualConditioner(nn.Module):
    """Per-frame residual MLP implemented with pointwise projections."""

    def __init__(
        self,
        input_channels: int,
        condition_channels: int,
        hidden_channels: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1)
        self.condition_projection = nn.Conv1d(condition_channels, hidden_channels, 1)
        self.residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_channels, 2 * hidden_channels, 1),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(2 * hidden_channels, hidden_channels, 1),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            raise ValueError("Pointwise flow conditioner requires a conditioning tensor.")

        hidden = (self.input_projection(inputs) + self.condition_projection(condition)) * mask
        for residual_layer in self.residual_layers:
            hidden = (hidden + residual_layer(hidden)) * mask
        return hidden


class ChannelFlip(nn.Module):
    """Reverse the channel order between coupling layers."""

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, condition=None, reverse: bool = False):
        del mask, condition
        output = torch.flip(inputs, dims=[1])
        if reverse:
            return output
        return output, inputs.new_zeros(inputs.size(0))


class PointwiseAffineCoupling(nn.Module):
    """Conditional affine coupling transform with no temporal mixing."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        n_layers: int,
        condition_channels: int,
        dropout: float = 0.0,
        log_scale_limit: float = 2.0,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"Flow channels must be divisible by two, got {channels}")
        if log_scale_limit <= 0.0:
            raise ValueError(f"log_scale_limit must be positive, got {log_scale_limit}")

        self.half_channels = channels // 2
        self.log_scale_limit = log_scale_limit
        self.conditioner = PointwiseResidualConditioner(
            input_channels=self.half_channels,
            condition_channels=condition_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.output_projection = nn.Conv1d(hidden_channels, 2 * self.half_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        first_half, second_half = inputs.chunk(2, dim=1)
        hidden = self.conditioner(first_half * mask, mask, condition)
        shift, unconstrained_log_scale = self.output_projection(hidden).chunk(2, dim=1)
        shift = shift * mask
        log_scale = self.log_scale_limit * torch.tanh(unconstrained_log_scale / self.log_scale_limit) * mask

        if reverse:
            second_half = (second_half - shift) * torch.exp(-log_scale) * mask
            return torch.cat([first_half, second_half], dim=1)

        second_half = (shift + second_half * torch.exp(log_scale)) * mask
        output = torch.cat([first_half, second_half], dim=1)
        log_determinant = torch.sum(log_scale * mask, dim=(1, 2))
        return output, log_determinant


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a monotonic rational-quadratic spline on ``[-tail_bound, tail_bound]``."""
    computation_dtype = torch.float32 if inputs.dtype in (torch.float16, torch.bfloat16) else inputs.dtype
    inputs = inputs.to(computation_dtype)
    unnormalized_widths = unnormalized_widths.to(computation_dtype)
    unnormalized_heights = unnormalized_heights.to(computation_dtype)
    unnormalized_derivatives = unnormalized_derivatives.to(computation_dtype)

    num_bins = unnormalized_widths.size(-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * F.softmax(unnormalized_widths, dim=-1)
    cumulative_widths = F.pad(torch.cumsum(widths, dim=-1), (1, 0))
    cumulative_widths = 2.0 * tail_bound * cumulative_widths - tail_bound
    widths = cumulative_widths[..., 1:] - cumulative_widths[..., :-1]

    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * F.softmax(unnormalized_heights, dim=-1)
    cumulative_heights = F.pad(torch.cumsum(heights, dim=-1), (1, 0))
    cumulative_heights = 2.0 * tail_bound * cumulative_heights - tail_bound
    heights = cumulative_heights[..., 1:] - cumulative_heights[..., :-1]

    internal_derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    derivatives = F.pad(internal_derivatives, (1, 1), value=1.0)

    bin_boundaries = cumulative_heights if inverse else cumulative_widths
    bin_indices = torch.sum(inputs.unsqueeze(-1) >= bin_boundaries[..., 1:], dim=-1).clamp(max=num_bins - 1)

    def _gather(values: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, dim=-1, index=bin_indices.unsqueeze(-1)).squeeze(-1)

    input_cumulative_widths = _gather(cumulative_widths[..., :-1])
    input_bin_widths = _gather(widths)
    input_cumulative_heights = _gather(cumulative_heights[..., :-1])
    input_bin_heights = _gather(heights)
    input_delta = input_bin_heights / input_bin_widths
    input_derivatives = _gather(derivatives[..., :-1])
    input_derivatives_plus_one = _gather(derivatives[..., 1:])
    derivative_sum_minus_delta = input_derivatives_plus_one + input_derivatives - 2.0 * input_delta

    if inverse:
        shifted_inputs = inputs - input_cumulative_heights
        quadratic_a = shifted_inputs * derivative_sum_minus_delta + input_bin_heights * (
            input_delta - input_derivatives
        )
        quadratic_b = input_bin_heights * input_derivatives - shifted_inputs * derivative_sum_minus_delta
        quadratic_c = -input_delta * shifted_inputs
        discriminant = (quadratic_b.square() - 4.0 * quadratic_a * quadratic_c).clamp_min(0.0)
        theta = 2.0 * quadratic_c / (-quadratic_b - torch.sqrt(discriminant))
        theta = theta.clamp(0.0, 1.0)
        outputs = input_cumulative_widths + theta * input_bin_widths
    else:
        theta = ((inputs - input_cumulative_widths) / input_bin_widths).clamp(0.0, 1.0)

    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = input_delta + derivative_sum_minus_delta * theta_one_minus_theta

    if not inverse:
        numerator = input_bin_heights * (input_delta * theta.square() + input_derivatives * theta_one_minus_theta)
        outputs = input_cumulative_heights + numerator / denominator

    derivative_numerator = input_delta.square() * (
        input_derivatives_plus_one * theta.square()
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta).square()
    )
    log_absolute_determinant = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
    if inverse:
        log_absolute_determinant = -log_absolute_determinant
    return outputs, log_absolute_determinant


class PointwiseRationalQuadraticSplineCoupling(nn.Module):
    """Conditional monotonic rational-quadratic spline coupling with linear tails."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        n_layers: int,
        condition_channels: int,
        dropout: float = 0.0,
        num_bins: int = 8,
        tail_bound: float = 5.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"Flow channels must be divisible by two, got {channels}")
        if num_bins < 2:
            raise ValueError(f"Spline num_bins must be at least two, got {num_bins}")
        if tail_bound <= 0.0:
            raise ValueError(f"Spline tail_bound must be positive, got {tail_bound}")
        if not 0.0 < min_bin_width < 1.0 / num_bins:
            raise ValueError(f"min_bin_width must be in (0, {1.0 / num_bins}), got {min_bin_width}")
        if not 0.0 < min_bin_height < 1.0 / num_bins:
            raise ValueError(f"min_bin_height must be in (0, {1.0 / num_bins}), got {min_bin_height}")
        if min_derivative <= 0.0:
            raise ValueError(f"min_derivative must be positive, got {min_derivative}")

        self.half_channels = channels // 2
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.parameters_per_channel = 3 * num_bins - 1
        self.conditioner = PointwiseResidualConditioner(
            input_channels=self.half_channels,
            condition_channels=condition_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.output_projection = nn.Conv1d(hidden_channels, self.parameters_per_channel * self.half_channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        identity_derivative = math.log(math.expm1(1.0 - min_derivative))
        with torch.no_grad():
            projection_bias = self.output_projection.bias.view(self.half_channels, self.parameters_per_channel)
            projection_bias[:, 2 * num_bins :].fill_(identity_derivative)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        first_half, second_half = inputs.chunk(2, dim=1)
        hidden = self.conditioner(first_half * mask, mask, condition)
        parameters = self.output_projection(hidden)
        parameters = parameters.view(
            inputs.size(0), self.half_channels, self.parameters_per_channel, inputs.size(2)
        ).permute(0, 1, 3, 2)
        unnormalized_widths = parameters[..., : self.num_bins]
        unnormalized_heights = parameters[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = parameters[..., 2 * self.num_bins :]

        transformed, elementwise_log_determinant = _rational_quadratic_spline(
            inputs=second_half.clamp(-self.tail_bound, self.tail_bound),
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        valid = mask.bool() & (second_half >= -self.tail_bound) & (second_half <= self.tail_bound)
        second_half = torch.where(valid, transformed.to(second_half.dtype), second_half) * mask
        output = torch.cat([first_half, second_half], dim=1)

        if reverse:
            return output

        elementwise_log_determinant = torch.where(
            valid, elementwise_log_determinant, torch.zeros_like(elementwise_log_determinant)
        )
        return output, elementwise_log_determinant.sum(dim=(1, 2))


class PointwiseCouplingBlock(nn.Module):
    """Stack pointwise coupling layers and channel flips."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        n_layers: int,
        n_flows: int = 4,
        condition_channels: int = 0,
        dropout: float = 0.0,
        coupling_type: str = AFFINE_COUPLING,
        spline_num_bins: int = 8,
        spline_tail_bound: float = 5.0,
        spline_min_bin_width: float = 1e-3,
        spline_min_bin_height: float = 1e-3,
        spline_min_derivative: float = 1e-3,
    ):
        super().__init__()
        coupling_type = _canonical_coupling_type(coupling_type)
        flows = []
        for _ in range(n_flows):
            if coupling_type == AFFINE_COUPLING:
                coupling = PointwiseAffineCoupling(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    n_layers=n_layers,
                    condition_channels=condition_channels,
                    dropout=dropout,
                )
            else:
                coupling = PointwiseRationalQuadraticSplineCoupling(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    n_layers=n_layers,
                    condition_channels=condition_channels,
                    dropout=dropout,
                    num_bins=spline_num_bins,
                    tail_bound=spline_tail_bound,
                    min_bin_width=spline_min_bin_width,
                    min_bin_height=spline_min_bin_height,
                    min_derivative=spline_min_derivative,
                )
            flows.append(coupling)
            flows.append(ChannelFlip())
        self.flows = nn.ModuleList(flows)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            for flow in reversed(self.flows):
                inputs = flow(inputs, mask, condition=condition, reverse=True)
            return inputs

        total_log_determinant = inputs.new_zeros(inputs.size(0))
        for flow in self.flows:
            inputs, log_determinant = flow(inputs, mask, condition=condition, reverse=False)
            total_log_determinant = total_log_determinant + log_determinant
        return inputs, total_log_determinant


class OneShotLocalFlow(OneShotLocalPredictor):
    """Conditional pointwise flow over a continuous FSQ-compressed acoustic codec embedding."""

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        hidden_channels: int,
        n_layers: int = 3,
        n_flows: int = 4,
        dropout: float = 0.0,
        coupling_type: str = AFFINE_COUPLING,
        spline_num_bins: int = 8,
        spline_tail_bound: float = 5.0,
        spline_min_bin_width: float = 1e-3,
        spline_min_bin_height: float = 1e-3,
        spline_min_derivative: float = 1e-3,
        match_affine_parameter_count: bool = True,
    ):
        super().__init__(acoustic_channels=acoustic_channels)
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        self.flow_channels = acoustic_channels + acoustic_channels % 2
        self.coupling_type = _canonical_coupling_type(coupling_type)
        self.requested_hidden_channels = hidden_channels
        if self.coupling_type == SPLINE_COUPLING and match_affine_parameter_count:
            hidden_channels = _parameter_matched_spline_hidden_channels(
                channels=self.flow_channels,
                condition_channels=condition_channels,
                affine_hidden_channels=hidden_channels,
                n_layers=n_layers,
                num_bins=spline_num_bins,
            )
        self.hidden_channels = hidden_channels
        self.flow = PointwiseCouplingBlock(
            channels=self.flow_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_flows=n_flows,
            condition_channels=condition_channels,
            dropout=dropout,
            coupling_type=self.coupling_type,
            spline_num_bins=spline_num_bins,
            spline_tail_bound=spline_tail_bound,
            spline_min_bin_width=spline_min_bin_width,
            spline_min_bin_height=spline_min_bin_height,
            spline_min_derivative=spline_min_derivative,
        )

    def encode(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if acoustic_embedding.shape[0] != condition.shape[0] or acoustic_embedding.shape[2] != condition.shape[2]:
            raise ValueError(
                "Acoustic target and condition must share batch/time dimensions, got "
                f"{acoustic_embedding.shape} and {condition.shape}"
            )

        if self.flow_channels != self.acoustic_channels:
            acoustic_embedding = torch.nn.functional.pad(acoustic_embedding, (0, 0, 0, 1))
        mask = self.length_mask(lengths, acoustic_embedding.size(2), acoustic_embedding.dtype)
        if frame_mask is not None:
            mask = mask * frame_mask.unsqueeze(1).to(mask.dtype)
        latent, log_determinant = self.flow(
            acoustic_embedding * mask,
            mask,
            condition=condition * mask,
            reverse=False,
        )
        return latent, mask, log_determinant

    def decode(self, latent: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        acoustic = self.flow(latent * mask, mask, condition=condition * mask, reverse=True) * mask
        return acoustic[:, : self.acoustic_channels]

    def compute_loss(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent, mask, log_determinant = self.encode(acoustic_embedding, condition, lengths, frame_mask=frame_mask)
        negative_log_likelihood = 0.5 * (latent.float().square() + math.log(2.0 * math.pi))
        denominator = (mask.sum() * latent.size(1)).clamp_min(1.0)
        return ((negative_log_likelihood * mask.float()).sum() - log_determinant.float().sum()) / denominator

    @torch.no_grad()
    def compute_diagnostics(
        self,
        acoustic_embedding: torch.Tensor,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return compact diagnostics for the worst sample in a spiking batch."""
        latent, mask, log_determinant = self.encode(
            acoustic_embedding,
            condition,
            lengths,
            frame_mask=frame_mask,
        )
        float_mask = mask.float()
        negative_log_likelihood = 0.5 * (latent.float().square() + math.log(2.0 * math.pi))
        valid_frames = float_mask.sum(dim=(1, 2))
        latent_denominator = (valid_frames * latent.size(1)).clamp_min(1.0)
        per_sample_loss = (
            (negative_log_likelihood * float_mask).sum(dim=(1, 2)) - log_determinant.float()
        ) / latent_denominator
        worst_sample_index = per_sample_loss.argmax()

        def _masked_abs_max(values: torch.Tensor) -> torch.Tensor:
            return (values.float().abs() * float_mask).amax(dim=(1, 2))

        def _masked_rms(values: torch.Tensor) -> torch.Tensor:
            denominator = (valid_frames * values.size(1)).clamp_min(1.0)
            return ((values.float().square() * float_mask).sum(dim=(1, 2)) / denominator).sqrt()

        return {
            "sample_index": worst_sample_index,
            "sample_loss": per_sample_loss[worst_sample_index],
            "valid_frames": valid_frames[worst_sample_index],
            "target_abs_max": _masked_abs_max(acoustic_embedding)[worst_sample_index],
            "target_rms": _masked_rms(acoustic_embedding)[worst_sample_index],
            "condition_abs_max": _masked_abs_max(condition)[worst_sample_index],
            "condition_rms": _masked_rms(condition)[worst_sample_index],
            "latent_abs_max": _masked_abs_max(latent)[worst_sample_index],
            "latent_rms": _masked_rms(latent)[worst_sample_index],
            "normalized_log_determinant": (
                log_determinant.float()[worst_sample_index] / latent_denominator[worst_sample_index]
            ),
        }

    def predict(
        self,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        mask = self.length_mask(lengths, condition.size(2), condition.dtype)
        latent = torch.randn(
            condition.size(0),
            self.flow_channels,
            condition.size(2),
            device=condition.device,
            dtype=condition.dtype,
        )
        latent = latent * noise_scale
        return self.decode(latent, condition, mask)

    def sample(
        self,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compatibility alias for downstream callers using the old API."""
        return self.predict(condition=condition, lengths=lengths, noise_scale=noise_scale)
