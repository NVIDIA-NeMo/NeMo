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
    """Reverse the channel order between affine coupling layers."""

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


class PointwiseCouplingBlock(nn.Module):
    """Stack pointwise affine coupling layers and channel flips."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        n_layers: int,
        n_flows: int = 4,
        condition_channels: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        flows = []
        for _ in range(n_flows):
            flows.append(
                PointwiseAffineCoupling(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    n_layers=n_layers,
                    condition_channels=condition_channels,
                    dropout=dropout,
                )
            )
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


class OneShotLocalFlow(nn.Module):
    """Conditional pointwise flow over a pre-quantization acoustic codec embedding."""

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        hidden_channels: int,
        n_layers: int = 3,
        n_flows: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.acoustic_channels = acoustic_channels
        self.flow_channels = acoustic_channels + acoustic_channels % 2
        self.flow = PointwiseCouplingBlock(
            channels=self.flow_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_flows=n_flows,
            condition_channels=condition_channels,
            dropout=dropout,
        )

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_length: int, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(max_length, device=lengths.device)
        return (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).to(dtype)

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
        mask = self._length_mask(lengths, acoustic_embedding.size(2), acoustic_embedding.dtype)
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

    def sample(
        self,
        condition: torch.Tensor,
        lengths: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        mask = self._length_mask(lengths, condition.size(2), condition.dtype)
        latent = torch.randn(
            condition.size(0),
            self.flow_channels,
            condition.size(2),
            device=condition.device,
            dtype=condition.dtype,
        )
        latent = latent * noise_scale
        return self.decode(latent, condition, mask)
