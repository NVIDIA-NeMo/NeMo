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


class VITSWaveNet(nn.Module):
    """Dilated WaveNet conditioner used by VITS residual coupling layers."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        condition_channels: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.input_layers = nn.ModuleList()
        self.residual_skip_layers = nn.ModuleList()

        if condition_channels > 0:
            condition_layer = nn.Conv1d(condition_channels, 2 * hidden_channels * n_layers, 1)
            self.condition_layer = nn.utils.weight_norm(condition_layer)
        else:
            self.condition_layer = None

        for layer_idx in range(n_layers):
            dilation = dilation_rate**layer_idx
            padding = (kernel_size * dilation - dilation) // 2
            input_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.input_layers.append(nn.utils.weight_norm(input_layer))

            output_channels = 2 * hidden_channels if layer_idx < n_layers - 1 else hidden_channels
            residual_skip_layer = nn.Conv1d(hidden_channels, output_channels, 1)
            self.residual_skip_layers.append(nn.utils.weight_norm(residual_skip_layer))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        output = torch.zeros_like(inputs)
        projected_condition = self.condition_layer(condition) if self.condition_layer is not None else None

        for layer_idx, (input_layer, residual_skip_layer) in enumerate(
            zip(self.input_layers, self.residual_skip_layers)
        ):
            activations = input_layer(inputs)
            if projected_condition is not None:
                offset = layer_idx * 2 * self.hidden_channels
                activations = activations + projected_condition[:, offset : offset + 2 * self.hidden_channels]

            tanh_part, sigmoid_part = activations.chunk(2, dim=1)
            activations = self.dropout(torch.tanh(tanh_part) * torch.sigmoid(sigmoid_part))
            residual_skip = residual_skip_layer(activations)

            if layer_idx < self.n_layers - 1:
                residual, skip = residual_skip.split(self.hidden_channels, dim=1)
                inputs = (inputs + residual) * mask
                output = output + skip
            else:
                output = output + residual_skip

        return output * mask


class ChannelFlip(nn.Module):
    """Reverse the channel order between affine coupling layers."""

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, condition=None, reverse: bool = False):
        del mask, condition
        output = torch.flip(inputs, dims=[1])
        if reverse:
            return output
        return output, inputs.new_zeros(inputs.size(0))


class ResidualCouplingLayer(nn.Module):
    """VITS residual affine coupling layer."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        condition_channels: int = 0,
        dropout: float = 0.0,
        mean_only: bool = True,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"Flow channels must be divisible by two, got {channels}")

        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.input_projection = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.conditioner = VITSWaveNet(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            condition_channels=condition_channels,
            dropout=dropout,
        )
        output_channels = self.half_channels if mean_only else 2 * self.half_channels
        self.output_projection = nn.Conv1d(hidden_channels, output_channels, 1)
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
        hidden = self.input_projection(first_half) * mask
        stats = self.output_projection(self.conditioner(hidden, mask, condition)) * mask

        if self.mean_only:
            mean = stats
            log_scale = torch.zeros_like(mean)
        else:
            mean, log_scale = stats.chunk(2, dim=1)

        if reverse:
            second_half = (second_half - mean) * torch.exp(-log_scale) * mask
            return torch.cat([first_half, second_half], dim=1)

        second_half = (mean + second_half * torch.exp(log_scale)) * mask
        output = torch.cat([first_half, second_half], dim=1)
        log_determinant = torch.sum(log_scale, dim=(1, 2))
        return output, log_determinant


class ResidualCouplingBlock(nn.Module):
    """Stack residual coupling layers and channel flips as in VITS."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        condition_channels: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        flows = []
        for _ in range(n_flows):
            flows.append(
                ResidualCouplingLayer(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    condition_channels=condition_channels,
                    dropout=dropout,
                    mean_only=True,
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
    ) -> torch.Tensor:
        if reverse:
            for flow in reversed(self.flows):
                inputs = flow(inputs, mask, condition=condition, reverse=True)
            return inputs

        for flow in self.flows:
            inputs, _ = flow(inputs, mask, condition=condition, reverse=False)
        return inputs


class OneShotLocalFlow(nn.Module):
    """Conditional normalizing flow over a stacked pre-quantization acoustic codec embedding."""

    def __init__(
        self,
        acoustic_channels: int,
        condition_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.acoustic_channels = acoustic_channels
        self.flow_channels = acoustic_channels + acoustic_channels % 2
        self.flow = ResidualCouplingBlock(
            channels=self.flow_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        latent = self.flow(
            acoustic_embedding * mask,
            mask,
            condition=condition * mask,
            reverse=False,
        )
        return latent, mask

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
        latent, mask = self.encode(acoustic_embedding, condition, lengths, frame_mask=frame_mask)
        negative_log_likelihood = 0.5 * (latent.float().square() + math.log(2.0 * math.pi))
        denominator = (mask.sum() * latent.size(1)).clamp_min(1.0)
        return (negative_log_likelihood * mask.float()).sum() / denominator

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
