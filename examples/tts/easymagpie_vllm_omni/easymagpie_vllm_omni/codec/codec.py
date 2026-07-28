# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""NeMo-free EasyMagpie spectral codec implementation.

This file deliberately contains a plain PyTorch implementation first. It is the
numeric oracle for the packed vLLM kernels and also provides a useful eager
fallback while the cache-aware path is being integrated with a runner.

It intentionally does not inherit NeMo's training-time ``ResNetDecoder``:
serving must remain installable without NeMo Speech, and the production vLLM
path uses packed request batches plus explicit cache state rather than NeMo's
length-masked ``NeuralModule`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packing import unstack_acoustic_codes


class FiniteScalarDequantizer(nn.Module):
    """Decode grouped FSQ indices directly to the codec's continuous latent."""

    def __init__(self, num_groups: int, num_levels_per_group: list[int]) -> None:
        super().__init__()
        levels = torch.tensor(num_levels_per_group, dtype=torch.int64)
        bases = torch.cumprod(torch.tensor([1, *num_levels_per_group[:-1]], dtype=torch.int64), dim=0)
        self.num_groups = int(num_groups)
        self.group_dim = len(num_levels_per_group)
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("bases", bases, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert ``[..., groups]`` integer indices to ``[..., groups*D]``."""
        if indices.shape[-1] != self.num_groups:
            raise ValueError(f"expected {self.num_groups} FSQ groups, got shape {tuple(indices.shape)}")
        nonnegative = torch.div(indices.unsqueeze(-1), self.bases, rounding_mode="floor") % self.levels
        scale = torch.div(self.levels, 2, rounding_mode="floor")
        codes = (nonnegative - scale) / scale
        return codes.flatten(start_dim=-2)


class HalfSnake(nn.Module):
    """Snake on the first half of channels and LeakyReLU on the second."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.snake_channels = channels // 2
        self.alpha = nn.Parameter(torch.ones(1, self.snake_channels, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        snake_in = inputs[:, : self.snake_channels]
        alpha = self.alpha
        snake_out = snake_in + torch.sin(alpha * snake_in).square() / (alpha + 1e-9)
        return torch.cat((snake_out, F.leaky_relu(inputs[:, self.snake_channels :])), dim=1)


class CausalConv1d(nn.Module):
    """Dense causal Conv1d with an explicit fixed-size streaming history."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, activate: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.history = kernel_size - 1
        self.activation = HalfSnake(out_channels) if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(F.pad(inputs, (self.history, 0))))

    def stream(self, inputs: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = inputs.new_zeros((inputs.shape[0], inputs.shape[1], self.history))
        if state.shape != (inputs.shape[0], inputs.shape[1], self.history):
            raise ValueError(
                f"invalid Conv1d state {tuple(state.shape)} for input {tuple(inputs.shape)} and history {self.history}"
            )
        joined = torch.cat((state, inputs), dim=-1)
        outputs = self.activation(self.conv(joined))
        return outputs, joined[..., -self.history :]


class CausalConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d with one input frame of streaming state."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, *, activate: bool = True) -> None:
        super().__init__()
        groups = out_channels
        if in_channels % groups != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by groups={groups}")
        self.stride = int(stride)
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            groups=groups,
        )
        self.activation = HalfSnake(out_channels) if activate else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        return self.activation(outputs[..., : -self.stride])

    def stream(self, inputs: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = inputs.new_zeros((inputs.shape[0], inputs.shape[1], 1))
        if state.shape != (inputs.shape[0], inputs.shape[1], 1):
            raise ValueError(f"invalid ConvTranspose1d state {tuple(state.shape)} for input {tuple(inputs.shape)}")
        joined = torch.cat((state, inputs), dim=-1)
        outputs = self.conv(joined)
        # The prepended history produces the first stride samples. The final
        # stride samples are the causal right trim used by NeMo.
        outputs = outputs[..., self.stride : -self.stride]
        return self.activation(outputs), joined[..., -1:]


@dataclass
class ResidualBlockState:
    input_conv: torch.Tensor | None = None
    skip_conv: torch.Tensor | None = None


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, filters: int, kernel_size: int) -> None:
        super().__init__()
        self.input_conv = CausalConv1d(channels, filters, kernel_size, activate=True)
        self.skip_conv = CausalConv1d(filters, channels, kernel_size)
        self.output_activation = HalfSnake(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_activation(inputs + self.skip_conv(self.input_conv(inputs)))

    def stream(
        self, inputs: torch.Tensor, state: ResidualBlockState | None
    ) -> tuple[torch.Tensor, ResidualBlockState]:
        state = state or ResidualBlockState()
        hidden, input_state = self.input_conv.stream(inputs, state.input_conv)
        hidden, skip_state = self.skip_conv.stream(hidden, state.skip_conv)
        return self.output_activation(inputs + hidden), ResidualBlockState(input_state, skip_state)


@dataclass
class DecoderState:
    pre_conv: torch.Tensor | None
    pre_resblocks: list[ResidualBlockState]
    pre_upsample: list[torch.Tensor | None]
    hidden_blocks: list[ResidualBlockState]
    upsample: list[torch.Tensor | None]
    upsample_resblocks: list[ResidualBlockState]
    post_conv: torch.Tensor | None

    @classmethod
    def empty(cls, config: EasyMagpieCodecConfig) -> "DecoderState":
        return cls(
            pre_conv=None,
            pre_resblocks=[ResidualBlockState() for _ in config.pre_upsample_rates],
            pre_upsample=[None for _ in config.pre_upsample_rates],
            hidden_blocks=[ResidualBlockState() for _ in range(config.num_hidden_layers)],
            upsample=[None for _ in config.resblock_upsample_rates],
            upsample_resblocks=[ResidualBlockState() for _ in config.resblock_upsample_rates],
            post_conv=None,
        )


class ResNetDecoder(nn.Module):
    """The causal decoder from ``25fps_spectral_codec_with_bandwidth_extension``."""

    def __init__(self, config: EasyMagpieCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_conv = CausalConv1d(config.input_dim, config.input_filters, config.kernel_size)

        channels = config.input_filters
        self.pre_resblocks = nn.ModuleList()
        self.pre_up_sample_layers = nn.ModuleList()
        for rate, filters in zip(config.pre_upsample_rates, config.pre_upsample_filters):
            self.pre_resblocks.append(ResidualBlock(channels, 2 * channels, config.kernel_size))
            self.pre_up_sample_layers.append(CausalConvTranspose1d(channels, filters, rate))
            channels = filters

        self.conv_layers = nn.ModuleList(
            ResidualBlock(channels, config.hidden_filters, config.kernel_size) for _ in range(config.num_hidden_layers)
        )

        self.resblock_up_sample_layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for rate, filters in zip(config.resblock_upsample_rates, config.resblock_upsample_filters):
            self.resblock_up_sample_layers.append(CausalConvTranspose1d(channels, filters, rate))
            self.resblocks.append(ResidualBlock(filters, 2 * filters, config.resblock_kernel_size))
            channels = filters

        self.post_conv = CausalConv1d(channels, 1, config.resblock_kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.pre_conv(inputs)
        for block, upsample in zip(self.pre_resblocks, self.pre_up_sample_layers):
            hidden = upsample(block(hidden))
        for block in self.conv_layers:
            hidden = block(hidden)
        for upsample, block in zip(self.resblock_up_sample_layers, self.resblocks):
            hidden = block(upsample(hidden))
        return self.post_conv(hidden).squeeze(1).clamp(-1.0, 1.0)

    def stream(self, inputs: torch.Tensor, state: DecoderState | None = None) -> tuple[torch.Tensor, DecoderState]:
        state = state or DecoderState.empty(self.config)
        hidden, pre_conv_state = self.pre_conv.stream(inputs, state.pre_conv)

        pre_resblock_states: list[ResidualBlockState] = []
        pre_upsample_states: list[torch.Tensor] = []
        for block, upsample, block_state, upsample_state in zip(
            self.pre_resblocks,
            self.pre_up_sample_layers,
            state.pre_resblocks,
            state.pre_upsample,
        ):
            hidden, block_state = block.stream(hidden, block_state)
            hidden, upsample_state = upsample.stream(hidden, upsample_state)
            pre_resblock_states.append(block_state)
            pre_upsample_states.append(upsample_state)

        hidden_block_states: list[ResidualBlockState] = []
        for block, block_state in zip(self.conv_layers, state.hidden_blocks):
            hidden, block_state = block.stream(hidden, block_state)
            hidden_block_states.append(block_state)

        upsample_states: list[torch.Tensor] = []
        upsample_resblock_states: list[ResidualBlockState] = []
        for upsample, block, upsample_state, block_state in zip(
            self.resblock_up_sample_layers,
            self.resblocks,
            state.upsample,
            state.upsample_resblocks,
        ):
            hidden, upsample_state = upsample.stream(hidden, upsample_state)
            hidden, block_state = block.stream(hidden, block_state)
            upsample_states.append(upsample_state)
            upsample_resblock_states.append(block_state)

        hidden, post_conv_state = self.post_conv.stream(hidden, state.post_conv)
        new_state = DecoderState(
            pre_conv=pre_conv_state,
            pre_resblocks=pre_resblock_states,
            pre_upsample=pre_upsample_states,
            hidden_blocks=hidden_block_states,
            upsample=upsample_states,
            upsample_resblocks=upsample_resblock_states,
            post_conv=post_conv_state,
        )
        return hidden.squeeze(1).clamp(-1.0, 1.0), new_state


class EasyMagpieCodec(nn.Module):
    """Stacked EasyMagpie codes to a waveform, with optional streaming state."""

    def __init__(self, config: EasyMagpieCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.dequantizer = FiniteScalarDequantizer(config.num_codebooks, config.num_levels_per_group)
        self.audio_decoder = ResNetDecoder(config)

    def _codes_to_latent(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 3 or codes.shape[-1] != self.config.num_stacked_codebooks:
            raise ValueError(f"expected [B, T, {self.config.num_stacked_codebooks}] codes, got {tuple(codes.shape)}")
        unstacked = unstack_acoustic_codes(
            codes,
            num_codebooks=self.config.num_codebooks,
            frame_stacking_factor=self.config.frame_stacking_factor,
        )
        latent = self.dequantizer(unstacked.clamp(0, self.config.codebook_size - 1))
        return latent.transpose(1, 2).contiguous()

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.audio_decoder(self._codes_to_latent(codes))

    def stream(self, codes: torch.Tensor, state: DecoderState | None = None) -> tuple[torch.Tensor, DecoderState]:
        return self.audio_decoder.stream(self._codes_to_latent(codes), state)
