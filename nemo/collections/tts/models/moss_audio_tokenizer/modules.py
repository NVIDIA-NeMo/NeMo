# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""Trainable NeMo modules for the MOSS Audio Tokenizer Nano architecture.

The transformer, patching, and LFQ topology is adapted from the Apache-2.0
MOSS-Audio-Tokenizer Nano implementation at upstream commit
8c50ac4c5d7287d2ed6ea20a08c90ca439887d23. The released upstream code is an
inference implementation and marks its codec and quantizer forwards no-grad.
This module retains the checkpoint-compatible topology while adding a
straight-through residual LFQ path and the codebook/commitment losses needed
to train it inside :class:`AudioCodecModel`.

Only the non-streaming path needed for NeMo training is included here. The
official implementation remains the reference for KV-cached streaming.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerBase
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types.elements import (
    AudioSignal,
    EncodedRepresentation,
    LengthsType,
    LossType,
    TokenIndex,
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType


def _to_plain_dict(value):
    """Convert DictConfig-like values without requiring OmegaConf here."""
    return {key: value[key] for key in value}


class _LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.scale * inputs


def _apply_rope(q: torch.Tensor, k: torch.Tensor, max_period: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same pairwise rotary embedding used by MOSS."""
    _, _, time, dim = q.shape
    if dim % 2:
        raise ValueError(f"RoPE requires an even head dimension, got {dim}")

    frequency = torch.exp(
        torch.arange(dim // 2, device=q.device, dtype=torch.float32) * (-math.log(max_period) * 2 / dim)
    )
    position = torch.arange(time, device=q.device, dtype=torch.float32).view(1, 1, time, 1)
    cosine = torch.cos(position * frequency)
    sine = torch.sin(position * frequency)

    def rotate(inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        paired = inputs.float().reshape(*inputs.shape[:-1], dim // 2, 2)
        real, imag = paired[..., 0], paired[..., 1]
        output = torch.stack([real * cosine - imag * sine, real * sine + imag * cosine], dim=-1)
        return output.reshape_as(inputs).to(input_dtype)

    return rotate(q), rotate(k)


class _CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, context: Optional[int], max_period: float):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context = context
        self.max_period = max_period
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        batch, time, _ = inputs.shape
        head_dim = self.embed_dim // self.num_heads
        qkv = self.in_proj(inputs).reshape(batch, time, 3, self.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = _apply_rope(q, k, self.max_period)

        positions = torch.arange(time, device=inputs.device)
        delta = positions.view(1, time, 1) - positions.view(1, 1, time)
        allowed = delta >= 0
        if self.context is not None:
            allowed = allowed & (delta < self.context)
        valid_keys = positions.view(1, 1, time) < input_lengths.view(-1, 1, 1)
        attention_mask = (allowed & valid_keys)[:, None, :, :]

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        valid_queries = (positions.view(1, time) < input_lengths.view(-1, 1))[:, None, :, None]
        output = torch.where(valid_queries, output, torch.zeros((), device=output.device, dtype=output.dtype))
        output = output.transpose(1, 2).reshape(batch, time, self.embed_dim)
        return self.out_proj(output)


class _TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: Optional[int],
        max_period: float,
        layer_scale: Optional[float],
    ):
        super().__init__()
        self.self_attn = _CausalSelfAttention(d_model, num_heads, context=context, max_period=max_period)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=False),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model, bias=False),
        )
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = _LayerScale(d_model, layer_scale)
            self.layer_scale_2 = _LayerScale(d_model, layer_scale)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        output = inputs + self.layer_scale_1(self.self_attn(self.norm1(inputs), input_lengths))
        return output + self.layer_scale_2(self.ffn(self.norm2(output)))


class _ProjectedTransformer(nn.Module):
    """MOSS transformer block with channel-first input/output projections."""

    downsample_ratio = 1

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        causal: bool = True,
        context: Optional[int] = None,
        positional_embedding: str = "rope",
        max_period: float = 10000.0,
        layer_scale: Optional[float] = None,
        gating: str = "none",
        norm: str = "layer_norm",
        **_kwargs,
    ):
        super().__init__()
        if not causal:
            raise ValueError("The integrated MOSS Nano path currently supports causal transformers only")
        if positional_embedding != "rope" or gating != "none" or norm != "layer_norm":
            raise ValueError("The integrated path supports the rope/none/layer_norm Nano configuration")
        self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    max_period=max_period,
                    layer_scale=layer_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, output_dimension, bias=False)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.input_proj(inputs.transpose(1, 2))
        for layer in self.layers:
            output = layer(output, input_lengths)
        return self.output_proj(output).transpose(1, 2), input_lengths


class _PatchedPretransform(nn.Module):
    def __init__(self, patch_size: int, is_downsample: bool):
        super().__init__()
        self.patch_size = patch_size
        self.is_downsample = is_downsample
        self.downsample_ratio = patch_size

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self.patch_size
        if self.is_downsample:
            batch, channels, _ = inputs.shape
            output = inputs.reshape(batch, channels, -1, patch_size)
            output = output.permute(0, 1, 3, 2).reshape(batch, channels * patch_size, -1)
            return output, torch.div(input_lengths, patch_size, rounding_mode="floor")

        batch, patched_channels, time = inputs.shape
        channels = patched_channels // patch_size
        output = inputs.reshape(batch, channels, patch_size, time)
        output = output.permute(0, 1, 3, 2).reshape(batch, channels, time * patch_size)
        return output, input_lengths * patch_size


def _build_stack(
    module_kwargs: List[dict],
    initial_frame_rate: float,
    default_context_duration: float,
    is_encoder: bool,
) -> Tuple[nn.ModuleList, float, int]:
    modules = nn.ModuleList()
    frame_rate = initial_frame_rate
    total_patch = 1
    for raw_kwargs in module_kwargs:
        kwargs = _to_plain_dict(raw_kwargs)
        module_type = kwargs.pop("module_type")
        if module_type == "PatchedPretransform":
            patch_size = int(kwargs["patch_size"])
            modules.append(_PatchedPretransform(patch_size=patch_size, is_downsample=is_encoder))
            total_patch *= patch_size
        elif module_type == "Transformer":
            context_duration = float(kwargs.pop("context_duration", default_context_duration))
            modules.append(_ProjectedTransformer(**kwargs, context=int(round(frame_rate * context_duration))))
        else:
            raise ValueError(f"Unsupported MOSS module_type: {module_type}")

        if is_encoder:
            frame_rate /= modules[-1].downsample_ratio
        else:
            frame_rate *= modules[-1].downsample_ratio
    return modules, frame_rate, total_patch


class MossAudioTokenizerEncoder(NeuralModule):
    """Nano causal-transformer encoder adapted to NeMo's mono codec API.

    When ``number_channels=2``, mono input is duplicated and channel-interleaved
    exactly as expected by the released 48 kHz stereo Nano topology.
    """

    def __init__(
        self,
        sampling_rate: int,
        downsample_rate: int,
        encoder_kwargs: List[dict],
        number_channels: int = 2,
        enable_channel_interleave: bool = True,
        causal_transformer_context_duration: float = 10.0,
    ):
        super().__init__()
        self.number_channels = number_channels
        self.enable_channel_interleave = enable_channel_interleave
        channel_interleave_factor = number_channels if enable_channel_interleave else 1
        internal_rate = sampling_rate * channel_interleave_factor
        self.encoder, _, actual_downsample = _build_stack(
            encoder_kwargs,
            initial_frame_rate=float(internal_rate),
            default_context_duration=causal_transformer_context_duration,
            is_encoder=True,
        )
        expected_internal_downsample = downsample_rate * channel_interleave_factor
        if actual_downsample != expected_internal_downsample:
            raise ValueError(
                f"Encoder patch product ({actual_downsample}) does not equal the external downsample rate "
                f"times the channel-interleave factor ({expected_internal_downsample})"
            )
        self.downsample_rate = downsample_rate
        self.internal_downsample_rate = expected_internal_downsample

    @property
    def input_types(self):
        return {
            "audio": NeuralType(("B", "T_audio"), AudioSignal()),
            "audio_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoded": NeuralType(("B", "D", "T_encoded"), EncodedRepresentation()),
            "encoded_len": NeuralType(tuple("B"), LengthsType()),
        }

    @typecheck()
    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform = audio.unsqueeze(1)
        internal_len = audio_len
        if self.number_channels > 1:
            waveform = waveform.repeat(1, self.number_channels, 1)
            if self.enable_channel_interleave:
                waveform = waveform.transpose(1, 2).contiguous().reshape(audio.shape[0], 1, -1)
                internal_len = internal_len * self.number_channels

        remainder = waveform.shape[-1] % self.internal_downsample_rate
        if remainder:
            waveform = F.pad(waveform, (0, self.internal_downsample_rate - remainder))
        output, output_len = waveform, internal_len
        for module in self.encoder:
            output, output_len = module(output, output_len)
        return output, output_len


class MossAudioTokenizerDecoder(NeuralModule):
    """Nano causal-transformer decoder with stereo-to-mono output adaptation."""

    def __init__(
        self,
        sampling_rate: int,
        downsample_rate: int,
        decoder_kwargs: List[dict],
        number_channels: int = 2,
        enable_channel_interleave: bool = True,
        causal_transformer_context_duration: float = 10.0,
        output_channel_mode: str = "mean",
    ):
        super().__init__()
        if output_channel_mode not in {"mean", "first"}:
            raise ValueError(f"Unsupported output_channel_mode: {output_channel_mode}")
        self.number_channels = number_channels
        self.enable_channel_interleave = enable_channel_interleave
        self.output_channel_mode = output_channel_mode
        channel_interleave_factor = number_channels if enable_channel_interleave else 1
        internal_rate = sampling_rate * channel_interleave_factor
        expected_internal_upsample = downsample_rate * channel_interleave_factor
        initial_frame_rate = internal_rate / expected_internal_upsample
        self.decoder, final_frame_rate, actual_upsample = _build_stack(
            decoder_kwargs,
            initial_frame_rate=initial_frame_rate,
            default_context_duration=causal_transformer_context_duration,
            is_encoder=False,
        )
        if actual_upsample != expected_internal_upsample or round(final_frame_rate) != internal_rate:
            raise ValueError("Decoder patch stack does not invert the configured encoder downsampling")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "D", "T_encoded"), VoidType()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(("B", "T_audio"), AudioSignal()),
            "audio_len": NeuralType(tuple("B"), LengthsType()),
        }

    @typecheck()
    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, output_len = inputs, input_len
        for module in self.decoder:
            output, output_len = module(output, output_len)

        if self.number_channels > 1 and self.enable_channel_interleave:
            output = output.squeeze(1).reshape(output.shape[0], -1, self.number_channels).transpose(1, 2)
            output_len = torch.div(output_len, self.number_channels, rounding_mode="floor")
        if output.shape[1] > 1:
            output = output.mean(dim=1) if self.output_channel_mode == "mean" else output[:, 0]
        else:
            output = output.squeeze(1)
        return output.float(), output_len


def _weight_norm_conv1d(in_channels: int, out_channels: int) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))


class _LFQCodebook(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.in_proj = _weight_norm_conv1d(input_dim, codebook_dim) if input_dim != codebook_dim else nn.Identity()
        self.out_proj = _weight_norm_conv1d(codebook_dim, input_dim) if input_dim != codebook_dim else nn.Identity()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def nearest(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.in_proj(inputs.float()).float()
        flat_encoded = F.normalize(encoded.transpose(1, 2).reshape(-1, encoded.shape[1]), dim=-1)
        normalized_codebook = F.normalize(self.codebook.weight.float(), dim=-1)
        distance = (
            flat_encoded.pow(2).sum(1, keepdim=True)
            - 2 * flat_encoded @ normalized_codebook.t()
            + normalized_codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = distance.argmin(dim=1).reshape(inputs.shape[0], -1)
        quantized = F.embedding(indices, self.codebook.weight).transpose(1, 2).float()
        return encoded, quantized, indices

    def decode_code(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = F.embedding(indices, self.codebook.weight).transpose(1, 2).float()
        return self.out_proj(quantized).float()


class MossAudioTokenizerResidualLFQ(VectorQuantizerBase):
    """Differentiable residual LFQ matching the Nano checkpoint topology.

    The returned ``commit_loss`` is the already weighted sum of codebook and
    encoder commitment losses. Set ``model.commit_loss_scale`` to 1.0 unless
    deliberately applying an additional global multiplier.
    """

    def __init__(
        self,
        input_dim: int = 768,
        rvq_dim: int = 512,
        output_dim: int = 768,
        num_quantizers: int = 16,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 1.0,
        codebook_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.25,
    ):
        super().__init__()
        self.rvq_dim = rvq_dim
        self._num_codebooks = num_quantizers
        self._codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout
        self.codebook_loss_weight = codebook_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.input_proj = _weight_norm_conv1d(input_dim, rvq_dim) if input_dim != rvq_dim else nn.Identity()
        self.output_proj = _weight_norm_conv1d(rvq_dim, output_dim) if rvq_dim != output_dim else nn.Identity()
        self.quantizers = nn.ModuleList(
            [_LFQCodebook(rvq_dim, codebook_size, codebook_dim) for _ in range(num_quantizers)]
        )

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(("B", "D", "T"), EncodedRepresentation()),
            "indices": NeuralType(("D", "B", "T"), TokenIndex()),
            "commit_loss": NeuralType((), LossType()),
        }

    def _active_quantizers(self) -> int:
        if self.training and self.quantizer_dropout > 0:
            if (
                torch.rand((), device=self.input_proj.weight.device if hasattr(self.input_proj, "weight") else None)
                < self.quantizer_dropout
            ):
                return int(torch.randint(1, self.num_codebooks + 1, ()).item())
        return self.num_codebooks

    @staticmethod
    def _masked_mse(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squared_error = (inputs - targets).pow(2) * mask
        denominator = mask.sum().clamp_min(1) * inputs.shape[1]
        return squared_error.sum() / denominator

    @typecheck()
    def forward(
        self, inputs: torch.Tensor, input_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            projected = self.input_proj(inputs.float()).float()
            positions = torch.arange(projected.shape[-1], device=projected.device)
            mask = (positions.view(1, 1, -1) < input_len.view(-1, 1, 1)).to(projected.dtype)
            residual = projected
            quantized_sum = torch.zeros_like(projected)
            indices = []
            codebook_losses = []
            commitment_losses = []

            active_quantizers = self._active_quantizers()
            for quantizer in self.quantizers[:active_quantizers]:
                encoded, quantized, this_indices = quantizer.nearest(residual * mask)
                codebook_losses.append(self._masked_mse(quantized, encoded.detach(), mask))
                commitment_losses.append(self._masked_mse(encoded, quantized.detach(), mask))
                straight_through = encoded + (quantized - encoded).detach()
                quantized_projected = quantizer.out_proj(straight_through).float() * mask
                quantized_sum = quantized_sum + quantized_projected
                residual = residual - quantized_projected
                indices.append(this_indices)

            all_indices = torch.stack(indices)
            dequantized = self.output_proj(quantized_sum).float() * mask
            codebook_loss = torch.stack(codebook_losses).mean()
            commitment_loss = torch.stack(commitment_losses).mean()
            loss = self.codebook_loss_weight * codebook_loss + self.commitment_loss_weight * commitment_loss
        return dequantized, all_indices, loss

    @typecheck(
        input_types={
            "inputs": NeuralType(("B", "D", "T"), EncodedRepresentation()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        },
        output_types={"indices": NeuralType(("D", "B", "T"), TokenIndex())},
    )
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        try:
            self.eval()
            _, indices, _ = self(inputs=inputs, input_len=input_len)
        finally:
            self.train(was_training)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(("D", "B", "T"), TokenIndex()),
            "input_len": NeuralType(tuple("B"), LengthsType()),
        },
        output_types={"dequantized": NeuralType(("B", "D", "T"), EncodedRepresentation())},
    )
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        if indices.shape[0] > self.num_codebooks:
            raise ValueError(f"Got {indices.shape[0]} codebooks, but this quantizer has {self.num_codebooks}")
        batch, time = indices.shape[1], indices.shape[2]
        quantized = torch.zeros(batch, self.rvq_dim, time, device=indices.device, dtype=torch.float32)
        for codebook_index, quantizer in enumerate(self.quantizers[: indices.shape[0]]):
            quantized = quantized + quantizer.decode_code(indices[codebook_index])
        output = self.output_proj(quantized).float()
        positions = torch.arange(time, device=indices.device)
        mask = (positions.view(1, 1, -1) < input_len.view(-1, 1, 1)).to(output.dtype)
        return output * mask
