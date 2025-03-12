# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import List

import torch


def build_position_ids(token_ids: torch.Tensor) -> torch.Tensor:
    """Create position ids"""
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids).clone()

    return position_ids


def build_loss_mask(input_ids: List[int], answer_start_idx: int, answer_only_loss: bool = True) -> List[float]:
    """Pad input_ids in batch to max batch length while building loss mask"""
    # function borrowed from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
    if answer_only_loss:
        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
    else:
        loss_mask = [1.0] * len(input_ids)

    return loss_mask


def ceil_to_nearest(n: int, m: int) -> int:
    """Ceil n to the nearest multiple of m"""
    return (n + m - 1) // m * m


def pad_or_trim_to_max_length(
    inputs: torch.Tensor, max_length: int, pad_value: int, ceil_to: int = 1, seq_dim: int = 1
) -> torch.Tensor:
    """
    Pad or trim a tensor to max_length
    Args:
        inputs: tensor to pad or trim, shape=[batch, seq, hid_dim] or [batch, seq]
        max_length: length to pad or trim to
        pad_value: value to pad with
        ceil_to: pad to the nearest multiple of this number
    """
    if seq_dim != 1:
        # transpose to [B,T,D]
        inputs = inputs.transpose(seq_dim, 1)
    if ceil_to > 1:
        # ceil max_length to the nearest multiple of ceil_to, used in context parallelism
        max_length = ceil_to_nearest(max_length, ceil_to)

    if inputs.size(1) < max_length:
        # pad
        pad_size = max_length - inputs.size(1)
        if inputs.dim() == 3:
            pad = torch.full(
                (inputs.size(0), pad_size, inputs.size(2)), pad_value, dtype=inputs.dtype, device=inputs.device
            )
        elif inputs.dim() == 2:
            pad = torch.full((inputs.size(0), pad_size), pad_value, dtype=inputs.dtype, device=inputs.device)
        else:
            raise ValueError(f"Unsupported input dim: {inputs.dim()}, must be [B,T,D] or [B,T]")
        inputs = torch.cat([inputs, pad], dim=1)
    elif inputs.size(1) > max_length:
        # trim
        inputs = inputs[:, :max_length]

    if seq_dim != 1:
        # transpose back
        inputs = inputs.transpose(seq_dim, 1)
    return inputs


def estimate_encoded_max_length(audio_signal: torch.Tensor, sample_rate: int, frame_length: float) -> int:
    """
    Estimate the length of the audio signal after encoding
    Args:
        audio_signal: audio signal tensor, shape=[batch, time]
        sample_rate: sample rate of the audio signal, e.g. 16000
        frame_length: frame length in seconds, e.g. 0.08 for FC
    """
    return int(math.ceil(audio_signal.size(1) / sample_rate / frame_length))
