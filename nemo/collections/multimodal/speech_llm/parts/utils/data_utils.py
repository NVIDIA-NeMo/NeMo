# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

import numpy as np
import torch


def maybe_cast_to_list(x):
    if isinstance(x, np.ndarray):
        return [item.tolist() for item in x]
    return x


def ceil_to_nearest(n, m):
    return (n + m - 1) // m * m


def get_num_samples_from_files(file_list):
    if isinstance(file_list, str):
        file_list = file_list.split(',')
    num_samples = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = list(f.readlines())
            num = len(lines)
            if lines[-1] == '\n':
                num -= 1
            num_samples.append(num)
    return num_samples


def shift_tokens_by_multi_audios(
    context_tokens, context_lengths, audio_feat_lens, context_start_idx, encoder_max_length
):
    """
    split and shift the context tokens by the audio segments, then concatenate them back. This function assumes that the whole context
    starts and ends with text tokens, and the audio segments are in between the text tokens. The audio segments are not allowed to be adjacent to each other.
    Args:
        context_tokens: tensor of shape [batch, max_context_len]
        context_lengths: tensor of shape [batch,]
        audio_feat_lens: List[List[int]]
        context_start_idx: List[List[int]]
        encoder_max_length: int
    """
    new_context_tokens = []
    for i in range(context_tokens.shape[0]):
        start_idx_list_i = context_start_idx[i] + [context_lengths[i]]
        input_len_list = [start_idx_list_i[j + 1] - start_idx_list_i[j] for j in range(len(start_idx_list_i) - 1)]
        context_tokens_list = context_tokens[i][: context_lengths[i]].split(input_len_list)
        context_tokens_i = [context_tokens_list[0]]
        for j in range(1, len(context_tokens_list)):
            context_tokens_i.append(
                torch.zeros(audio_feat_lens[i][j - 1], dtype=torch.long, device=context_tokens.device)
            )
            context_tokens_i.append(context_tokens_list[j])
        context_tokens_i = torch.cat(context_tokens_i)
        context_tokens_i = torch.nn.functional.pad(
            context_tokens_i, (0, encoder_max_length - context_tokens_i.shape[0])
        )
        new_context_tokens.append(context_tokens_i)
    new_context_tokens = torch.stack(new_context_tokens)
    return new_context_tokens


def get_nested_dict_value(d, key, sep="."):
    """
    Get the value of a nested dict given a key
    Args:
        d: dict
        key: str
    """
    for k in key.split(sep):
        d = d[k]
    return d


def align_feat_seq_list(
    seq_list: List[torch.Tensor],
    seq_len_list: List[torch.Tensor],
    mode: str = "min",
    pooling: str = 'mean',
    target_len: Optional[int] = None,
):
    """
    Align a list of feature sequences to the same length by repeating or discarding frames.
    Args:
        seq_list: List[torch.Tensor], list of tensors of shape [batch, hidden_size, seq_len]
        seq_len_list: List[torch.Tensor], list of tensors of shape [batch,]
        mode: str, "min" or "max"
        pooling: str, "mean", "max", or "min"
    Returns:
        new_seq_list: List[torch.Tensor], list of tensors of shape [batch, hidden_size, new_seq_len]
        new_seq_len_list: List[torch.Tensor], list of tensors of shape [batch,]
    """
    MODES = ["min", "max"]
    if mode not in MODES:
        raise ValueError(f"mode {mode} not supported, available modes: {MODES}")
    POOLING = ["mean", "max", "min", "avg"]
    if pooling not in POOLING:
        raise ValueError(f"pooling {pooling} not supported, available modes: {POOLING}")

    new_seq_len_list = []
    new_seq_list = []

    if target_len is None:
        target_len = [x.size(-1) for x in seq_list]
        target_len = min(target_len) if mode == "min" else max(target_len)

    for seq, seq_len in zip(seq_list, seq_len_list):
        curr_len = seq.size(-1)
        if curr_len > target_len:
            ratio = round(curr_len / target_len)
            res = abs(ratio * target_len - curr_len)
            if ratio * target_len > curr_len:  # e.g., ratio = 1.9
                # repeat the last res frames
                seq = torch.cat([seq, seq[:, :, -res:]], dim=-1)
                seq_len += res * (seq_len > target_len).long()
            elif ratio * target_len < curr_len:  # e.g., ratio = 2.1
                # discard the last res frames
                seq = seq[:, :, :-res]
                seq_len -= res * (seq_len > target_len).long()
            new_seq = seq.reshape(seq.size(0), seq.size(1), ratio, target_len)
            if pooling == "min":
                new_seq = new_seq.min(dim=2)
            elif pooling == "max":
                new_seq = new_seq.max(dim=2)
            else:
                new_seq = new_seq.mean(dim=2)
            new_seq_len = torch.round(seq_len / ratio).long()
        else:  # curr_len <= target_len
            ratio = round(target_len / curr_len)
            res = abs(ratio * curr_len - target_len)
            new_seq = torch.repeat_interleave(seq, ratio, dim=-1)
            new_seq_len = seq_len * ratio
            if ratio * curr_len > target_len:  # e.g., ratio = 1.9
                new_seq = new_seq[:, :, :target_len]
                new_seq_len = (
                    seq_len * ratio - (ratio * seq_len - target_len) * (ratio * seq_len > target_len).long()
                )  # subtract additional frames
            elif ratio * curr_len < target_len:  # e.g., ratio = 2.1
                new_seq = torch.cat([new_seq, seq[:, :, -res:]], dim=-1)
        new_seq_list.append(new_seq)
        new_seq_len_list.append(new_seq_len)
    return new_seq_list, new_seq_len_list
