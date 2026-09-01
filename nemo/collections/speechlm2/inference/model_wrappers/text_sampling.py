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

"""Shared text-token sampling for native and vLLM-Omni inference."""

import torch

from nemo.utils import logging


def sample_text_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    current_step: int,
    *,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    special_token_ids: set[int],
    special_ids_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one token per row with the VoiceChat policy."""
    batch_size, _ = logits.shape
    device = logits.device
    greedy_tokens = logits.argmax(dim=-1)

    if top_p >= 1.0 and repetition_penalty == 1.0 and temperature in (0.0, 1.0):
        return greedy_tokens
    if temperature == 0.0:
        return greedy_tokens

    sampled_tokens = greedy_tokens.clone()
    if special_ids_tensor is not None and special_ids_tensor.device != device:
        special_ids_tensor = special_ids_tensor.to(device)

    for batch_idx in range(batch_size):
        if greedy_tokens[batch_idx].item() in special_token_ids:
            continue

        batch_logits = logits[batch_idx].clone()
        if repetition_penalty != 1.0 and current_step > 0:
            unique_prev = generated_tokens[batch_idx, :current_step].unique()
            if special_ids_tensor is not None:
                ids_t = special_ids_tensor
                if ids_t.device != unique_prev.device:
                    ids_t = ids_t.to(unique_prev.device)
                unique_prev = unique_prev[~torch.isin(unique_prev, ids_t)]

            if unique_prev.numel() > 0:
                if unique_prev.device != batch_logits.device:
                    unique_prev = unique_prev.to(batch_logits.device)
                prev_logits = batch_logits[unique_prev]
                batch_logits[unique_prev] = torch.where(
                    prev_logits > 0,
                    prev_logits / repetition_penalty,
                    prev_logits * repetition_penalty,
                )

        if temperature != 1.0:
            batch_logits = batch_logits / temperature

        if not torch.isfinite(batch_logits).all():
            logging.warning(
                f"sample_text_token: logits contain NaN or inf at step {current_step}, "
                f"batch {batch_idx}: nan={batch_logits.isnan().sum().item()}, "
                f"inf={batch_logits.isinf().sum().item()}, "
                f"min={batch_logits[~batch_logits.isnan()].min().item() if not batch_logits.isnan().all() else 'all_nan'}, "
                f"max={batch_logits[~batch_logits.isnan()].max().item() if not batch_logits.isnan().all() else 'all_nan'}"
            )
            sampled_tokens[batch_idx] = greedy_tokens[batch_idx]
            continue

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(batch_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            batch_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

        probs = torch.softmax(batch_logits, dim=-1)
        sampled_tokens[batch_idx] = torch.multinomial(probs, num_samples=1).item()

    return sampled_tokens
