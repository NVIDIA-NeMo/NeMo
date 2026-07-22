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

"""Utility functions for preparing model inputs including text and ASR channels."""

import torch
from nemo.utils import logging


def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS.
    Safe for GPU execution.
    """
    B, T = tokens.shape
    tokens = tokens.clone()
    device = tokens.device

    # Find all EOS positions
    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    # Flattened indices of EOS tokens
    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]  # [N]
    eos_pos = eos_indices[:, 1]  # [N]
    new_pos = eos_pos + shift  # [N]

    # Filter: new position must be in bounds and not overwrite EOS
    valid = (new_pos < T)
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        # Now, check overwrite safety in new positions
        target_vals = tokens[b_idx, new_pos]
        safe = (target_vals != eos_token_id)

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            # Move EOS token: clear original, set new
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


def prepare_labels(
    batch,
    target_tokens,
    source_encoded,
    cfg,
    predict_user_text,
    user_bos_id,
    user_eos_id,
    text_pad_id,
    text_bos_id,
    text_eos_id,
    advance_text_channel_by=None,
    use_tp=False,
    device_mesh=None,
    function_channel=None,
    function_channel_loss_mask=None,
    prompt_token_lens=None,
):
    """
    Prepare text and ASR labels from batch data.
    
    This function handles:
    - Text channel delay/advance adjustments
    - User text prediction with delayed source tokens
    - User turn masking and agent turn boundary preservation
    - ASR head processing for conversational models
    - Tensor parallelism adjustments
    - Function calling channel synchronization with temporal shifts
    
    Args:
        batch: Dictionary containing batch data including source_tokens, target_tokens, etc.
        target_tokens: Target text tokens (B, T)
        source_encoded: Encoded source audio features (B, T, D)
        cfg: Configuration object with model settings
        predict_user_text: Whether to predict user text in addition to agent text
        user_bos_id: Token ID for user turn beginning
        user_eos_id: Token ID for user turn ending
        text_pad_id: Token ID for text padding
        text_bos_id: Token ID for agent text beginning
        text_eos_id: Token ID for agent text ending
        advance_text_channel_by: Number of frames to advance text channel prediction
        use_tp: Whether tensor parallelism is enabled
        device_mesh: Device mesh for tensor parallelism
        function_channel: Function calling channel tokens (B, T) if function calling is enabled
        function_channel_loss_mask: Loss mask for function calling channel (B, T)
        
    Returns:
        dict: Dictionary containing:
            - text_inputs: Text input tokens (B, T-1)
            - text_labels: Text label tokens (B, T-1)
            - asr_inputs: ASR input tokens (B, T-1) if predict_user_text is True
            - asr_labels: ASR label tokens (B, T-1) if predict_user_text is True
            - function_inputs: Function calling input tokens (B, T-1) if function_channel provided
            - function_labels: Function calling label tokens (B, T-1) if function_channel provided
            - function_loss_mask: Function calling loss mask (B, T-1) if function_channel provided
    """
    
    # Apply text channel delay and advance adjustments
    # move back text channel by x, in inference it advance the text channel prediction
    # it is the oposite of speech delay applied on text channel
    if advance_text_channel_by:
        if cfg.get("debug_fc", False):
            logging.info(f"[prepare_labels] Applying advance_text_channel_by={advance_text_channel_by}")
            logging.info(f"[prepare_labels] Before advance - target_tokens shape: {target_tokens.shape}")
            if function_channel is not None:
                logging.info(f"[prepare_labels] Function channel will NOT be shifted (to preserve function call timing)")
        
        pad = torch.full(
            (target_tokens.shape[0], advance_text_channel_by),
            fill_value=text_pad_id,
            device=target_tokens.device,
            dtype=torch.long,
        )
        
        # Protect system prompt region from being shifted (optional)
        protect_prompt_from_shift = cfg.get("protect_prompt_from_shift", False)
        if prompt_token_lens is not None and protect_prompt_from_shift:
            if cfg.get("debug_fc", False):
                logging.info(f"[prepare_labels] Protecting system prompt region from advance shift")
            for b in range(target_tokens.shape[0]):
                prompt_len = prompt_token_lens[b].item()
                if prompt_len > 0:
                    if cfg.get("debug_fc", False) and b == 0:
                        logging.info(f"[prepare_labels] Batch {b}: Prompt length={prompt_len}, keeping positions [0:{prompt_len}] fixed, shifting [{prompt_len}:] by {advance_text_channel_by}")
                    # Keep prompt region [0:prompt_len], shift only content after it
                    prompt_region_tokens = target_tokens[b, :prompt_len]
                    shifted_tokens = target_tokens[b, prompt_len + advance_text_channel_by:]
                    padded_end = torch.full((advance_text_channel_by,), fill_value=text_pad_id, device=target_tokens.device, dtype=torch.long)
                    target_tokens[b] = torch.cat([prompt_region_tokens, shifted_tokens, padded_end], dim=-1)
                else:
                    # No prompt, apply full shift
                    target_tokens[b] = torch.cat([target_tokens[b, advance_text_channel_by:], pad[b]], dim=-1)
        else:
            # No prompt_token_lens, apply full shift as before
            target_tokens = torch.cat([target_tokens[:, advance_text_channel_by :], pad], dim=-1)
        # make sure that eos/bos is in the place (it can cut tokens from the first advance_text_channel_by tokens and this will breaks everything)
        
        # DO NOT shift function calling channel - function calls must stay at their true timeline positions
        # Agent text has PAD tokens at function positions anyway, so no conflict
        if function_channel is not None:
            if cfg.get("debug_fc", False):
                logging.info(f"[prepare_labels] After advance - target_tokens shape: {target_tokens.shape}")
                logging.info(f"[prepare_labels] ✓ Function channel NOT shifted - preserves function call timing at true positions")

    if cfg.get("delay_text_channel_by", 0) > 0:
        delay_by = cfg.get("delay_text_channel_by", 0)
        
        if cfg.get("debug_fc", False):
            logging.info(f"[prepare_labels] Applying delay_text_channel_by={delay_by}")
            logging.info(f"[prepare_labels] Before delay - target_tokens shape: {target_tokens.shape}")
            if function_channel is not None:
                logging.info(f"[prepare_labels] Function channel will NOT be shifted (to preserve function call timing)")

        eos_mask = (target_tokens == text_eos_id) & (torch.arange(target_tokens.size(1), device=target_tokens.device).unsqueeze(0) >= (target_tokens.size(1) - delay_by))
        for i in range(target_tokens.size(0)):
            if eos_mask[i].any():
                target_tokens[i, -(delay_by)] = text_eos_id
        target_tokens = torch.where(eos_mask, text_pad_id, target_tokens)
        pad = torch.full(
            (target_tokens.shape[0], delay_by),
            fill_value=text_pad_id,
            device=target_tokens.device,
            dtype=torch.long,
        )
        
        # Protect system prompt region from being shifted (optional)
        protect_prompt_from_shift = cfg.get("protect_prompt_from_shift", False)
        if prompt_token_lens is not None and protect_prompt_from_shift:
            if cfg.get("debug_fc", False):
                logging.info(f"[prepare_labels] Protecting system prompt region from delay shift")
            for b in range(target_tokens.shape[0]):
                prompt_len = prompt_token_lens[b].item()
                if prompt_len > 0:
                    if cfg.get("debug_fc", False) and b == 0:
                        logging.info(f"[prepare_labels] Batch {b}: Prompt length={prompt_len}, keeping positions [0:{prompt_len}] fixed, delaying [{prompt_len}:] by {delay_by}")
                    # Keep prompt region [0:prompt_len], delay only content after it
                    prompt_region_tokens = target_tokens[b, :prompt_len]
                    content_after_prompt = target_tokens[b, prompt_len:-delay_by]
                    delay_pad = torch.full((delay_by,), fill_value=text_pad_id, device=target_tokens.device, dtype=torch.long)
                    target_tokens[b] = torch.cat([prompt_region_tokens, delay_pad, content_after_prompt], dim=-1)
                else:
                    # No prompt, apply full shift
                    target_tokens[b] = torch.cat([pad[b], target_tokens[b, :-delay_by]], dim=-1)
        else:
            # No prompt_token_lens, apply full shift as before
            target_tokens = torch.cat([pad, target_tokens[:, :-delay_by]], dim=-1)
        # batch["target_token_lens"] = batch["target_token_lens"] + delay_by
        
        # DO NOT shift function calling channel - function calls must stay at their true timeline positions
        # Agent text has PAD tokens at function positions anyway, so no conflict
        if function_channel is not None:
            if cfg.get("debug_fc", False):
                logging.info(f"[prepare_labels] After delay - target_tokens shape: {target_tokens.shape}")
                logging.info(f"[prepare_labels] ✓ Function channel NOT shifted - preserves function call timing at true positions")

    original_target_tokens = target_tokens.clone()
    if cfg.get("delay_text_eos_by", None):
        target_tokens = delay_eos(target_tokens, text_eos_id, text_pad_id, shift=cfg.delay_text_eos_by)

    if cfg.get("delay_text_bos_by", None):
        target_tokens = delay_eos(target_tokens, text_bos_id, text_pad_id, shift=cfg.delay_text_bos_by)
    
    if predict_user_text:
        source_tokens = batch["source_tokens"]
        
        if source_tokens.shape != target_tokens.shape:
            min_len = min(source_tokens.shape[1], target_tokens.shape[1])
            source_tokens = source_tokens[:, :min_len]
            target_tokens = target_tokens[:, :min_len]
            source_encoded = source_encoded[:, :min_len]

        # Optionally delay the prediction of source_tokens by a flag
        delay_source_text_by = cfg.get("delay_source_text_by", 0)
        if delay_source_text_by > 0:
            pad = torch.full(
                (source_tokens.shape[0], delay_source_text_by),
                fill_value=text_pad_id,
                device=source_tokens.device,
                dtype=torch.long,
            )
            source_tokens_delayed = torch.cat([pad, source_tokens[:, :-delay_source_text_by]], dim=-1)
        else:
            source_tokens_delayed = source_tokens

        source_tokens_flat = source_tokens_delayed.clone()
        target_tokens_flat = target_tokens.clone()

        # To be consistent with the single channel case, replace the user_eos_id with agent_eos_id
        source_tokens_flat = source_tokens_flat.clone()
        source_tokens_flat[source_tokens_flat == user_eos_id] = text_eos_id
        asr_inputs = source_tokens_flat[:, :-1]
        asr_labels = source_tokens_flat[:, 1:]
        text_inputs = target_tokens_flat[:, :-1]
        text_labels = target_tokens_flat[:, 1:]

        # Keep user and agent text in separate channels and allow overlap between them
        if cfg.get("debug", False):
            i = 0
            target_tokens_flat_masked = target_tokens_flat[i] * (target_tokens_flat[i] != text_pad_id)
            print(f"target_tokens_flat[i]:", target_tokens_flat_masked)
            target_tokens_masked = target_tokens[i] * (target_tokens[i] != text_pad_id)
            print(f"target_tokens[i]:", target_tokens_masked)
            source_tokens_flat_masked = source_tokens_flat[i] * (source_tokens_flat[i] != text_pad_id)
            print(f"source_tokens_flat[i]:", source_tokens_flat_masked)
            stacked = torch.stack([source_tokens_flat_masked, target_tokens_flat_masked], dim=1)
            print("stacked[:500]:", stacked[:500])

        if asr_inputs.shape[1] != text_inputs.shape[1]:
            print(f"mismatch between asr_inputs.shape: {asr_inputs.shape} and text_inputs.shape: {text_inputs.shape}")

        result = {
            "asr_inputs": asr_inputs,
            "asr_labels": asr_labels,
            "source_token_lens": batch["source_token_lens"],
            "text_inputs": text_inputs,
            "text_labels": text_labels,
            "target_token_lens": batch["target_token_lens"],
            "source_encoded": source_encoded,
        }
        
        # Add function calling channel if present
        if function_channel is not None:
            result["function_inputs"] = function_channel[:, :-1]
            result["function_labels"] = function_channel[:, 1:]
            result["function_loss_mask"] = function_channel_loss_mask[:, 1:]
        
        return result
    else:
        target_tokens_flat = target_tokens

    if cfg.get("debug", False):
        i = 0
        target_tokens_flat_masked = target_tokens_flat[i] * (target_tokens_flat[i] != text_pad_id)
        print(f"target_tokens_flat[i]:", target_tokens_flat_masked)
        target_tokens_masked = target_tokens[i] * (target_tokens[i] != text_pad_id)
        print(f"target_tokens[i]:", target_tokens_masked)
        if predict_user_text:
            source_tokens_flat_masked = source_tokens_flat[i] * (source_tokens_flat[i] != text_pad_id)
            print(f"source_tokens_flat[i]:", source_tokens_flat_masked)
            stacked = torch.stack([source_tokens_flat_masked, target_tokens_flat_masked], dim=1)
            print("ori_stacked[:500]:", stacked[:500])

    if use_tp:
        tp_world_size = device_mesh["tensor_parallel"].size()
        if (remainder := (target_tokens.shape[1] - 1) % tp_world_size) != 0:
            target_tokens = target_tokens[:, :-remainder]
            source_encoded = source_encoded[:, :-remainder]
            # Also adjust function channel if present
            if function_channel is not None:
                function_channel = function_channel[:, :-remainder]
                function_channel_loss_mask = function_channel_loss_mask[:, :-remainder]
    
    text_inputs = target_tokens[:, :-1]
    text_labels = target_tokens[:, 1:]
    
    result = {
        "text_inputs": text_inputs,
        "text_labels": text_labels,
    }
    
    # Split the merged text channel into asr and text channels (no overlap between them)
    if cfg.get("predict_user_text", False):
        asr_ids = target_tokens.clone()
        asr_inputs = asr_ids[:, :-1]
        asr_labels = asr_ids[:, 1:]
        
        result["asr_inputs"] = asr_inputs
        result["asr_labels"] = asr_labels
    
    # Add function calling channel if present
    if function_channel is not None:
        result["function_inputs"] = function_channel[:, :-1]
        result["function_labels"] = function_channel[:, 1:]
        result["function_loss_mask"] = function_channel_loss_mask[:, 1:]

    if cfg.get("debug", False):
        ori_stacked = torch.stack(
            [
                batch['source_tokens'][0] * (batch['source_tokens'][0] != text_pad_id),
                batch['target_tokens'][0] * (batch['target_tokens'][0] != text_pad_id)
            ],
            dim=1
        )
        print("ori_stacked[:500]:", ori_stacked)
        i = 0
        asr_masked = result.get("asr_labels", result["text_labels"])[i][:1000] * (
            result.get("asr_labels", result["text_labels"])[i][:1000] != text_pad_id
        )
        text_masked = result["text_labels"][i][:1000] * (result["text_labels"][i][:1000] != text_pad_id)
        stacked = torch.stack([asr_masked, text_masked], dim=1)
        print("delayed stacked[:500]:", stacked[:500])

    result["source_encoded"] = source_encoded

    return result

