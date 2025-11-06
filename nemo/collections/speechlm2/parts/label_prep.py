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

"""Utility functions for preparing model inputs including text, audio, and ASR channels."""

import torch
from nemo.utils import logging


def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
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

    # Filter: new position must be in bounds and not overwrite EOS or PAD
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
    target_codes,
    target_tokens,
    source_encoded,
    asr_emb,
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
):
    """
    Prepare text, audio, and ASR labels from batch data.
    
    This function handles:
    - Text channel delay/advance adjustments
    - User text prediction with delayed source tokens
    - User turn masking and agent turn boundary preservation
    - ASR head processing for conversational models
    - Tensor parallelism adjustments
    
    Args:
        batch: Dictionary containing batch data including source_tokens, target_tokens, etc.
        target_codes: Encoded target audio codes (B, T, K), or None if audio is not being predicted
        target_tokens: Target text tokens (B, T)
        source_encoded: Encoded source audio features (B, T, D)
        asr_emb: ASR embedding features (B, T, D)
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
        
    Returns:
        dict: Dictionary containing:
            - text_inputs: Text input tokens (B, T-1)
            - text_labels: Text label tokens (B, T-1)
            - audio_inputs: Audio input codes (B, T-1, K) or None if target_codes is None
            - audio_labels: Audio label codes (B, T-1, K) or None if target_codes is None
            - asr_inputs: ASR input tokens (B, T-1) if use_separate_asr_head is True
            - asr_labels: ASR label tokens (B, T-1) if use_separate_asr_head is True
    """
    
    # Apply text channel delay and advance adjustments
    # move back text channel by x, in inference it advance the text channel prediction
    # it is the oposite of speech delay applied on text channel
    if advance_text_channel_by:
        pad = torch.full(
            (target_tokens.shape[0], advance_text_channel_by),
            fill_value=text_pad_id,
            device=target_tokens.device,
            dtype=torch.long,
        )
        target_tokens = torch.cat([target_tokens[:, advance_text_channel_by :], pad], dim=-1)
        # make sure that eos/bos is in the place (it can cut tokens from the first advance_text_channel_by tokens and this will breaks everything)

    if cfg.get("delay_text_channel_by", 0) > 0:
        # if batch.get("formatter", None) and batch["formatter"][0] == 'nemo_tarred_to_duplex':
        #     delay_by = cfg.get("delay_source_text_by", 0)
        # else:
        #     delay_by = cfg.get("delay_text_channel_by", 0)
        delay_by = cfg.get("delay_text_channel_by", 0)

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
        target_tokens = torch.cat([pad, target_tokens[:, :-delay_by]], dim=-1)
        # batch["target_token_lens"] = batch["target_token_lens"] + delay_by

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
            if target_codes is not None:
                target_codes = target_codes[:, :min_len]
            source_encoded = source_encoded[:, :min_len]
            asr_emb = asr_emb[:, :min_len]

        # Optionally delay the prediction of source_tokens by a flag
        delay_source_text_by = cfg.get("delay_source_text_by", 0)
        if delay_source_text_by > 0:
            pad = torch.full(
                (source_tokens.shape[0], delay_source_text_by),
                fill_value=text_pad_id,
                device=source_tokens.device,
                dtype=torch.long,
            )
            # if cfg.get("debug", False):
                # import pdb; pdb.set_trace()
            # source_tokens_delayed = torch.cat([pad, source_tokens], dim=-1)
            # batch["source_token_lens"] = batch["source_token_lens"] + delay_source_text_by
            source_tokens_delayed = torch.cat([pad, source_tokens[:, :-delay_source_text_by]], dim=-1)
            # Add back user_eos_id since it may be truncated
            # source_tokens_delayed[:, -1] = user_eos_id
        else:
            source_tokens_delayed = source_tokens

        source_tokens_flat = source_tokens_delayed.clone()
        target_tokens_flat = target_tokens.clone()
        
        if cfg.get("allow_user_text_in_agent_turn", False):

            assert cfg.get("use_separate_asr_head", False), "allow_user_text_in_agent_turn is only supported when use_separate_asr_head is True"

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
                import pdb; pdb.set_trace()

            # To be consistent with the single channel case, replace the user_eos_id with agent_eos_id
            source_tokens_flat = source_tokens_flat.clone()
            source_tokens_flat[source_tokens_flat == user_eos_id] = text_eos_id
            asr_inputs = source_tokens_flat[:, :-1]
            asr_labels = source_tokens_flat[:, 1:]

            if target_codes is not None:
                input_ids = torch.cat([target_codes, target_tokens_flat[..., None]], dim=-1)
                text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
                text_labels = input_ids[:, 1:, -1]  # (B, T-1)
                audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
                audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)
            else:
                text_inputs = target_tokens_flat[:, :-1]
                text_labels = target_tokens_flat[:, 1:]
                audio_inputs = None
                audio_labels = None

            print(f"asr_inputs.shape: {asr_inputs.shape}")
            print(f"text_inputs.shape: {text_inputs.shape}")
            if audio_inputs is not None:
                print(f"audio_inputs.shape: {audio_inputs.shape}")
            else:
                print("audio_inputs: None")
            if asr_inputs.shape[1] != text_inputs.shape[1]:
                import pdb; pdb.set_trace()
            if audio_inputs is not None and text_inputs.shape[1] != audio_inputs.shape[1]:
                import pdb; pdb.set_trace()

            result = {
                "asr_inputs": asr_inputs,
                "asr_labels": asr_labels,
                "source_token_lens": batch["source_token_lens"],
                "text_inputs": text_inputs,
                "text_labels": text_labels,
                "target_token_lens": batch["target_token_lens"],
                "audio_inputs": audio_inputs,
                "audio_labels": audio_labels,
                "source_encoded": source_encoded,
            }
            return result

        # Merge user and agent text into a single channel
        mask = torch.zeros_like(source_tokens_flat, dtype=torch.bool)
        for i in range(source_tokens_flat.size(0)):
            src = source_tokens_flat[i]
            user_bos_indices = (src == user_bos_id).nonzero(as_tuple=True)[0]
            user_eos_indices = (src == user_eos_id).nonzero(as_tuple=True)[0]
            for user_bos_idx in user_bos_indices:
                user_eos_after = user_eos_indices[user_eos_indices > user_bos_idx]
                if len(user_eos_after) == 0:
                    continue
                user_eos_idx = user_eos_after[0]
                
                # In the case of agent_bos appear during user turn, take the agent_bos
                # uuuuu
                #    aaaaa --> uuuaaa
                bos_in_target = (target_tokens_flat[i, user_bos_idx:user_eos_idx] == text_bos_id).nonzero(as_tuple=True)
                if bos_in_target[0].numel() > 0:
                    bos_in_target_idx = bos_in_target[0][0].item() + user_bos_idx
                    user_eos_idx = min(user_eos_idx, bos_in_target_idx)
                
                # Check if there's a text_eos_id (agent turn end) between bos_idx and eos_idx
                # If so, move it to just before bos_idx to preserve agent turn boundary
                # aaaaa
                #    uuuuu --> aaauuuuu
                text_eos_in_range = (target_tokens_flat[i, user_bos_idx:user_eos_idx] == text_eos_id).nonzero(as_tuple=True)
                if text_eos_in_range[0].numel() > 0:
                    text_eos_idx = text_eos_in_range[0][0].item() + user_bos_idx
                    # Move the text_eos_id before bos_idx
                    if cfg.get("force_move_text_eos_to_user_speech_start", False):
                        new_eos_idx = min(user_bos_idx - 1, text_eos_idx - cfg.get("delay_text_channel_by", 1))
                    else:
                        new_eos_idx = user_bos_idx - 1
                    target_tokens_flat[i, new_eos_idx] = text_eos_id
                    # Clear the original text_eos_id position
                    target_tokens_flat[i, text_eos_idx] = text_pad_id
                
                # Mark mask from bos_idx to eos_idx-1 (inclusive of bos, exclusive of eos)
                mask[i, user_bos_idx:user_eos_idx] = True

        target_tokens = torch.where(mask, source_tokens_flat, target_tokens_flat)
        logging.info(f"target_tokens[0] w/ delay of {delay_source_text_by}: {target_tokens[0]}")
    else:
        target_tokens_flat = target_tokens

    if cfg.get("debug", False):
        import pdb; pdb.set_trace()
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
        import pdb; pdb.set_trace()

    if target_codes is not None:
        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if use_tp:
            tp_world_size = device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]
                asr_emb = asr_emb[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
    else:
        if use_tp:
            tp_world_size = device_mesh["tensor_parallel"].size()
            if (remainder := (target_tokens.shape[1] - 1) % tp_world_size) != 0:
                target_tokens = target_tokens[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]
                asr_emb = asr_emb[:, :-remainder]
        
        text_inputs = target_tokens[:, :-1]
        text_labels = target_tokens[:, 1:]
    
    result = {
        "text_inputs": text_inputs,
        "text_labels": text_labels,
    }
    
    # Split the merged text channel into asr and text channels (no overlap between them)
    if cfg.get("use_separate_asr_head", False):
        if target_codes is not None:
            asr_ids = input_ids.clone()[:, :, -1]
        else:
            asr_ids = target_tokens.clone()
        if cfg.get("is_conv", False):
            # Remove all asr ids between text_bos_id and text_eos_id and replace with text_pad_id.
            # Keep the text_eos_id and remove the text_bos_id.
            for i in range(asr_ids.shape[0]):
                bos_indices = (asr_ids[i] == text_bos_id).nonzero(as_tuple=True)[0]
                eos_indices = (asr_ids[i] == text_eos_id).nonzero(as_tuple=True)[0]
                for bos_idx in bos_indices:
                    eos_after = eos_indices[eos_indices > bos_idx]
                    if len(eos_after) == 0:
                        # This is the last turn
                        eos_after = torch.tensor([asr_ids.shape[1] - 1], device=asr_ids.device)
                    eos_idx = eos_after[0]
                    asr_ids[i, bos_idx + 1:eos_idx + 1] = text_pad_id
        asr_inputs = asr_ids[:, :-1]
        asr_labels = asr_ids[:, 1:]
        
        result["asr_inputs"] = asr_inputs
        result["asr_labels"] = asr_labels
        
        if not cfg.get("force_use_asr_head_for_user_agent_text", False):
            result["text_inputs"] = target_tokens_flat[:, :-1]
            result["text_labels"] = target_tokens_flat[:, 1:]
    
    if target_codes is not None:
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)
    else:
        audio_inputs = None
        audio_labels = None
    
    result["audio_inputs"] = audio_inputs
    result["audio_labels"] = audio_labels

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
        import pdb; pdb.set_trace()

    result["source_encoded"] = source_encoded

    return result

