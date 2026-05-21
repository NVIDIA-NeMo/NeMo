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

import torch

@torch.no_grad()
def patched_infer_codes_one_step(
    self,
    current_subword_id,
    prev_subword_id,
    current_subword_mask,
    prev_audio_tokens,
    past_key_values,
    guidance_enabled=True,
    generation_config=None,
    ignore_eos_flag_stop=True,
    request_id=None, # change signature to include request_id
):
    if self.cfg.tts_config.context_hidden_size is not None:
        # get context_hidden_state it is always one step behind current_subword_id
        # for the first step uses the last step from warmup
        context_hidden_state = self.embed_tokens(prev_subword_id)
    else:
        context_hidden_state = None

    # force silence as next token
    if self.cfg.get('inference_force_speech_silence_on_eos', True):
        silence_codes = self.codec_silence_tokens.view(1, 1, -1).expand(prev_audio_tokens.shape)
        prev_audio_tokens = torch.where(
            current_subword_id.unsqueeze(-1) == self.text_eos_id,
            silence_codes,  # silence
            prev_audio_tokens,  # keep original
        )
    # get subword_ids
    inputs = {
        "code": prev_audio_tokens,
        "context_hidden_state": context_hidden_state,
        "subword_ids": current_subword_id,
        "subword_mask": current_subword_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
        "guidance_enabled": guidance_enabled,
        "generation_config": generation_config,
        "ignore_eos_flag_stop": ignore_eos_flag_stop,
        "request_id": request_id,  # pass request_id to the model
    }
    outputs = self.tts_model(**inputs)
    return outputs["codes"], outputs["past_key_values"]