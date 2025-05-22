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
from itertools import groupby

import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio, collate_vectors

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import TextTurn
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts import Llama2PromptFormatter, Llama3PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id


class SALMDataset(torch.utils.data.Dataset):
    """
    A dataset for Speech-Augmented Language Models (SALM) that processes multimodal conversations
    containing both text and audio turns.

    This dataset handles NeMoMultimodalConversation objects which combine text messages
    and audio segments in a conversational format. It uses audio_locator_tag in the text,
    where each such placeholder corresponds to an entire audio segment.

    Args:
        tokenizer (AutoTokenizer):
            Tokenizer for converting text to token IDs and vice versa. Must have a special
            audio_locator_tag token that will be replaced with audio embeddings during model's
            training step.

    Returns:
        A dictionary with the following keys:
            - audios: Tensor of audio waveform samples [B_audio, T_samples]
            - audio_lens: Tensor of audio lengths [B_audio]
            - input_ids: Tensor of text token IDs [B, T_tokens], including audio_locator_tag tokens
            - loss_mask: Boolean tensor [B, T_tokens] indicating which tokens are part of the
                assistant's responses (True) and should be used for computing loss

    Notes:
        - Each audio_locator_tag token in input_ids corresponds to an audio segment in audios
        - The SALM model later replaces these audio_locator_tag tokens with encoded audio embeddings
        - The loss_mask identifies which tokens are part of the target sequences (assistant responses)
          and which are part of the source sequences (user prompts)
        - The input_ids and loss_mask will be expanded during model forward pass to account for
          the variable-length audio segments that replace each audio_locator_tag token
    """

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_id = get_pad_id(tokenizer)

    def __getitem__(self, conversations: CutSet) -> dict:
        all_cuts = []
        example_idx_to_audio_idxs = []
        cntr = 0
        for conversation in conversations:
            assert isinstance(conversation, NeMoMultimodalConversation)
            example_idx_to_audio_idxs.append([])
            for cut in conversation.list_cuts():
                all_cuts.append(cut)
                example_idx_to_audio_idxs[-1].append(cntr)
                cntr += 1
        audios, audio_lens = collate_audio(CutSet(all_cuts))
        return {
            "audios": audios,
            "audio_lens": audio_lens,
            "input_ids": collate_vectors([c.input_ids for c in conversations], padding_value=self.pad_id),
            "loss_mask": collate_vectors([c.mask for c in conversations], padding_value=0).to(torch.bool),
        }


@registered_prompt_format_fn(NeMoMultimodalConversation, Llama3PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation, prompt: Llama3PromptFormatter
):
    # Collapse consecutive same-role turns into single turn for proper prompt formatting.
    turns = groupby(
        [
            {
                "role": turn.role,
                "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
            }
            for turn in example.turns
        ],
        key=lambda turn: turn["role"],
    )
    turns = [
        {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
        for role, turn_grp in turns
    ]
    if hasattr(example, "system_prompt"):
        turns = [{"role": "system", "slots": {"message": example.system_prompt}}] + turns
    return prompt.encode_dialog(turns)


@registered_prompt_format_fn(NeMoMultimodalConversation, Llama2PromptFormatter)
def default_multimodal_conversation_prompt_format_fn(
    example: NeMoMultimodalConversation, prompt: Llama2PromptFormatter
):
    # Collapse consecutive same-role turns into single turn for proper prompt formatting.
    turns = groupby(
        [
            {
                "role": turn.role,
                "slots": {"message": turn.value if isinstance(turn, TextTurn) else turn.audio_locator_tag},
            }
            for turn in example.turns
        ],
        key=lambda turn: turn["role"],
    )
    turns = [
        {"role": role, "slots": {"message": " ".join(t["slots"]["message"] for t in turn_grp)}}
        for role, turn_grp in turns
    ]
    if hasattr(example, "system_prompt"):
        turns[0]["role"] = "system_and_user"
        turns[0]["slots"]["system"] = example.system_prompt
    return prompt.encode_dialog(turns)
