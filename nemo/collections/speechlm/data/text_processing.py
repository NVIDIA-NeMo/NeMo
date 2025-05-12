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

from dataclasses import dataclass
from typing import Optional

import torch

from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation
from nemo.collections.common.data.prompt_fn import get_prompt_format_fn
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.utils import logging

__all__ = ['MultimodalConversationTextProcessor', 'TextProcessorOutput']


@dataclass
class TextProcessorOutput:
    """
    A dataclass to store the output of the text processor.
    """

    input_ids: torch.Tensor
    answer_start_idx: torch.Tensor
    context_ids: torch.Tensor
    context_length: torch.Tensor
    answer_ids: torch.Tensor
    context_start_idx: torch.Tensor
    num_audios: torch.Tensor


class MultimodalConversationTextProcessor:
    """
    Text processor for multi-modal conversation with lhotse dataloader.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        prompt_format: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        add_boa_eoa: Optional[bool] = False,
        boa_string: Optional[str] = "<BOA>",
        eoa_string: Optional[str] = "<EOA>",
    ):
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt_format: The prompt format string.
            max_seq_length: The maximum sequence length.
            add_boa_eoa: Whether to add BOA and EOA strings before and after audio.
            boa_string: The BOA string to use.
            eoa_string: The EOA string to use.
        """
        super().__init__()
        self.prompt = PromptFormatter.resolve(prompt_format)(tokenizer)
        self.prompt_format_fn = get_prompt_format_fn(NeMoMultimodalConversation, self.prompt)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if max_seq_length is None or max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be a positive integer, got {max_seq_length}")

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id != None and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = (
                self.tokenizer.eos_id if self.tokenizer.eos_id is not None and self.tokenizer.eos_id > 0 else 0
            )
        self.add_boa_eoa = add_boa_eoa
        self.boa_string = boa_string
        self.eoa_string = eoa_string

    def __call__(self, lhotse_input: NeMoMultimodalConversation) -> TextProcessorOutput:
        """
        process a single input sample.
        Args:
            lhotse_input: a NeMoMultimodalConversation sample from lhotse dataset.
        """
        return self.process_sample(lhotse_input)

    def process_sample(self, lhotse_input: NeMoMultimodalConversation) -> TextProcessorOutput:
        """
        process a single input sample.
        Args:
            lhotse_input: a NeMoMultimodalConversation sample from lhotse dataset.
        """
        if not isinstance(lhotse_input, NeMoMultimodalConversation):
            raise ValueError(f"Input must be of type NeMoMultimodalConversation, got {type(lhotse_input)}")

        audio_turns = [turn for turn in lhotse_input.turns if isinstance(turn, AudioTurn)]
        num_audios = len(audio_turns)
        audio_locator_str = audio_turns[0].audio_locator_tag if num_audios > 0 else None

        processed_sample = self.prompt_format_fn(lhotse_input, self.prompt)

        if num_audios == 0:
            context_start_idx = [0]
            input_ids = processed_sample["input_ids"].cpu().numpy().tolist()
            context_ids = processed_sample["context_ids"].cpu().numpy().tolist()
            answer_ids = processed_sample["answer_ids"].cpu().numpy().tolist()
        else:
            if isinstance(self.tokenizer, AutoTokenizer):
                # HF tokenizer skips special tokens when converting ids to text by default,
                # which makes text_to_ids and ids_to_text not inverses of each other.
                context = self.tokenizer.ids_to_text(processed_sample["context_ids"], remove_special_tokens=False)
            else:
                context = self.tokenizer.ids_to_text(processed_sample["context_ids"])
            context_ids = []
            context_start_idx = []
            segments = context.split(audio_locator_str)
            for i, context_seg in enumerate(segments):
                context_start_idx.append(len(context_ids))
                if self.add_boa_eoa:
                    if i == 0:
                        context_seg = context_seg + ' ' + self.boa_string
                    elif i == len(segments) - 1:
                        context_seg = self.eoa_string + ' ' + context_seg
                    else:
                        context_seg = self.eoa_string + ' ' + context_seg + ' ' + self.boa_string
                context_ids.extend(self.tokenizer.text_to_ids(context_seg))
            answer_ids = processed_sample["answer_ids"].cpu().numpy().tolist()
            input_ids = context_ids + answer_ids

        if len(input_ids) > self.max_seq_length:
            logging.warning(
                f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}, truncating.'
            )
            input_ids = input_ids[: self.max_seq_length]

        return TextProcessorOutput(
            input_ids=torch.tensor(input_ids).long(),
            answer_start_idx=torch.tensor(len(context_ids)).long(),
            context_ids=torch.tensor(context_ids).long(),
            context_length=torch.tensor(len(context_ids)).long(),
            answer_ids=torch.tensor(answer_ids).long(),
            context_start_idx=torch.tensor(context_start_idx).long(),
            num_audios=torch.tensor(num_audios),
        )
