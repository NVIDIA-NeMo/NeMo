# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""
Prompt tuning dataset
Expects data to be in the format:
{"prompt_tag": "tag1", "text": "example question1", "answer": "answer1"}
{"prompt_tag": "tag1", "text": "example question2", "answer": "answer2"}
{"prompt_tag": "tag2", "text": "example question3", "answer": "answer3"}

"""
import json

import torch
from tqdm import tqdm

from nemo.core import Dataset
from nemo.utils import logging

__all__ = ["GPTPromptTuningDataset"]


class GPTPromptTuningDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
        prompt_table,
        num_prompt_tokens: int,
        micro_batch_size: int,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        calc_loss_on_answer_only=False,
    ):
        self.tokenizer = tokenizer
        self.prompt_tag_to_id = dict(prompt_table)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.calc_loss_on_answer_only = calc_loss_on_answer_only
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.num_prompt_tokens = num_prompt_tokens
        self.micro_batch_size = micro_batch_size
        self.max_sent_length = max_seq_length - num_prompt_tokens
        self.prompt_ids_and_tokens = []

        print(f"\n\nMicro batch size: {micro_batch_size}")

        assert min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert max_seq_length > 0, "Max sequence length should be greater than 0"
        assert num_prompt_tokens > 0, "There should be at least one soft prompt token"
        assert num_prompt_tokens < max_seq_length, "Soft prompt tokens should not exceed max sequence length"

        dataset_file = open(dataset_path, 'r', encoding='utf-8')

        logging.info("Loading and tokenizing dataset ... ")

        skipped = 0
        for json_line in tqdm(dataset_file):
            doc = json.loads(json_line)
            prompt_tag = doc["prompt_tag"]
            question = str(doc["text"])
            answer = str(doc["answer"])  # Incase 'True' or 'False' gets read as bool
            sent = question + answer

            sent_ids = tokenizer.text_to_ids(sent)
            answer_ids = tokenizer.text_to_ids(answer)
            answer_len = len(answer_ids)

            if self.add_bos:
                sent_ids = [tokenizer.bos_id] + sent_ids

            if self.add_eos:
                sent_ids = sent_ids + [tokenizer.eos_id]
                answer_len += 1  # To account for EOS token

            # Need to leave space for prompt tokens in sequence
            if self.min_seq_length <= len(sent_ids) <= self.max_sent_length:
                prompt_id = self.prompt_tag_to_id[prompt_tag]
                self.prompt_ids_and_tokens.append((prompt_id, sent_ids, answer_len))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too long or too short')

    def __len__(self):
        return len(self.prompt_ids_and_tokens)

    def __getitem__(self, idx):
        return self.prompt_ids_and_tokens[idx]

    def collate_fn(self, batch):
        """ Prepares global batch, then splits into micro batches if pipeline parallel is > 1"""

        prompt_ids, input_ids, answer_lens = zip(*batch)
        prompt_ids = torch.tensor(prompt_ids)

        # Prepare global batch
        tokens, labels, loss_mask, attention_mask, text_position_ids = self.process_global_batch(
            input_ids, answer_lens,
        )

        return tokens, labels, loss_mask, attention_mask, text_position_ids, prompt_ids

    def process_global_batch(self, input_ids, answer_lens):
        """ Perpare tokens, labels, loss mask, attention_mask, and position ids for global batch """
        # Get max sequence length of batch
        batch_size = len(input_ids)
        batch_max = max(len(ids) for ids in input_ids)
        tokens, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, answer_lens, batch_max)

        # Labels for prompt tokens, just padding because the loss mask masks these out
        prompt_token_labels = torch.full(
            size=(batch_size, self.num_prompt_tokens - 1), fill_value=self.tokenizer.bos_id, dtype=torch.long,
        )

        # Should be a label for every token in batch, label is the next token, starting with the virtual tokens
        labels = torch.cat((prompt_token_labels, tokens.contiguous()), dim=1)
        tokens = tokens[:, :-1].contiguous()
        text_position_ids, attention_mask = self.get_ltor_attention_mask_and_position_ids(batch_size, tokens)

        return tokens, labels, loss_mask, attention_mask, text_position_ids

    def pad_batch_and_build_loss_mask(self, input_ids, answer_lens, batch_max):
        """ Pad tokens in batch to max batch length while building loss mask """
        loss_mask = []
        for idx, ids in enumerate(input_ids):
            text_length = len(ids)
            answer_length = answer_lens[idx]

            # Loss mask should match labels
            # Subtracting one because loss mask should align with labels
            prompt_loss_mask = [0.0] * (self.num_prompt_tokens - 1)

            # Loss mask everything except the answer
            if self.calc_loss_on_answer_only:
                question_loss_mask = [0.0] * (text_length - answer_length)
                answer_loss_mask = [1.0] * answer_length
                text_loss_mask = prompt_loss_mask + question_loss_mask + answer_loss_mask

            # Loss mask soft prompt and padding only, calc loss on all text after soft prompt
            else:
                text_loss_mask = [1.0] * text_length
                text_loss_mask = prompt_loss_mask + text_loss_mask

            # Pad loss mask and text tokens
            padding_length = batch_max - text_length
            ids.extend([self.tokenizer.eos_id] * padding_length)
            text_loss_mask.extend([0.0] * padding_length)
            loss_mask.append(torch.tensor(text_loss_mask, dtype=torch.float))

        # Make into a torch tensor
        tokens = torch.tensor(input_ids, dtype=torch.long)
        loss_mask = torch.stack(loss_mask)

        return tokens, loss_mask

    def get_ltor_attention_mask_and_position_ids(self, batch_size, tokens):
        """ Makes prompt tuning left to right attention mask and position ids.
            position ids for text start after soft tokens. Position ids for soft
            prompts are always the same so they are automatically infered during
            the forward pass
        """

        # Full length of every sequence in the batch
        full_seq_length = len(tokens[0]) + self.num_prompt_tokens

        # Position ids for text
        text_position_ids = torch.arange(start=self.num_prompt_tokens, end=full_seq_length, dtype=torch.long,)
        text_position_ids = text_position_ids.unsqueeze(0).expand_as(tokens).clone()

        # Attention mask (lower triangular) starting with prompt tokens
        attention_mask = torch.tril(torch.ones((batch_size, full_seq_length, full_seq_length))).view(
            batch_size, 1, full_seq_length, full_seq_length
        )

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

        return text_position_ids, attention_mask
