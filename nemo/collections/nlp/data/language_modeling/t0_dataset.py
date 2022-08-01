# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json

import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import TextMemMapDataset


class T0JSONLMemMapDataset(TextMemMapDataset):
    """
    Memory-mapped iteration over a JSONL file.
    """

    def __init__(
        self,
        dataset_paths,
        newline_int=10,
        header_lines=1,
        workers=None,
        tokenizer=None,
        sort_dataset_paths=True,
        max_src_seq_length=512,
        max_tgt_seq_length=512,
    ):
        super().__init__(
            dataset_paths=dataset_paths,
            newline_int=newline_int,
            header_lines=header_lines,
            workers=workers,
            tokenizer=None,  # Make sure parent tokenizer is None so that it returns the raw JSON string
            sort_dataset_paths=sort_dataset_paths,
        )
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided to use T0JSONLMemMapDataset, got None")
        self.tokenizer = tokenizer
        self.max_src_seq_length = max_src_seq_length
        self.max_tgt_seq_length = max_tgt_seq_length

    def _build_data_from_text(self, text):
        """Return a CSV field from text"""
        example = json.loads(text)
        tokenized_input = self.tokenizer.text_to_ids(example['input'])
        tokenized_output = self.tokenizer.text_to_ids(example['output'])
        if len(tokenized_input) > self.max_src_seq_length - 2:
            tokenized_input = tokenized_input[: self.max_src_seq_length - 2]
        if len(tokenized_output) > self.max_tgt_seq_length - 2:
            tokenized_output = tokenized_output[: self.max_tgt_seq_length - 2]
        text_enc = [self.tokenizer.bos_id] + tokenized_input + [self.tokenizer.eos_id]
        target = [self.tokenizer.bos_id] + tokenized_output + [self.tokenizer.eos_id]
        original = ""
        template = ""
        for item in example['chunked_idx'].split(', '):
            item = item.split('-')
            if item[0] == "original_text":
                original += example['input'][int(item[1]) : int(item[2])]
            elif item[0] == "template":
                template += example['input'][int(item[1]) : int(item[2])]
            else:
                raise ValueError(f"Unknown chunk type: {item[0]}")

        return {
            'text_enc': text_enc,
            'text_dec': target[:-1],
            'labels': target[1:],
            'original': self.tokenizer.text_to_ids(original),
            'template': self.tokenizer.text_to_ids(template),
            'prompt': self.tokenizer.text_to_ids(example['prompt']),
        }

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def collate_fn(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]
        original = [item['original'] for item in batch]
        template = [item['template'] for item in batch]
        prompt = [item['prompt'] for item in batch]

        enc_query = self._maybe_cast_to_list(enc_query)
        dec_input = self._maybe_cast_to_list(dec_input)
        labels = self._maybe_cast_to_list(labels)
        original = self._maybe_cast_to_list(original)
        template = self._maybe_cast_to_list(template)
        prompt = self._maybe_cast_to_list(prompt)

        max_dec_input_length = max([len(item) for item in dec_input]) if dec_input else 0
        max_enc_query_length = max([len(item) for item in enc_query]) if enc_query else 0
        max_label_length = max([len(item) for item in labels]) if labels else 0
        max_original_length = max([len(item) for item in original]) if original else 0
        max_template_length = max([len(item) for item in template]) if template else 0
        max_prompt_length = max([len(item) for item in prompt]) if prompt else 0

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]
        original = [item + [self.tokenizer.pad_id] * (max_original_length - len(item)) for item in original]
        template = [item + [self.tokenizer.pad_id] * (max_template_length - len(item)) for item in template]
        prompt = [item + [self.tokenizer.pad_id] * (max_prompt_length - len(item)) for item in prompt]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)
        original = torch.LongTensor(original)
        template = torch.LongTensor(template)
        prompt = torch.LongTensor(prompt)

        enc_mask = (enc_query != self.tokenizer.pad_id).long()
        dec_mask = (dec_input != self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'original': original,
            'template': template,
            'prompt': prompt,
        }
