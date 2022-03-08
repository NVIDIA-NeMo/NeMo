# Copyright 2022 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Some code of this file was adapted from the HuggingFace library available at
# https://github.com/huggingface/transformers

from typing import Dict, List, Optional

import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.glue_benchmark.gpt_ptune_dataset import (
    TaskDataset,
    TemplateProcessor,
    processors,
    register_taskdata_processor,
)
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = [
    'T5PTuneDataset',
    'register_taskdata_processor',
    'T5PTuneInferenceDataset',
    'TemplateProcessor',
]

SMALL_NUM = -100
TASK_KEY = 'prompt_tag'
LABEL_KEY = 'label'


class T5PTuneDataset(TaskDataset):
    """Multiple Task Dataset used in P-Tuning models."""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return

    def __init__(
        self,
        file_name: str,
        data_type: str,
        tokenizer: TokenizerSpec,
        templates: List[int],
        pseudo_token_id: int,
        pad_id: int,
        max_seq_length: int,
        max_seq_length_decoder: int = None,
    ):
        """
        Processes TextToText PTuning Dataset
        Args:
            file_name: path to file
            data_type: train/dev/test
            tokenizer: such as AutoTokenizer
            templates: virtual token template, list of integers
            max_seq_length: max sequence length for encoder
            max_seq_length_decoder: max seq length for decoder
        """
        super().__init__(file_name, data_type, tokenizer)
        self.max_seq_length = max_seq_length
        self.max_seq_length_decoder = max_seq_length_decoder
        self.templates = templates
        self.pseudo_token_id = pseudo_token_id
        self.pad_id = pad_id
        self.features = self.convert_examples_to_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc_query, dec_input, labels, taskname = self.features[idx]
        return {'text_enc': enc_query, 'text_dec': dec_input, 'labels': labels, 'taskname_enc': taskname}

    def collate_fn(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]
        enc_taskname = [item['taskname_enc'] for item in batch]

        max_dec_input_length = max([len(item) for item in dec_input]) if dec_input else 0
        max_enc_query_length = max([len(item) for item in enc_query]) if enc_query else 0
        max_label_length = max([len(item) for item in labels]) if labels else 0
        max_taskname_length = max([len(item) for item in enc_taskname])

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]
        enc_taskname = [item + [self.pad_id] * (max_taskname_length - len(item)) for item in enc_taskname]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)
        enc_taskname = torch.LongTensor(enc_taskname)

        enc_mask = (enc_query != self.tokenizer.pad_id).long()
        dec_mask = (dec_input != self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'enc_taskname': enc_taskname,
        }

    def convert_examples_to_features(self):
        """
        Converts examples into Text-to-Text batches to be used in GPT
        Inputs are prefixed with a text prompt that indicates the task to perform.
        """
        features = []
        max_label_len = 0
        # find ou the max label length
        labels_list = []
        for ex_index, example in enumerate(self.examples):
            processor = example.processor
            label_ids = (
                [self.tokenizer.bos_id]
                + self.tokenizer.text_to_ids(processor.label2string(example.label))
                + [self.tokenizer.eos_id]
            )
            max_label_len = max(len(label_ids), max_label_len)
            labels_list.append(label_ids)
        if self.max_seq_length_decoder is None:
            self.max_seq_length_decoder = max_label_len
        else:
            self.max_seq_length_decoder = max(
                self.max_seq_length_decoder, max_label_len
            )  # take the max of the two to be conservative
        for ex_index, example in enumerate(self.examples):
            taskname = example.taskname
            taskname_ids = self.tokenizer.text_to_ids(taskname)
            processor = example.processor
            if ex_index % 10000 == 0:
                logging.info(f"Writing example {ex_index} of {len(self.examples)}")
            enc_query = processor.get_ptune_query(
                example.content,
                self.pseudo_token_id,
                self.max_seq_length - self.max_seq_length_decoder + 1,
                self.templates,
                self.tokenizer,
            )
            dec_query = labels_list[ex_index]
            dec_input = dec_query[:-1]
            labels = dec_query[1:]
            features.append([enc_query, dec_input, labels, taskname_ids])
        return features


class T5PTuneInferenceDataset(TaskDataset):
    """Multiple Task Dataset used in P-Tuning inference. Assumes no label in the data"""

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return

    def __init__(
        self,
        queries: List[Dict],
        data_type: str,
        tokenizer: TokenizerSpec,
        templates: List[int],
        pseudo_token_id: int,
        pad_id: int,
        max_seq_length: int,
        max_seq_length_decoder: int = None,
    ):
        """
        Processes TextToText PTuning Dataset
        Args:
            queries: query content
            data_type: train/dev/test
            tokenizer: such as AutoTokenizer
            templates: virtual token template, list of integers
            max_seq_length: max sequence length for encoder
            max_seq_length_decoder: max seq length for decoder
        """
        logging.info(f'Processing...')
        self.tokenizer = tokenizer
        file_contents = queries
        self.examples = []
        for content in file_contents:
            task_name = content[TASK_KEY]
            processor = processors[task_name]
            self.examples.append(processor.create_example(content, data_type))

        self.max_seq_length = max_seq_length
        self.max_seq_length_decoder = max_seq_length_decoder
        self.templates = templates
        self.pseudo_token_id = pseudo_token_id
        self.pad_id = pad_id
        self.features = self.convert_examples_to_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc_query, taskname = self.features[idx]
        return {'text_enc': enc_query, 'taskname_enc': taskname}

    def collate_fn(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        enc_taskname = [item['taskname_enc'] for item in batch]

        max_enc_query_length = max([len(item) for item in enc_query]) if enc_query else 0
        max_taskname_length = max([len(item) for item in enc_taskname])

        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        enc_taskname = [item + [self.pad_id] * (max_taskname_length - len(item)) for item in enc_taskname]

        enc_query = torch.LongTensor(enc_query)
        enc_taskname = torch.LongTensor(enc_taskname)

        enc_mask = (enc_query != self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'enc_mask': enc_mask,
            'enc_taskname': enc_taskname,
        }

    def convert_examples_to_features(self):
        """
        Converts examples into Text-to-Text batches to be used in GPT
        Inputs are prefixed with a text prompt that indicates the task to perform.
        """
        features = []
        for ex_index, example in enumerate(self.examples):
            taskname = example.taskname
            taskname_ids = self.tokenizer.text_to_ids(taskname)
            processor = example.processor
            if ex_index % 10000 == 0:
                logging.info(f"Writing example {ex_index} of {len(self.examples)}")
            enc_query = processor.get_ptune_query(
                example.content,
                self.pseudo_token_id,
                self.max_seq_length - self.max_seq_length_decoder + 1,
                self.templates,
                self.tokenizer,
            )
            features.append([enc_query, taskname_ids])
        return features
