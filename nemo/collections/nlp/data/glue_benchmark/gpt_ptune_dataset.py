# Copyright 2018 The Google AI Language Team Authors and
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

import csv
import functools
import json
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.megatron.utils import (
    make_inference_attention_mask_3d,
    make_inference_history_mask_3d,
)
from nemo.core.classes import Dataset
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType
from nemo.utils import logging

__all__ = [
    'DataProcessor',
    'GPTPTuneDataset',
    'register_taskdata_processor',
    'GPTPTuneInferenceDataset',
    'TemplateProcessor',
]

SMALL_NUM = -100
TASK_KEY = 'prompt_tag'
LABEL_KEY = 'label'


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self):
        pass

    def create_example(self, content, set_type):
        """Creates examples for the training and dev sets."""
        return InputExample(content=content, processor=self)

    def label2string(self, label):
        raise NotImplementedError()

    def get_ptune_query(
        self, content: Dict, prompt_token_id: int, max_seq_len: int, templates: List[int], tokenizer: TokenizerSpec,
    ):
        raise NotImplementedError()


class InputExample(object):
    """A single training/test example, The default key for label is `label`
    Args:
        content: data content in Python Dict
        processor: the data processor for a particular task.
    """

    def __init__(self, content: Dict, processor: DataProcessor = None):
        """Constructs a InputExample."""
        self.content = content
        self.processor = processor

    @property
    def label(self):
        return self.content.get(LABEL_KEY, None)

    @property
    def taskname(self):
        return ' ' + self.content.get(TASK_KEY, None)

    def __repr__(self):
        return f"InputExample(content='{self.content}')"


class TemplateProcessor(DataProcessor):
    """Processor convert the input data according to template. 
        E.g. template "{v0} {var1} some text {v1} {var2} some text {var3} {v2}"
    """

    def __init__(self, template: str, limit_length_field: str):
        super().__init__()
        start = 0
        self.pieces = []
        while True:
            result = re.search('{v\d}', template)
            if result is None:
                break
            start = result.end()
            sentence = template[: result.start()]
            if len(sentence) != 0:
                self.pieces.append(sentence)
            self.pieces.append(int(template[result.start() + 2 : start - 1]))
            template = template[start:]
        sentence = template
        if len(sentence) != 0:
            self.pieces.append(sentence)
        self.limit_length_field = limit_length_field

    def get_ptune_query(
        self, content: Dict, prompt_token_id: int, max_seq_len: int, templates: List[int], tokenizer: TokenizerSpec,
    ):
        all_ids = []
        limits = []
        for piece in self.pieces:
            if isinstance(piece, str):
                # replace variables if any
                variables = re.findall(r'{\w*}', piece)
                variable_text = {}
                limit_length = False
                for var in variables:
                    varname = var[1:-1]
                    variable_text[varname] = content[varname]
                    if varname == self.limit_length_field:
                        limit_length = True
                text = piece.format(**variable_text)
                text_ids = tokenizer.text_to_ids(text)
                all_ids.append(text_ids)
                limits.append(limit_length)
            else:
                # this is virtual token
                all_ids.append([prompt_token_id] * templates[piece])
                limits.append(False)
        total_num_of_ids = sum([len(i) for i in all_ids])
        if total_num_of_ids > max_seq_len:
            logging.warning("Input sequence is longer than the LM model max seq, will cut it off to fit")
            cut = total_num_of_ids - max_seq_len
            new_ids = []
            for i in range(len(limits)):
                if limits[i]:
                    if len(all_ids[i]) < cut:
                        raise ValueError(
                            f"Some other field length is too long, cutting {self.limit_length_field} is not enough"
                        )
                    new_ids.append(all_ids[i][cut:])
                else:
                    new_ids.append(all_ids[i])
            return functools.reduce(lambda x, y: x + y, new_ids)
        else:
            return functools.reduce(lambda x, y: x + y, all_ids)

    def label2string(self, label):
        return ' ' + label


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def __init__(self):
        super().__init__()

    def get_ptune_query(
        self, content: Dict, prompt_token_id: int, max_seq_len: int, templates: List[int], tokenizer: TokenizerSpec,
    ):
        text_a = content['sentence']
        text_b = content['question']
        sentence_a = f" Paragraph: {text_a}"
        sentence_b = f" Question: {text_b}?"
        a_input_token_ids = tokenizer.text_to_ids(sentence_a)
        b_input_token_ids = tokenizer.text_to_ids(sentence_b)
        c_input_token_ids = tokenizer.text_to_ids(" Answer:")
        cut = 0
        total_num_ids = len(a_input_token_ids) + len(b_input_token_ids) + len(c_input_token_ids) + sum(templates)
        if total_num_ids > max_seq_len:
            logging.warning("Input sequence is longer than the LM model max seq, will cut it off to fit")
            cut = total_num_ids - max_seq_len
        return (
            [prompt_token_id] * templates[0]
            + a_input_token_ids[cut:]
            + [prompt_token_id] * templates[1]
            + b_input_token_ids
            + [prompt_token_id] * templates[2]
            + c_input_token_ids
        )

    def label2string(self, label):
        return ' ' + label


class SentimentProcessor(DataProcessor):
    """Processor for the sentiment analysis data set."""

    def __init__(self):
        super().__init__()

    def get_ptune_query(
        self, content: Dict, prompt_token_id: int, max_seq_len: int, templates: List[int], tokenizer: TokenizerSpec,
    ):
        text_a = content['sentence']
        sentence_a = f" Sentence: {text_a}"
        sentence_b = f" Sentiment:"
        a_input_token_ids = tokenizer.text_to_ids(sentence_a)
        b_input_token_ids = tokenizer.text_to_ids(sentence_b)
        cut = 0
        total_num_ids = len(a_input_token_ids) + len(b_input_token_ids) + sum(templates)
        if total_num_ids > max_seq_len:
            logging.warning("Input sequence is longer than the LM model max seq, will cut it off to fit")
            cut = total_num_ids - max_seq_len
        return (
            [prompt_token_id] * (templates[0] + templates[1])
            + a_input_token_ids[cut:]
            + [prompt_token_id] * templates[2]
            + b_input_token_ids
        )

    def label2string(self, label):
        return ' ' + label


processors = {"boolq-full-text": BoolQProcessor(), "sentiment-bankp": SentimentProcessor()}


def register_taskdata_processor(taskname: str, processor: DataProcessor):
    processors[taskname] = processor


class TaskDataset(Dataset):
    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(tuple('B'), CategoricalValuesType()),
        }

    def __init__(
        self, file_name: str, data_type: str, tokenizer: TokenizerSpec,
    ):
        """
        Processes Task datasets
        Args:
            file_name: path to file
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
            use_cache: whether to use data cache
        """
        logging.info(f'Processing {file_name}')
        self.tokenizer = tokenizer
        file_contents = self._read_jsonl(file_name)
        self.examples = []
        for content in file_contents:
            task_name = content[TASK_KEY]
            processor = processors[task_name]
            self.examples.append(processor.create_example(content, data_type))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return (
            np.array(feature.input_ids),
            np.array(feature.segment_ids),
            np.array(feature.input_mask, dtype=np.long),
            np.array(feature.label_id),
        )


class GPTPTuneDataset(TaskDataset):
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
        input_ids, labels, query, taskname = self.features[idx]
        return {'input_enc': input_ids, 'labels': labels, 'query_enc': query, 'taskname_enc': taskname}

    def collate_fn(self, batch):
        enc_input = [item['input_enc'] for item in batch]
        labels = [item['labels'] for item in batch]
        enc_query = [item['query_enc'] for item in batch]
        enc_taskname = [item['taskname_enc'] for item in batch]

        max_input_length = max([len(item) for item in enc_input])
        max_label_length = max([len(item) for item in labels])
        max_query_length = max([len(item) for item in enc_query])
        max_taskname_length = max([len(item) for item in enc_taskname])

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_input = [item + [self.pad_id] * (max_input_length - len(item)) for item in enc_input]
        labels = [item + [self.pad_id] * (max_label_length - len(item)) for item in labels]
        enc_query = [item + [self.pad_id] * (max_query_length - len(item)) for item in enc_query]
        enc_taskname = [item + [self.pad_id] * (max_taskname_length - len(item)) for item in enc_taskname]

        enc_query = torch.LongTensor(enc_query)
        enc_input = torch.LongTensor(enc_input)
        labels = torch.LongTensor(labels)
        enc_taskname = torch.LongTensor(enc_taskname)
        loss_mask = torch.LongTensor(loss_mask)
        label_position = loss_mask.sum(axis=1)
        label_start = (labels == SMALL_NUM).sum(axis=1)
        label_position = torch.cat([label_start.unsqueeze(1), label_position.unsqueeze(1)], 1)
        loss_mask[labels == SMALL_NUM] = 0

        input_attn_mask = make_inference_attention_mask_3d(enc_input, enc_input, self.pad_id)
        input_attn_mask = (input_attn_mask * make_inference_history_mask_3d(enc_input)).long()

        return {
            'enc_input': enc_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_query': enc_query,
            'input_attn_mask': input_attn_mask,
            'label_position': label_position,
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
            label_ids = self.tokenizer.text_to_ids(processor.label2string(example.label)) + [self.tokenizer.eos_id]
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
            label_ids = labels_list[ex_index]
            enc_query = processor.get_ptune_query(
                example.content,
                self.pseudo_token_id,
                self.max_seq_length - self.max_seq_length_decoder + 1,
                self.templates,
                self.tokenizer,
            )
            input_ids = enc_query + label_ids[:-1]
            labels = [SMALL_NUM for i in range(len(enc_query) - 1)] + label_ids
            features.append([input_ids, labels, enc_query, taskname_ids])
        return features


class GPTPTuneInferenceDataset(TaskDataset):
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
        query, taskname = self.features[idx]
        return {'query_enc': query, 'taskname_enc': taskname}

    def collate_fn(self, batch):
        enc_query = [item['query_enc'] for item in batch]
        enc_taskname = [item['taskname_enc'] for item in batch]

        label_start = [len(item) - 1 for item in enc_query]
        max_query_length = max(label_start)
        max_taskname_length = max([len(item) for item in enc_taskname])

        enc_query = [item + [self.pad_id] * (max_query_length - len(item)) for item in enc_query]
        enc_taskname = [item + [self.pad_id] * (max_taskname_length - len(item)) for item in enc_taskname]

        enc_query = torch.LongTensor(enc_query)
        label_start = torch.LongTensor(label_start)
        enc_taskname = torch.LongTensor(enc_taskname)

        label_position = torch.cat([label_start.unsqueeze(1), label_start.unsqueeze(1)], 1)
        return {
            'enc_query': enc_query,
            'label_position': label_position,
            'enc_taskname': enc_taskname,
        }

    def convert_examples_to_features(self):
        """
        Converts examples into Text-to-Text batches to be used with a model like T5.
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
