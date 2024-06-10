# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

import os

import numpy as np
import torch
from tqdm import trange

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import INFERENCE_MODE, TRAINING_MODE
from nemo.collections.nlp.data.question_answering.dataset.qa_dataset import QADataset
from nemo.collections.nlp.data.question_answering.input_example.qa_s2s_input_example import S2SQAInputExample
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning


class S2SQADataset(QADataset):
    """Creates a Dataset for T5/BART architecture based Generative QA"""

    def __init__(
        self,
        data_file: str,
        processor: object,
        tokenizer: object,
        keep_doc_spans: str = False,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_seq_length: int = 512,
        max_answer_length: int = 64,
        check_if_answer_in_context: bool = False,
        num_samples: int = -1,
        mode: str = TRAINING_MODE,
        use_cache: bool = False,
    ):
        # deprecation warning
        deprecated_warning("S2SQADataset")

        super().__init__(
            data_file=data_file, processor=processor, tokenizer=tokenizer, mode=mode, num_samples=num_samples
        )

        self.keep_doc_spans = keep_doc_spans
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.check_if_answer_in_context = check_if_answer_in_context
        self.num_samples = num_samples
        self.mode = mode
        self.use_cache = use_cache

        self._set_cached_features_filename()
        if use_cache and os.path.exists(self.cached_features_file):

            # delete self.examples during training mode to save memory
            if self.mode == TRAINING_MODE:
                del self.examples
                del self.processor
            self.features = QADataset.load_features_from_cache(self.cached_features_file)
        else:
            self._convert_examples_to_features()
            if use_cache:
                QADataset.dump_features_to_cache(self.cached_features_file, self.features)

        logging.info("Converting dict features into object features")
        for i in trange(len(self.features)):
            self.features[i] = S2SQAInputExample(**self.features[i])

    def _set_cached_features_filename(self):
        """Creates cache filename using dataset config parameters"""

        vocab_size = getattr(self.tokenizer, "vocab_size", 0)
        self.cached_features_file = (
            self.data_file
            + '_cache'
            + '_{}_{}_{}_{}_{}_{}_{}'.format(
                self.mode,
                self.tokenizer.name,
                str(vocab_size),
                str(self.max_query_length),
                str(self.max_seq_length),
                str(self.max_answer_length),
                str(self.num_samples),
            )
        )

    def _convert_examples_to_features(self):
        """
        Iterates through each QA example, formats into input and output template,
            and encodes the input and output template
        Input template: `context: <context text> question: <question text>`
        Output template: `<answer text>`
        """

        logging.info(f"Preprocessing data into features.")

        unique_id = 1000000000
        self.features = []
        context_prefix = "context: "
        context_prefix_tokens = self.tokenizer.tokenizer.tokenize(context_prefix)

        for example_index in trange(len(self.examples)):
            if example_index % 1000 == 0:
                S2SQADataset.check_if_sufficient_memory()

            example = self.examples[example_index]

            query_tokens, formatted_query = self._prep_query(example)
            context_tokens, context_spans = self._prep_context(example, query_tokens, context_prefix_tokens)

            unique_id = self._encode_all_context_spans(
                unique_id,
                context_spans,
                context_tokens,
                formatted_query,
                example,
                example_index,
            )

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            self.examples = []
            del self.processor

    def _prep_query(self, example):
        """
        Formats a question into input format: ` question: <question text>`
        The space at the start allows concatention with the context for input
        """
        formatted_query = f" question: {example.question_text}"
        query_tokens = self.tokenizer.tokenizer.tokenize(formatted_query)[: self.max_query_length]

        return query_tokens, formatted_query

    def _prep_context(self, example, query_tokens, context_prefix_tokens):
        """
        Calculates the maximum possible length for a given context given a question
            as inputs are of fixed length
        Divides the context into multiple spans based on the calculated max length
        """

        context_tokens = self.tokenizer.tokenizer.tokenize(example.context_text)
        max_context_length = (
            self.max_seq_length
            - len(query_tokens)
            - len(context_prefix_tokens)
            - 1  # -1 accounts for </s> token in T5/BART
        )
        context_spans = S2SQADataset.get_docspans(context_tokens, max_context_length, self.doc_stride)
        context_spans = tuple(context_spans)

        return context_tokens, context_spans

    def _encode_all_context_spans(
        self,
        unique_id,
        context_spans,
        context_tokens,
        formatted_query,
        example,
        example_index,
    ):
        """
        Fromats all spans extracted from a single context as:
            `context: <context span text> question: <question text> answer: <answer text>` and encodes
        If the answer text (example.answer_text) is not present in a given context span,
            the answer is converted to a blank answer
        """

        for context_span_idx, context_span in enumerate(context_spans):

            # format query and context span text
            context_span_tokens = context_tokens[context_span.start : context_span.start + context_span.length]
            context_span_text = self.tokenizer.tokenizer.convert_tokens_to_string(context_span_tokens)
            source = f"context: {context_span_text}{formatted_query}"

            # encode input
            encoded_input_dict = self.tokenizer.tokenizer(
                source,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = torch.squeeze(encoded_input_dict["input_ids"])
            input_attn_mask = torch.squeeze(encoded_input_dict["attention_mask"])

            # encode output based on mode and is question answerable given context
            labels = self._encode_answer(example, context_span_text)

            # create dictionary features
            feature = {
                "unique_id": unique_id,
                "input_ids": input_ids,
                "input_attn_mask": input_attn_mask,
                "labels": labels,
                "example_index": example_index,
                "context_span_index": context_span_idx,
                "is_impossible": example.is_impossible,
            }

            self.features.append(feature)
            unique_id += 1

        return unique_id

    def _encode_answer(self, example, context_span_text):
        """
        Answer is set and encoded as:
            - blank if in inference mode, else
            - blank if question is unanswerable given context, else
            - blank if answer text is not present in context span
                and the check flag is set to true, else
            - formatted answer
        """

        is_answer_in_context_check = (
            self.check_if_answer_in_context  # checks if the flag for this check is set
            and example.answer_text  # checks if answer text is valid, i.e. question is not unanswerable
            and example.answer_text not in context_span_text  # checks if answer text is a substring of context
        )

        if (
            self.mode == INFERENCE_MODE
            or example.is_impossible  # question not answerable given context
            or is_answer_in_context_check
        ):
            target = ""
        else:
            target = example.answer_text

        encoded_output_dict = self.tokenizer.tokenizer(
            target,
            truncation=True,
            max_length=self.max_answer_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.squeeze(encoded_output_dict["input_ids"])
        labels[labels == self.tokenizer.tokenizer.pad_token_id] = -100

        return labels

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        if self.mode == INFERENCE_MODE:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
            )
        else:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
                np.array(feature.labels),
            )
