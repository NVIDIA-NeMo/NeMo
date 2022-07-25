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

import copy
import os

import numpy as np
import torch
from tqdm import trange

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import INFERENCE_MODE, TRAINING_MODE
from nemo.collections.nlp.data.question_answering.dataset.qa_dataset import QADataset
from nemo.collections.nlp.data.question_answering.input_example.qa_gpt_input_example import GPTQAInputExample
from nemo.utils import logging


class GPTQADataset(QADataset):
    """ Creates a Dataset for GPT architecture based Generative QA """

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
            self.features[i] = GPTQAInputExample(**self.features[i])

    def _set_cached_features_filename(self):
        """ Creates cache filename using dataset config parameters """

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
        Iterates through each QA example, formats into template and encodes
        Template: `context: <context text> question: <question text> answer:<answer text>`
        """

        logging.info(f"Preprocessing data into features.")

        unique_id = 1000000000
        self.features = []

        context_prefix = "context: "
        query_prefix = " question: "
        answer_prefix = " answer:"

        context_prefix_tokens = self.tokenizer.tokenizer.tokenize(context_prefix)
        answer_prefix_tokens = self.tokenizer.tokenizer.tokenize(answer_prefix)

        for example_index in trange(len(self.examples)):
            if example_index % 1000 == 0:
                GPTQADataset.check_if_sufficient_memory()

            example = self.examples[example_index]

            formatted_query, query_tokens_length = self._prep_query(query_prefix, example)
            formatted_answer, answer_tokens_length = self._prep_answer(example)
            context_tokens, context_spans = self._prep_context(
                example, query_tokens_length, answer_tokens_length, context_prefix_tokens, answer_prefix_tokens,
            )

            unique_id = self._encode_all_context_spans(
                unique_id,
                context_spans,
                context_tokens,
                context_prefix,
                formatted_query,
                answer_prefix,
                formatted_answer,
                example,
                example_index,
            )

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            self.examples = []
            del self.processor

    def _prep_query(self, query_prefix, example):
        """
        Formats a question into input format: ` question: <question text>`
            The space at the start allows concatention with the context and answer for input
        Returns formatted query, query tokens, and length of query tokens
        """

        formatted_query = f"{query_prefix}{example.question_text}"

        return self._get_truncated_sentence_and_len(formatted_query, self.max_query_length)

    def _prep_answer(self, example):
        """
        Formats an answer into suitable model input:
            - In inference mode, answer is returned as an empty string, else
            - Sets EOS token as answer if question is impossible to answer, else
            - Appends answer with EOS token as the final answer
        Returns formatted answer string, answer tokens, and length of answer tokens
        """

        if self.mode == INFERENCE_MODE:
            target = ""
        elif example.is_impossible:  # example is impossible to answer given context
            target = self.tokenizer.tokenizer.eos_token
        else:
            target = f"{example.answer_text}{self.tokenizer.tokenizer.eos_token}"

        return self._get_truncated_sentence_and_len(target, self.max_answer_length)

    def _prep_context(
        self, example, query_tokens_length, answer_tokens_length, context_prefix_tokens, answer_prefix_tokens,
    ):
        """
        Calculates the maximum possible length for a given context given a question
            as inputs are fixed length
        Divides the context into multiple spans based on the calculated max length
        """

        context_tokens = self.tokenizer.tokenizer.tokenize(example.context_text)
        max_context_length = (
            self.max_seq_length
            - query_tokens_length
            - answer_tokens_length
            - len(context_prefix_tokens)
            - len(answer_prefix_tokens)
            - 1  # -1 accounts for EOS token
        )
        context_spans = GPTQADataset.get_docspans(context_tokens, max_context_length, self.doc_stride)
        context_spans = tuple(context_spans)

        return context_tokens, context_spans

    def _encode_all_context_spans(
        self,
        unique_id,
        context_spans,
        context_tokens,
        context_prefix,
        formatted_query,
        answer_prefix,
        formatted_answer,
        example,
        example_index,
    ):
        """
        Formats all spans extracted from a single context as:
            `context: <context span text> question: <question text> answer:<answer text>`
        <answer text> is set as:
            - blank if in inference mode, else
            - EOS token if answer text is not present in context span
                and the check flag is set to true, else
            - formatted answer
        """

        for context_span_idx, context_span in enumerate(context_spans):
            context_span_tokens = context_tokens[context_span.start : context_span.start + context_span.length]
            context_span_text = self.tokenizer.tokenizer.convert_tokens_to_string(context_span_tokens)

            input_without_answer = f"{context_prefix}{context_span_text}{formatted_query}{answer_prefix}"
            _, training_mask_end = self._get_truncated_sentence_and_len(input_without_answer, self.max_seq_length)

            is_answer_in_context_check = (
                self.check_if_answer_in_context  # checks if the flag for this check is set
                and example.answer_text  # checks if answer text is valid, i.e. question is not unanswerable
                and example.answer_text not in context_span_text  # checks if answer text is a substring of context
            )

            if self.mode == INFERENCE_MODE:
                input_to_encode = input_without_answer
            elif is_answer_in_context_check:
                input_to_encode = f"{input_without_answer}{self.tokenizer.tokenizer.eos_token}"
            else:
                input_to_encode = f"{input_without_answer}{formatted_answer}"

            encoded_input_dict = self.tokenizer.tokenizer(
                input_to_encode,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = torch.squeeze(encoded_input_dict["input_ids"])
            input_attn_mask = torch.squeeze(encoded_input_dict["attention_mask"])

            labels = GPTQADataset.update_labels_for_no_pad_loss(input_ids, training_mask_end, input_attn_mask)

            # create dictionary features
            feature = {
                "unique_id": unique_id,
                "input_ids": input_ids,
                "input_attn_mask": input_attn_mask,
                "training_mask_end": training_mask_end,
                "labels": labels,
                "example_index": example_index,
                "context_span_index": context_span_idx,
                "is_impossible": example.is_impossible,
            }

            self.features.append(feature)
            unique_id += 1

        return unique_id

    def _get_truncated_sentence_and_len(self, sentence, max_length):
        if not sentence:
            return "", 0
        tokens = self.tokenizer.tokenizer.tokenize(sentence)[:max_length]
        trunc_sentence = self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
        seq_length = len(tokens)

        return trunc_sentence, seq_length

    @classmethod
    def update_labels_for_no_pad_loss(cls, input_ids, training_mask_end, input_attn_mask):
        """
        Loss mask for GPT is constructed to ignore loss for padding tokens
        GPT eos token is same as pas token and needs to be excluded from loss mask
        This is done using the attention mask inversion as described in:
            https://github.com/huggingface/transformers/issues/7135#issuecomment-1172962080
        """
        labels = copy.copy(torch.squeeze(input_ids))
        inv_bool_attn_mask = torch.eq(torch.squeeze(input_attn_mask), 0)
        labels.data = torch.tensor(
            [
                -100 if ((i < training_mask_end) or (inv_bool_attn_mask[i])) else labels.data[i]
                for i in range(len(labels.data))
            ]
        )

        return labels

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        if self.mode == INFERENCE_MODE:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
                np.array(feature.training_mask_end),
            )
        else:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
                np.array(feature.training_mask_end),
                np.array(feature.labels),
            )
