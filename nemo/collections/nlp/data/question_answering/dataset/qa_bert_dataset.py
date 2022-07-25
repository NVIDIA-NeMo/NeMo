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
from tqdm import trange

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import INFERENCE_MODE, TRAINING_MODE
from nemo.collections.nlp.data.question_answering.dataset.qa_dataset import QADataset
from nemo.collections.nlp.data.question_answering.input_example.qa_bert_input_example import BERTQAInputExample
from nemo.utils import logging


class BERTQADataset(QADataset):
    """ Creates a Dataset for BERT architecture based Exractive QA """

    def __init__(
        self,
        data_file: str,
        processor: object,
        tokenizer: object,
        keep_doc_spans: str = False,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_seq_length: int = 512,
        version_2_with_negative: bool = False,
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
        self.version_2_with_negative = version_2_with_negative
        self.num_samples = num_samples
        self.mode = mode
        self.use_cache = use_cache

        # structures for hashing to reduce memory use
        self.input_mask_id = 0
        self.input_mask_id_to_input_mask = {}
        self.input_mask_to_input_mask_id = {}

        self.segment_mask_id = 0
        self.segment_mask_id_to_segment_mask = {}
        self.segment_mask_to_segment_mask_id = {}

        self._set_cached_features_filename()
        if use_cache and os.path.exists(self.cached_features_file):
            if self.mode == TRAINING_MODE:
                del self.examples
                del self.processor
            (
                self.features,
                self.input_mask_id_to_input_mask,
                self.input_mask_to_input_mask_id,
                self.segment_mask_id_to_segment_mask,
                self.segment_mask_to_segment_mask_id,
            ) = QADataset.load_features_from_cache(self.cached_features_file)
        else:
            self._convert_examples_to_features()
            if use_cache:
                items_to_pickle = [
                    self.features,
                    self.input_mask_id_to_input_mask,
                    self.input_mask_to_input_mask_id,
                    self.segment_mask_id_to_segment_mask,
                    self.segment_mask_to_segment_mask_id,
                ]
                QADataset.dump_features_to_cache(self.cached_features_file, items_to_pickle)

        logging.info("Converting dict features into object features")
        for i in trange(len(self.features)):
            self.features[i] = BERTQAInputExample(**self.features[i])

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
                str(self.max_seq_length),
                str(self.doc_stride),
                str(self.max_query_length),
                str(self.num_samples),
            )
        )

    def _convert_examples_to_features(self):
        """ Converts loaded examples to features """

        logging.info(f"Preprocessing data into features.")

        has_groundtruth = self.mode != INFERENCE_MODE
        unique_id = 1000000000
        text_to_tokens_dict = {}
        self.features = []

        for example_index in trange(len(self.examples)):

            if example_index % 1000 == 0:
                QADataset.check_if_sufficient_memory()

            example = self.examples[example_index]
            if example.question_text not in text_to_tokens_dict:
                text_to_tokens_dict[example.question_text] = self.tokenizer.text_to_tokens(example.question_text)[
                    : self.max_query_length
                ]
            query_tokens = text_to_tokens_dict[example.question_text]

            # context: index of token -> index of word
            tok_to_orig_index = []

            # context: index of word -> index of first token in token list
            orig_to_tok_index = []

            # context without white spaces after tokenization
            all_doc_tokens = []

            # doc tokens is word separated context
            (
                doc_tokens,
                char_to_word_offset,
                start_position,
                end_position,
                context_text,
            ) = QADataset.get_doc_tokens_and_offset_from_context_id(
                example.context_id,
                example.start_position_character,
                example.is_impossible,
                example.answer_text,
                self.processor.doc_id_to_context_text,
            )

            example.start_position = start_position
            example.end_position = end_position
            if self.mode != TRAINING_MODE:
                example.doc_tokens = doc_tokens

            # the text to tokens step is the slowest step
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                if token not in text_to_tokens_dict:
                    text_to_tokens_dict[token] = self.tokenizer.text_to_tokens(token)
                sub_tokens = text_to_tokens_dict[token]

                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # idx of query token start and end in context
            tok_start_position = None
            tok_end_position = None
            if has_groundtruth and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if has_groundtruth and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

                (tok_start_position, tok_end_position) = QADataset.improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, self.tokenizer, example.answer_text
                )

            # The -3 accounts for tokenizer.cls_token, tokenizer.sep_token and tokenizer.sep_token
            # doc_spans contains all possible contexts options of given length
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
            doc_spans = QADataset.get_docspans(all_doc_tokens, max_tokens_for_doc, self.doc_stride)
            doc_spans = QADataset.keep_relevant_docspans(
                doc_spans, tok_start_position, tok_end_position, self.keep_doc_spans
            )

            # make compatible for hashing
            doc_spans = tuple(doc_spans)

            for (doc_span_index, doc_span) in enumerate(doc_spans):

                tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token]
                segment_ids = [0 for i in range(len(tokens))]

                token_is_max_context = {}

                # maps context tokens idx in final input -> word idx in context
                token_to_orig_map = {}

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    is_max_context = QADataset.check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append(self.tokenizer.sep_token)
                segment_ids.append(1)

                input_ids = self.tokenizer.tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(self.tokenizer.pad_id)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                # calculate start and end position in final array
                # of tokens in answer if no answer,
                # 0 for both pointing to tokenizer.cls_token
                start_position = 0
                end_position = 0
                if has_groundtruth and not example.is_impossible:
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                if has_groundtruth and example.is_impossible:
                    # if our document chunk does not contain
                    # an annotation we throw it out, since there is nothing
                    # to predict.
                    start_position = 0
                    end_position = 0

                if example_index < 1:
                    logging.info("*** Example ***")
                    logging.info("unique_id: %s" % (unique_id))
                    logging.info("example_index: %s" % (example_index))
                    logging.info("doc_span_index: %s" % (doc_span_index))
                    logging.info("tokens: %s" % " ".join(tokens))
                    logging.info(
                        "token_to_orig_map: %s" % " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()])
                    )
                    logging.info(
                        "token_is_max_context: %s"
                        % " ".join(["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()])
                    )
                    logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if has_groundtruth and example.is_impossible:
                        logging.info("impossible example")
                    if has_groundtruth and not example.is_impossible:
                        answer_text = " ".join(tokens[start_position : (end_position + 1)])
                        logging.info("start_position: %d" % (start_position))
                        logging.info("end_position: %d" % (end_position))
                        logging.info("answer: %s" % (answer_text))

                # memoization to save CPU memory for large datasets
                input_mask = tuple(input_mask)
                if input_mask in self.input_mask_to_input_mask_id:
                    feature_input_mask_id = self.input_mask_to_input_mask_id[input_mask]
                else:
                    self.input_mask_id_to_input_mask[self.input_mask_id] = input_mask
                    self.input_mask_to_input_mask_id[input_mask] = self.input_mask_id
                    feature_input_mask_id = self.input_mask_id
                    self.input_mask_id += 1

                segment_mask = tuple(segment_ids)
                if segment_mask in self.segment_mask_to_segment_mask_id:
                    feature_segment_mask_id = self.segment_mask_to_segment_mask_id[segment_mask]
                else:
                    self.segment_mask_id_to_segment_mask[self.segment_mask_id] = segment_mask
                    self.segment_mask_to_segment_mask_id[segment_mask] = self.segment_mask_id
                    feature_segment_mask_id = self.segment_mask_id
                    self.segment_mask_id += 1

                if self.mode == TRAINING_MODE:
                    input_feature = {
                        "unique_id": unique_id,
                        "input_ids": input_ids,
                        "input_mask": feature_input_mask_id,
                        "segment_ids": feature_segment_mask_id,
                        "start_position": start_position,
                        "end_position": end_position,
                    }
                else:
                    input_feature = {
                        "unique_id": unique_id,
                        "input_ids": input_ids,
                        "input_mask": feature_input_mask_id,
                        "segment_ids": feature_segment_mask_id,
                        "start_position": start_position,
                        "end_position": end_position,
                        "example_index": example_index,
                        "doc_span_index": doc_span_index,
                        "tokens": tokens,
                        "token_to_orig_map": token_to_orig_map,
                        "token_is_max_context": token_is_max_context,
                        "is_impossible": example.is_impossible,
                    }

                self.features.append(input_feature)
                unique_id += 1

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            self.examples = []
            del self.processor

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        if self.mode == INFERENCE_MODE:
            return (
                np.array(feature.input_ids),
                np.array(self.segment_mask_id_to_segment_mask[feature.segment_ids]),
                np.array(self.input_mask_id_to_input_mask[feature.input_mask]),
                np.array(feature.unique_id),
            )
        else:
            return (
                np.array(feature.input_ids),
                np.array(self.segment_mask_id_to_segment_mask[feature.segment_ids]),
                np.array(self.input_mask_id_to_input_mask[feature.input_mask]),
                np.array(feature.unique_id),
                np.array(feature.start_position),
                np.array(feature.end_position),
            )
