# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import collections
import pickle
from functools import lru_cache
from typing import List

import psutil
import torch

from nemo.collections.nlp.data.data_utils import is_whitespace
from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
)
from nemo.core.classes import Dataset
from nemo.utils import logging


class QADataset(Dataset):
    ''' Abstract base class for QA Datasets with common utility methods '''

    def __init__(
        self, data_file: str, processor: object, tokenizer: object, mode: str, num_samples: int, **kwargs,
    ):
        self.mode = mode
        self.data_file = data_file
        self.processor = processor
        self.tokenizer = tokenizer
        self.features = None

        if self.mode not in [TRAINING_MODE, EVALUATION_MODE, INFERENCE_MODE]:
            raise ValueError(
                f"mode should be either {TRAINING_MODE}, {EVALUATION_MODE}, {INFERENCE_MODE} but got {self.mode}"
            )

        # get examples from processor and keep according to limit
        self.examples = self.processor.get_examples()
        if num_samples == 0:
            raise ValueError(
                f"num_samples has to be positive or -1 (to use the entire dataset), however got {num_samples}."
            )
        elif num_samples > 0:
            self.examples = self.examples[:num_samples]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    @staticmethod
    def load_features_from_cache(cached_filename):
        logging.info(f"loading from {cached_filename}")
        with open(cached_filename, "rb") as reader:
            features = pickle.load(reader)

        return features

    @staticmethod
    def dump_features_to_cache(cached_filename, features):
        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if master_device:
            logging.info(f"Saving train features into cached file {cached_filename}")
            with open(cached_filename, "wb") as writer:
                pickle.dump(features, writer)

    @staticmethod
    def check_if_sufficient_memory():
        """
        Check if there is sufficient memory to prevent system from being unresponsive
        Otherwise system can become unresponsive as memory is slowly filled up, possibly leading to system unable to kill process
        Interrupts run if CPU memory use is more than 75%, to leave some capacity for model loading
        """

        percent_memory = psutil.virtual_memory().percent
        if percent_memory > 75:
            raise ValueError('Please use a device with more CPU ram or a smaller dataset')

    @staticmethod
    @lru_cache(maxsize=10000)
    def get_best_span_index(doc_spans, position):
        """
        For a particular position, identify which doc_span gives the most context around token
        Helper function for check_is_max_context; see check_is_max_context for more details
        """

        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return best_span_index

    @staticmethod
    def check_is_max_context(doc_spans, cur_span_index, position):
        """
        Check if this is the 'max context' doc span for the token.
        Because of the sliding window approach taken to scoring documents,
        a single token can appear in multiple documents.
        Example:
            Doc: the man went to the store and bought a gallon of milk
            Span A: the man went to the
            Span B: to the store and bought
            Span C: and bought a gallon of
            ...
        Now the word 'bought' will have two scores from spans B and C. We only
        want to consider the score with "maximum context", which we define as
        the *minimum* of its left and right context (the *sum* of left and
        right context will always be the same, of course).
        In the example the maximum context for 'bought' would be span C since
        it has 1 left context and 3 right context, while span B has 4 left context
        and 0 right context.
        Code adapted from the code by the Google AI and HuggingFace.
        """

        best_span_index = QADataset.get_best_span_index(doc_spans, position)

        return cur_span_index == best_span_index

    @staticmethod
    def get_docspans(all_doc_tokens, max_tokens_for_doc, doc_stride):
        """
        Get docspans which are sliding window spans from a document

        Args:
            all_doc_tokens: list of all tokens in document
            max_tokens_for_doc: maximum number of tokens in each doc span
            doc_stride: stride size which sliding window moves with
        
        Returns:
            doc_spans: all possible doc_spans from document
        """

        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        return doc_spans

    @staticmethod
    def get_average_dist_to_tok_start_and_end(doc_span, tok_start_position, tok_end_position):
        """
        Find distance between doc_span and answer_span to determine if doc_span is likely to be useful for the answer
        Helper function to filter out doc_spans that may not be helpful

        Args:
            doc_span
            tok_start_position: start position of answer in document
            tok_end_position: end position of answer in document
        
        Returns:
            average distance of doc_span to answer
        """

        center_answer = (tok_start_position + tok_end_position) // 2
        dist_to_start = abs(doc_span.start - center_answer)
        dist_to_end = abs(doc_span.start + doc_span.length - 1 - center_answer)

        return (dist_to_start + dist_to_end) // 2

    @staticmethod
    def keep_relevant_docspans(doc_spans, tok_start_position, tok_end_position, mode):
        """
        Filters out doc_spans, which might not be relevant to answering question, 
        which can be helpful when document is extremely long leading to many doc_spans with no answers

        Args:
            doc_spans: all possible doc_spans
            tok_start_position: start position of answer in document
            tok_end_position: end position of answer in document
            mode:
                all: do not filter
                only_positive: only keep doc_spans containing the answer
                limited_negative: only keep 10 doc_spans that are nearest to answer
        
        Returns:
            doc_spans: doc_spans after filtering
        """

        if mode == 'all':
            return doc_spans
        elif mode == 'only_positive':
            if tok_start_position in [-1, None] or tok_end_position in [-1, None]:
                return []
            else:
                return [
                    doc_span
                    for doc_span in doc_spans
                    if tok_start_position >= doc_span.start
                    and tok_end_position <= doc_span.start + doc_span.length - 1
                ]
        elif mode == 'limited_negative':
            n_candidates = 10
            if tok_start_position in [-1, None] or tok_end_position in [-1, None]:
                pass
            else:
                doc_spans.sort(
                    key=lambda doc_span: QADataset.get_average_dist_to_tok_start_and_end(
                        doc_span, tok_start_position, tok_end_position
                    )
                )
            return doc_spans[:n_candidates]
        else:
            raise ValueError('mode can only be in {all, only_positive and limited_negative')

    @staticmethod
    def split_into_words(context_text):
        """
        Split on whitespace so that different tokens
        may be attributed to their original position.
        ex: context_text = "hi yo"
            char_to_word_offset = [0, 0, 0, 1, 1]
            doc_tokens = ["hi", "yo"]
        """

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        return doc_tokens, char_to_word_offset

    @staticmethod
    def get_doc_tokens_and_offset_from_context_id(
        context_id, start_position_character, is_impossible, answer_text, context_id_to_context_text
    ):
        start_position, end_position = 0, 0
        context_text = context_id_to_context_text[context_id]
        doc_tokens, char_to_word_offset = QADataset.split_into_words(context_text)

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:

            # start_position is index of word, end_position inclusive
            start_position = char_to_word_offset[start_position_character]
            end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

        return doc_tokens, char_to_word_offset, start_position, end_position, context_text

    @staticmethod
    def improve_answer_span(
        doc_tokens: List[str], input_start: int, input_end: int, tokenizer: object, orig_answer_text: str,
    ):
        """ Returns tokenized answer spans that better match the annotated answer """

        tok_answer_text = " ".join(tokenizer.text_to_tokens(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)
