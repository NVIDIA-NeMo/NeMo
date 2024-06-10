# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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
import os
import pickle
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch
from tqdm import trange

from nemo.collections.common.parts.utils import _compute_softmax
from nemo.collections.nlp.data.data_utils import is_whitespace
from nemo.collections.nlp.data.question_answering_squad.qa_squad_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
    SquadProcessor,
    _improve_answer_span,
    apply_no_ans_threshold,
    exact_match_score,
    f1_score,
    find_all_best_thresh,
    get_best_indexes,
    get_final_text,
    make_eval_dict,
    merge_eval,
    normalize_answer,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning

__all__ = ['SquadDataset', 'InputFeatures', '_check_is_max_context']


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id: int,
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        example_index: int = None,
        doc_span_index: int = None,
        tokens: List[str] = None,
        token_to_orig_map: Dict[int, int] = None,
        token_is_max_context: Dict[int, bool] = None,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        is_impossible: Optional[int] = None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.
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
    best_span_index = get_best_span_index(doc_spans, position)
    return cur_span_index == best_span_index


@lru_cache(maxsize=10000)
def get_best_span_index(doc_spans, position):
    """
    For a particular position, identify which doc_span gives the most context around token

    Helper function for _check_is_max_context; see _check_is_max_context for more details
    """
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
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


class SquadDataset(Dataset):
    """
    Creates SQuAD dataset for Question Answering.
    Args:
        data_file (str): train.*.json eval.*.json or test.*.json.
        tokenizer (obj): Tokenizer object, e.g. AutoTokenizer.
        version_2_with_negative (bool): True if training should allow
            unanswerable questions.
        doc_stride (int): When splitting up a long document into chunks,
            how much stride to take between chunks.
        max_query_length (iny): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_seq_length (int): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        num_samples: number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        mode (str): Use TRAINING_MODE/EVALUATION_MODE/INFERENCE_MODE to define between
            training, evaluation and inference dataset.
        use_cache (bool): Caches preprocessed data for future usage
    """

    def __init__(
        self,
        data_file: str,
        keep_doc_spans: str,
        tokenizer: object,
        doc_stride: int,
        max_query_length: int,
        max_seq_length: int,
        version_2_with_negative: bool,
        num_samples: int,
        mode: str,
        use_cache: bool,
    ):
        # deprecation warning
        deprecated_warning("SquadDataset")

        self.tokenizer = tokenizer
        self.version_2_with_negative = version_2_with_negative
        self.processor = SquadProcessor(data_file=data_file, mode=mode)
        self.mode = mode
        self.keep_doc_spans = keep_doc_spans

        # hashing to reduce memory use
        self.input_mask_id = 0
        self.input_mask_id_to_input_mask = {}
        self.input_mask_to_input_mask_id = {}

        self.segment_mask_id = 0
        self.segment_mask_id_to_segment_mask = {}
        self.segment_mask_to_segment_mask_id = {}

        if mode not in [TRAINING_MODE, EVALUATION_MODE, INFERENCE_MODE]:
            raise ValueError(
                f"mode should be either {TRAINING_MODE}, {EVALUATION_MODE}, {INFERENCE_MODE} but got {mode}"
            )
        self.examples = self.processor.get_examples()

        vocab_size = getattr(tokenizer, "vocab_size", 0)
        cached_features_file = (
            data_file
            + '_cache'
            + '_{}_{}_{}_{}_{}_{}_{}'.format(
                mode,
                tokenizer.name,
                str(vocab_size),
                str(max_seq_length),
                str(doc_stride),
                str(max_query_length),
                str(num_samples),
            )
        )

        # check number of samples. Should be either -1 not to limit or positive number
        if num_samples == 0:
            raise ValueError(
                f"num_samples has to be positive or -1 (to use the entire dataset), however got {num_samples}."
            )
        elif num_samples > 0:
            self.examples = self.examples[:num_samples]

        if use_cache and os.path.exists(cached_features_file):
            logging.info(f"loading from {cached_features_file}")
            # delete self.examples during training mode to save memory
            if self.mode == TRAINING_MODE:
                del self.examples
                del self.processor

            with open(cached_features_file, "rb") as reader:
                items_to_pickle = pickle.load(reader)
                (
                    self.features,
                    self.input_mask_id_to_input_mask,
                    self.input_mask_to_input_mask_id,
                    self.segment_mask_id_to_segment_mask,
                    self.segment_mask_to_segment_mask_id,
                ) = items_to_pickle
                items_to_pickle = None
                del items_to_pickle

        else:
            logging.info(f"Preprocessing data.")

            self.features = self.convert_examples_to_features(
                examples=self.examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                has_groundtruth=mode != INFERENCE_MODE,
            )

            if use_cache:
                master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                if master_device:
                    logging.info("  Saving train features into cached file %s", cached_features_file)
                    with open(cached_features_file, "wb") as writer:
                        items_to_pickle = [
                            self.features,
                            self.input_mask_id_to_input_mask,
                            self.input_mask_to_input_mask_id,
                            self.segment_mask_id_to_segment_mask,
                            self.segment_mask_to_segment_mask_id,
                        ]
                        pickle.dump(items_to_pickle, writer)

            # delete self.examples during training mode to save memory
            if self.mode == TRAINING_MODE:
                self.examples = []
                del self.processor

        logging.info("Converting dict features into object features")
        for i in trange(len(self.features)):
            self.features[i] = InputFeatures(**self.features[i])

    @staticmethod
    def get_doc_tokens_and_offset_from_context_id(
        context_id, start_position_character, is_impossible, answer_text, context_id_to_context_text
    ):
        start_position, end_position = 0, 0
        context_text = context_id_to_context_text[context_id]
        doc_tokens, char_to_word_offset = SquadDataset.split_into_words(context_text)

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            # start_position is index of word, end_position inclusive
            start_position = char_to_word_offset[start_position_character]
            end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

        return doc_tokens, char_to_word_offset, start_position, end_position, context_text

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

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Some features are obtained from hashmap to reduce CPU memory use"""
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
                    key=lambda doc_span: SquadDataset.get_average_dist_to_tok_start_and_end(
                        doc_span, tok_start_position, tok_end_position
                    )
                )
            return doc_spans[:n_candidates]
        else:
            raise ValueError('mode can only be in {all, only_positive and limited_negative')

    def convert_examples_to_features(
        self,
        examples: List[object],
        tokenizer: object,
        max_seq_length: int,
        doc_stride: int,
        max_query_length: int,
        has_groundtruth: bool,
    ):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        features = []
        text_to_tokens_dict = {}

        for example_index in trange(len(examples)):

            if example_index % 1000 == 0:
                SquadDataset.check_if_sufficient_memory()

            example = examples[example_index]
            if example.question_text not in text_to_tokens_dict:
                text_to_tokens_dict[example.question_text] = tokenizer.text_to_tokens(example.question_text)[
                    :max_query_length
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
            ) = SquadDataset.get_doc_tokens_and_offset_from_context_id(
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
            for i, token in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                if token not in text_to_tokens_dict:
                    text_to_tokens_dict[token] = tokenizer.text_to_tokens(token)
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

                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
                )

            # The -3 accounts for tokenizer.cls_token, tokenizer.sep_token and tokenizer.sep_token
            # doc_spans contains all possible contexts options of given length
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            doc_spans = SquadDataset.get_docspans(all_doc_tokens, max_tokens_for_doc, doc_stride)

            doc_spans = SquadDataset.keep_relevant_docspans(
                doc_spans, tok_start_position, tok_end_position, self.keep_doc_spans
            )

            # make compatible for hashing
            doc_spans = tuple(doc_spans)

            for doc_span_index, doc_span in enumerate(doc_spans):

                tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token]
                segment_ids = [0 for i in range(len(tokens))]

                token_is_max_context = {}
                # maps context tokens idx in final input -> word idx in context
                token_to_orig_map = {}

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append(tokenizer.sep_token)
                segment_ids.append(1)

                input_ids = tokenizer.tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(tokenizer.pad_id)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

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
                # end memoization

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

                features.append(input_feature)
                unique_id += 1
        return features

    def get_predictions(
        self,
        unique_ids: List[int],
        start_logits: List[List[float]],
        end_logits: List[List[float]],
        n_best_size: int,
        max_answer_length: int,
        do_lower_case: bool,
        version_2_with_negative: bool,
        null_score_diff_threshold: float,
    ):
        example_index_to_features = collections.defaultdict(list)

        unique_id_to_pos = {}
        for index, unique_id in enumerate(unique_ids):
            unique_id_to_pos[unique_id] = index

        for feature in self.features:
            example_index_to_features[feature.example_index].append(feature)

        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()
        for example_index, example in enumerate(self.examples):

            # finish this loop if we went through all batch examples
            if example_index >= len(unique_ids):
                break

            features = example_index_to_features[example_index]

            doc_tokens, _, _, _, _ = SquadDataset.get_doc_tokens_and_offset_from_context_id(
                example.context_id,
                example.start_position_character,
                example.is_impossible,
                example.answer_text,
                self.processor.doc_id_to_context_text,
            )
            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            # large and positive
            score_null = 1000000
            # the paragraph slice with min null score
            min_null_feature_index = 0
            # start logit at the slice with min null score
            null_start_logit = 0
            # end logit at the slice with min null score
            null_end_logit = 0
            for feature_index, feature in enumerate(features):
                pos = unique_id_to_pos[feature.unique_id]
                start_indexes = get_best_indexes(start_logits[pos], n_best_size)
                end_indexes = get_best_indexes(end_logits[pos], n_best_size)
                # if we could have irrelevant answers,
                # get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = start_logits[pos][0] + end_logits[pos][0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = start_logits[pos][0]
                        null_end_logit = end_logits[pos][0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions,
                        # e.g., predict that the start of the span is in the
                        # question. We throw out all invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=start_logits[pos][start_index],
                                end_logit=end_logits[pos][end_index],
                            )
                        )

            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

            _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

                # In very rare edge cases we could only
                # have single null pred. We just create a nonce prediction
                # in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for i, entry in enumerate(nbest):
                output = collections.OrderedDict()
                output["question"] = example.question_text
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = (
                    entry.start_logit
                    if (isinstance(entry.start_logit, float) or isinstance(entry.start_logit, int))
                    else list(entry.start_logit)
                )
                output["end_logit"] = (
                    entry.end_logit
                    if (isinstance(entry.end_logit, float) or isinstance(entry.end_logit, int))
                    else list(entry.end_logit)
                )
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score -
                # the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

        return all_predictions, all_nbest_json, scores_diff_json

    def evaluate_predictions(
        self,
        all_predictions: Dict[str, str],
        no_answer_probs: Optional[float] = None,
        no_answer_probability_threshold: float = 1.0,
    ):
        qas_id_to_has_answer = {
            example.qas_id: bool(example.answers) for example in self.examples[: len(all_predictions)]
        }
        has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
        no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
        if no_answer_probs is None:
            no_answer_probs = {k: 0.0 for k in all_predictions}

        exact, f1 = self.get_raw_scores(all_predictions)

        exact_threshold = apply_no_ans_threshold(
            exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
        )
        f1_threshold = apply_no_ans_threshold(
            f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
        )

        evaluation = make_eval_dict(exact_threshold, f1_threshold)

        if has_answer_qids:
            has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
            merge_eval(evaluation, has_ans_eval, "HasAns")

        if no_answer_qids:
            no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
            merge_eval(evaluation, no_ans_eval, "NoAns")

        if no_answer_probs:
            find_all_best_thresh(evaluation, all_predictions, exact, f1, no_answer_probs, qas_id_to_has_answer)

        return evaluation["best_exact"], evaluation["best_f1"]

    def get_raw_scores(self, preds: Dict[str, str]):
        """
        Computes the exact and f1 scores from the examples
        and the model predictions
        """
        exact_scores = {}
        f1_scores = {}

        for example in self.examples:
            qas_id = example.qas_id
            gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

            if not gold_answers:
                # For unanswerable questions,
                # only correct answer is empty string
                gold_answers = [""]

            if qas_id not in preds:
                logging.warning("Missing prediction for %s" % qas_id)
                continue

            prediction = preds[qas_id]
            exact_scores[qas_id] = max(exact_match_score(a, prediction) for a in gold_answers)
            f1_scores[qas_id] = max(f1_score(a, prediction) for a in gold_answers)

        return exact_scores, f1_scores

    def evaluate(
        self,
        unique_ids: List[str],
        start_logits: List[List[float]],
        end_logits: List[List[float]],
        n_best_size: int,
        max_answer_length: int,
        do_lower_case: bool,
        version_2_with_negative: bool,
        null_score_diff_threshold: float,
    ):
        (all_predictions, all_nbest_json, scores_diff_json) = self.get_predictions(
            unique_ids,
            start_logits,
            end_logits,
            n_best_size,
            max_answer_length,
            do_lower_case,
            version_2_with_negative,
            null_score_diff_threshold,
        )

        exact_match, f1 = self.evaluate_predictions(all_predictions)

        return exact_match, f1, all_predictions, all_nbest_json
