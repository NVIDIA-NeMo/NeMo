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
from typing import List

import ijson
import numpy as np
from transformers.models.bert.tokenization_bert import BasicTokenizer

from nemo.collections.nlp.data.data_utils import (
    DataProcessor,
    check_chinese_char,
    normalize_answer,
    normalize_chinese_answer,
)
from nemo.utils import logging

"""
Utility functions for Question Answering NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers
"""

TRAINING_MODE = "train"
EVALUATION_MODE = "eval"
INFERENCE_MODE = "infer"


def _get_tokens(s):
    """get normalized tokens for both Chinese and English"""
    if not s:
        return []

    # separate answers to en and ch pieces
    ch_seq = ""
    en_seq = ""
    pos = 0

    # Normalize and connect
    final_tokens = []

    while pos < len(s):
        if check_chinese_char(s[pos]):
            if en_seq != "":
                final_tokens.extend(normalize_answer(en_seq).split())
                en_seq = ""
            ch_seq += s[pos]
        else:
            if ch_seq != "":
                final_tokens.extend(normalize_chinese_answer(ch_seq))
                ch_seq = ""
            en_seq += s[pos]
        pos += 1

    if en_seq != "":
        final_tokens.extend(normalize_answer(en_seq).split())

    if ch_seq != "":
        final_tokens.extend(normalize_chinese_answer(ch_seq))

    return final_tokens


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    best_indices = np.argsort(logits)[::-1]
    return best_indices[:n_best_size]


def get_final_text(pred_text: str, orig_text: str, do_lower_case: bool, verbose_logging: bool = False):
    """Project the tokenized prediction back to the original text.
    When we created the data, we kept track of the alignment between original
    (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    now `orig_text` contains the span of our original text corresponding to
    the span that we predicted.

    However, `orig_text` may contain extra characters that we don't want in
    our prediction.

    For example, let's say:
      pred_text = steve smith
      orig_text = Steve Smith's

    We don't want to return `orig_text` because it contains the extra "'s".

    We don't want to return `pred_text` because it's already been normalized
    (the SQuAD eval script also does punctuation stripping/lower casing but
    our tokenizer does additional normalization like stripping accent
    characters).

    What we really want to return is "Steve Smith".

    Therefore, we have to apply a semi-complicated alignment heuristic
    between `pred_text` and `orig_text` to get a character-to-character
    alignment. This can fail in certain cases in which case we just return
    `orig_text`."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logging.warning("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logging.warning(
                "Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logging.warning("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logging.warning("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def f1_score(prediction, ground_truth):
    """computes f1 score between prediction and ground truth"""
    prediction_tokens = _get_tokens(prediction)
    ground_truth_tokens = _get_tokens(ground_truth)
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(ground_truth_tokens == prediction_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """computes exact match between prediction and ground truth"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    """Applies no answer threshhold"""
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    """returns dictionary with formatted evaluation scores"""
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    """Merges 2 evaluation dictionaries into the first one by adding prefix as key for name collision handling"""
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    """
    Find best threshholds to maximize all evaluation metrics.
    """
    best_exact, exact_thresh = _find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = _find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def _find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """
    Find best threshhold to maximize evaluation metric
    """
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def _improve_answer_span(
    doc_tokens: List[str], input_start: int, input_end: int, tokenizer: object, orig_answer_text: str
):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.text_to_tokens(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    used by the version 1.1 and version 2.0 of SQuAD, respectively.

    Args:
        data_file: data file path
        mode: TRAINING_MODE/EVALUATION_MODE/INFERENCE_MODE for creating training/evaluation/inference dataset
    """

    def __init__(self, data_file: str, mode: str):
        self.data_file = data_file
        self.mode = mode
        # Memoizes documents to reduce memory use (as the same document is often used for many questions)
        self.doc_id = 0
        self.context_text_to_doc_id = {}
        self.doc_id_to_context_text = {}

    def get_examples(self):
        """
        Get examples from raw json file
        """
        if self.data_file is None:
            raise ValueError(f"{self.mode} data file is None.")

        # remove this line and the replace cache line below - which is a temp fix
        with open(self.data_file.replace('_cache', ''), "r", encoding="utf-8") as reader:
            input_data = ijson.items(reader, "data.item")

            examples = []
            for entry in input_data:
                len_docs = []
                title = entry["title"]
                for paragraph in entry["paragraphs"]:
                    context_text = paragraph["context"]
                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        if not question_text:
                            continue
                        start_position_character = None
                        answer_text = None
                        answers = []
                        if "is_impossible" in qa:
                            is_impossible = qa["is_impossible"] or len(qa["answers"]) < 1
                        else:
                            is_impossible = False

                        if not is_impossible:
                            if self.mode in [TRAINING_MODE, EVALUATION_MODE]:
                                answer = qa["answers"][0]
                                answer_text = answer["text"]
                                start_position_character = answer["answer_start"]
                            if self.mode == EVALUATION_MODE:
                                answers = qa["answers"]
                        if context_text in self.context_text_to_doc_id:
                            doc_id = self.context_text_to_doc_id[context_text]
                        else:
                            doc_id = self.doc_id
                            self.context_text_to_doc_id[context_text] = doc_id
                            self.doc_id_to_context_text[doc_id] = context_text
                            self.doc_id += 1
                            len_docs.append(len(context_text))

                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            context_text=context_text,
                            context_id=doc_id,
                            answer_text=answer_text,
                            start_position_character=start_position_character,
                            title=title,
                            is_impossible=is_impossible,
                            answers=answers,
                        )

                        examples.append(example)

                logging.info('mean no. of chars in doc: {}'.format(np.mean(len_docs)))
                logging.info('max no. of chars in doc: {}'.format(np.max(len_docs)))
        return examples


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        context_id: id representing context string
        answer_text: The answer string
        start_position_character: The character position of the start of
            the answer, 0 indexed
        title: The title of the example
        answers: None by default, this is used during evaluation.
            Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has
            no possible answer.
    """

    def __init__(
        self,
        qas_id: str,
        question_text: str,
        context_text: str,
        context_id: int,
        answer_text: str,
        start_position_character: int,
        title: str,
        answers: List[str] = [],
        is_impossible: bool = False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_id = context_id
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position_character = start_position_character
