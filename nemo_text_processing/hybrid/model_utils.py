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

import math
import re
from typing import List, Union

import torch
from tqdm import tqdm

from nemo.collections.common.parts import MLMScorer
from nemo.utils import logging


def init_models(model_name_list: str):
    """
    returns dictionary of Masked Language Models by their HuggingFace name.
    """
    model_names = model_name_list.split(",")
    models = {}
    for model_name in model_names:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models[model_name] = MLMScorer(model_name=model_name, device=device)
    return models


def get_score(texts: Union[List[str], str], model: MLMScorer):
    """Computes MLM score for list of text using model"""
    try:
        if isinstance(texts, str):
            texts = [texts]
        score = -1 * sum(model.score_sentences(texts)) / len(texts)
    except Exception as e:
        print(e)
        print(f"Scoring error: {texts}")
        score = math.inf
    return score


def get_masked_score(text, model, do_lower=True):
    """text is normalized prediction which contains <> around semiotic tokens.
    If multiple tokens are present, multiple variants of the text are created where all but one ambiguous semiotic tokens are masked
    to avoid unwanted reinforcement of neighboring semiotic tokens."""
    text = text.lower() if do_lower else text
    spans = re.findall("<\s.+?\s>", text)
    if len(spans) > 0:
        text_with_mask = []

        for match in re.finditer("<\s.+?\s>", text):
            new_text = (
                text[: match.span()[0]] + match.group().replace("< ", "").replace(" >", "") + text[match.span()[1] :]
            )
            new_text = re.sub("<\s.+?\s>", model.MASK_LABEL, new_text)
            text_with_mask.append(new_text)
        text = text_with_mask

    return get_score(text, model)


def _get_ambiguous_positions(sentences: List[str]):
    """returns None or index list of ambigous semiotic tokens for list of sentences.
    E.g. if sentences = ["< street > < three > A", "< saint > < three > A"], it returns [1, 0] since only 
    the first semiotic span <street>/<saint> is ambiguous."""
    l_sets = [set([x]) for x in re.findall("<\s.+?\s>", sentences[0])]
    for sentence in sentences[1:]:
        spans = re.findall("<\s.+?\s>", sentence)
        if len(spans) != len(l_sets):
            return None
        for i in range(len(spans)):
            l_sets[i].add(spans[i])

    ambiguous = []
    for span in l_sets:
        ambiguous.append(len(span) > 1)
    return ambiguous


def score_options(sentences: List[str], context_len, model, do_lower=True):
    """return list of scores for each sentence in list where model is used for MLM Scoring."""
    scores = []
    if context_len is not None:
        diffs = [find_diff(s, context_len) for s in sentences]
        if len(set([len(d) for d in diffs])) == 1:
            sentences = diffs

    ambiguous_positions = None
    if sentences and isinstance(sentences[0], str):
        ambiguous_positions = _get_ambiguous_positions(sentences)

    for sent in tqdm(sentences):
        if isinstance(sent, list):  # in case of set context len
            option_scores = [get_masked_score(s, model, do_lower) for s in sent]
            logging.debug(sent)
            logging.debug(option_scores)
            logging.debug("=" * 50)
            if any(math.isnan(x) for x in option_scores):
                av_score = math.inf
            else:
                av_score = round(sum(option_scores) / len(option_scores), 4)
            scores.append(av_score)
        elif isinstance(sent, str):  # in case of full context
            if ambiguous_positions:
                matches = list(re.finditer("<\s.+?\s>", sent))
                for match, pos in zip(matches[::-1], ambiguous_positions[::-1]):
                    if not pos:
                        sent = (
                            sent[: match.span()[0]]
                            + match.group().replace("< ", "").replace(" >", "")
                            + sent[match.span()[1] :]
                        )
            scores.append(round(get_masked_score(sent, model, do_lower=do_lower)))
        else:
            raise ValueError()
    return scores


def find_diff(text, context_len=3):
    """Finds parts of text normalized by WFST and returns them in list with a context of context_len"""
    diffs = []
    pattern_start = "< "
    pattern_end = " >"

    def __clean(s):
        return s.replace(pattern_start, "").replace(pattern_end, "").replace("  ", " ")

    index_start = 0
    while pattern_start in text[index_start:]:
        index_start = index_start + text[index_start:].index(pattern_start)
        offset = index_start
        if pattern_end in text[offset:]:
            index_end = offset + text[offset:].index(pattern_end) + len(pattern_end)
            center = __clean(text[index_start:index_end])

            left_context = " ".join(__clean(text[:index_start]).split()[-context_len:])
            if len(left_context) > 0 and text[:index_start][-1].isspace():
                left_context = left_context + " "
            right_context = " ".join(__clean(text[index_end:]).split()[:context_len])
            if len(right_context) > 0 and text[index_end][0].isspace():
                right_context = " " + right_context
            diffs.append(left_context + center + right_context)
            index_end += 1
            index_start = index_end + 1
        else:
            break
    if len(diffs) == 0:
        diffs = [text]
    return diffs
