# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import string
from copy import deepcopy

import regex as re
from tqdm import tqdm

from nemo.collections.nlp.data.text_normalization import constants

__all__ = [
    'read_data_file',
    'normalize_str',
    'flatten',
    'convert_fraction',
    'convert_superscript',
    'add_space_around_dash',
]


def flatten(l):
    """ flatten a list of lists """
    return [item for sublist in l for item in sublist]


def add_space_around_dash(input: str):
    """ adds space around dash between numbers and non-numbers"""
    input = re.sub(r"([^\s0-9])-([0-9])", r"\1 - \2", input)
    input = re.sub(r"([0-9])-([^\s0-9])", r"\1 - \2", input)
    input = re.sub(r"([^\s0-9])-([0-9])", r"\1 - \2", input)
    input = re.sub(r"([0-9])-([^\s0-9])", r"\1 - \2", input)
    return input


def convert_superscript(written: str):
    """convert superscript to regular character"""
    written = re.sub("²", "2", written)
    written = re.sub("³", "3", written)
    return written


def convert_fraction(written: str):
    """
    converts fraction to standard form, e.g "½" -> "1/2", "1 ½" -> "1 1/2"

    Args:
        written: written form
    Returns:
        written: modified form
    """
    written = re.sub(" ½", " 1/2", written)
    written = re.sub(" ⅓", " 1/3", written)
    written = re.sub(" ⅔", " 2/3", written)
    written = re.sub(" ¼", " 1/4", written)
    written = re.sub(" ¾", " 3/4", written)
    written = re.sub(" ⅕", " 1/5", written)
    written = re.sub(" ⅖", " 2/5", written)
    written = re.sub(" ⅗", " 3/5", written)
    written = re.sub(" ⅘", " 4/5", written)
    written = re.sub(" ⅙", " 1/6", written)
    written = re.sub(" ⅚", " 5/6", written)
    written = re.sub(" ⅛", " 1/8", written)
    written = re.sub(" ⅜", " 3/8", written)
    written = re.sub(" ⅝", " 5/8", written)
    written = re.sub(" ⅞", " 7/8", written)
    written = re.sub("^½", "1/2", written)
    written = re.sub("^⅓", "1/3", written)
    written = re.sub("^⅔", "2/3", written)
    written = re.sub("^¼", "1/4", written)
    written = re.sub("^¾", "3/4", written)
    written = re.sub("^⅕", "1/5", written)
    written = re.sub("^⅖", "2/5", written)
    written = re.sub("^⅗", "3/5", written)
    written = re.sub("^⅘", "4/5", written)
    written = re.sub("^⅙", "1/6", written)
    written = re.sub("^⅚", "5/6", written)
    written = re.sub("^⅛", "1/8", written)
    written = re.sub("^⅜", "3/8", written)
    written = re.sub("^⅝", "5/8", written)
    written = re.sub("^⅞", "7/8", written)
    written = re.sub("-½", "-1/2", written)
    written = re.sub("-⅓", "-1/3", written)
    written = re.sub("-⅔", "-2/3", written)
    written = re.sub("-¼", "-1/4", written)
    written = re.sub("-¾", "-3/4", written)
    written = re.sub("-⅕", "-1/5", written)
    written = re.sub("-⅖", "-2/5", written)
    written = re.sub("-⅗", "-3/5", written)
    written = re.sub("-⅘", "-4/5", written)
    written = re.sub("-⅙", "-1/6", written)
    written = re.sub("-⅚", "-5/6", written)
    written = re.sub("-⅛", "-1/8", written)
    written = re.sub("-⅜", "-3/8", written)
    written = re.sub("-⅝", "-5/8", written)
    written = re.sub("-⅞", "-7/8", written)
    written = re.sub("([0-9])\s?½", "\\1 1/2", written)
    written = re.sub("([0-9])\s?⅓", "\\1 1/3", written)
    written = re.sub("([0-9])\s?⅔", "\\1 2/3", written)
    written = re.sub("([0-9])\s?¼", "\\1 1/4", written)
    written = re.sub("([0-9])\s?¾", "\\1 3/4", written)
    written = re.sub("([0-9])\s?⅕", "\\1 1/5", written)
    written = re.sub("([0-9])\s?⅖", "\\1 2/5", written)
    written = re.sub("([0-9])\s?⅗", "\\1 3/5", written)
    written = re.sub("([0-9])\s?⅘", "\\1 4/5", written)
    written = re.sub("([0-9])\s?⅙", "\\1 1/6", written)
    written = re.sub("([0-9])\s?⅚", "\\1 5/6", written)
    written = re.sub("([0-9])\s?⅛", "\\1 1/8", written)
    written = re.sub("([0-9])\s?⅜", "\\1 3/8", written)
    written = re.sub("([0-9])\s?⅝", "\\1 5/8", written)
    written = re.sub("([0-9])\s?⅞", "\\1 7/8", written)
    return written


def input_preprocessing(sent: str, lang: str):
    """ Function for preprocessing the input texts. The function first does
    some basic tokenization. For English, it then also processes Greek letters
    such as Δ or λ (if any).

    Args:
        sent: input text.
        lang: language

    Returns: preprocessed input text.
    """
    # Basic Preprocessing and Tokenization
    if lang == constants.ENGLISH:
        sent = sent.replace('+', ' plus ')
        sent = sent.replace('=', ' equals ')
        sent = sent.replace('@', ' at ')
        sent = sent.replace('*', ' times ')
        # Greek letters processing
        for jx, tok in enumerate(sent):
            if tok in constants.EN_GREEK_TO_SPOKEN:
                sent = sent[:jx] + constants.EN_GREEK_TO_SPOKEN[tok] + sent[jx + 1 :]

    sent = convert_superscript(sent)
    sent = convert_fraction(sent)
    sent = add_space_around_dash(sent)

    return sent


def read_data_file(fp: str, lang: str, max_insts: int = -1):
    """ Reading the raw data from a file of NeMo format
    For more info about the data format, refer to the
    `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

    Args:
        fp: file paths
        lang: language
        max_insts: Maximum number of instances (-1 means no limit)
    Returns:
        insts: List of sentences parsed as list of words
    """
    insts, w_words, s_words, classes = [], [], [], []
    # Read input file
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in input_preprocessing(line.strip(), lang=lang).split('\t')]
            if es[0] == '<eos>':
                inst = (deepcopy(classes), deepcopy(w_words), deepcopy(s_words))
                insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []

                if max_insts > 0 and len(insts) >= max_insts:
                    break
            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
    return insts


def normalize_str(input_str):
    """ Normalize an input string """
    return input_str.strip().lower().replace("  ", " ")


def remove_puncts(input_str):
    """ Remove punctuations from an input string """
    return input_str.translate(str.maketrans('', '', string.punctuation))
