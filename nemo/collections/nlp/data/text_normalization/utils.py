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
import sys
from copy import deepcopy
from unicodedata import category

import regex as re
from tqdm import tqdm

from nemo.collections.nlp.data.text_normalization import constants
from nemo.utils import logging

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


def post_process_punct(input: str, normalized_text: str, add_unicode_punct: bool = False):
    """
    Post-processing of the normalized output to match input in terms of spaces around punctuation marks.
    After NN normalization, Moses detokenization puts a space after
    punctuation marks, and attaches an opening quote "'" to the word to the right.
    E.g., input to the TN NN model is "12 test' example",
    after normalization and detokenization -> "twelve test 'example" (the quote is considered to be an opening quote,
    but it doesn't match the input and can cause issues during TTS voice generation.)
    The current function will match the punctuation and spaces of the normalized text with the input sequence.
    "12 test' example" -> "twelve test 'example" -> "twelve test' example" (the quote was shifted to match the input).

    Args:
        input: input text (original input to the NN, before normalization or tokenization)
        normalized_text: output text (output of the TN NN model)
        add_unicode_punct: set to True to handle unicode punctuation marks as well as default string.punctuation (increases post processing time)
    """
    # in the post-processing WFST graph "``" are repalced with '"" quotes (otherwise single quotes "`" won't be handled correctly)
    # this function fixes spaces around them based on input sequence, so here we're making the same double quote replacement
    # to make sure these new double quotes work with this function
    if "``" in input and "``" not in normalized_text:
        input = input.replace("``", '"')
    input = [x for x in input]
    normalized_text = [x for x in normalized_text]
    punct_marks = [x for x in string.punctuation if x in input]

    if add_unicode_punct:
        punct_unicode = [
            chr(i)
            for i in range(sys.maxunicode)
            if category(chr(i)).startswith("P") and chr(i) not in punct_default and chr(i) in input
        ]
        punct_marks = punct_marks.extend(punct_unicode)

    for punct in punct_marks:
        try:
            equal = True
            if input.count(punct) != normalized_text.count(punct):
                equal = False
            idx_in, idx_out = 0, 0
            while punct in input[idx_in:]:
                idx_out = normalized_text.index(punct, idx_out)
                idx_in = input.index(punct, idx_in)

                def _is_valid(idx_out, idx_in, normalized_text, input):
                    """Check if previous or next word match (for cases when punctuation marks are part of
                    semiotic token, i.e. some punctuation can be missing in the normalized text)"""
                    return (idx_out > 0 and idx_in > 0 and normalized_text[idx_out - 1] == input[idx_in - 1]) or (
                        idx_out < len(normalized_text) - 1
                        and idx_in < len(input) - 1
                        and normalized_text[idx_out + 1] == input[idx_in + 1]
                    )

                if not equal and not _is_valid(idx_out, idx_in, normalized_text, input):
                    idx_in += 1
                    continue
                if idx_in > 0 and idx_out > 0:
                    if normalized_text[idx_out - 1] == " " and input[idx_in - 1] != " ":
                        normalized_text[idx_out - 1] = ""

                    elif normalized_text[idx_out - 1] != " " and input[idx_in - 1] == " ":
                        normalized_text[idx_out - 1] += " "

                if idx_in < len(input) - 1 and idx_out < len(normalized_text) - 1:
                    if normalized_text[idx_out + 1] == " " and input[idx_in + 1] != " ":
                        normalized_text[idx_out + 1] = ""
                    elif normalized_text[idx_out + 1] != " " and input[idx_in + 1] == " ":
                        normalized_text[idx_out] = normalized_text[idx_out] + " "
                idx_out += 1
                idx_in += 1
        except:
            logging.debug(f"Skipping post-processing of {''.join(normalized_text)} for '{punct}'")

    normalized_text = "".join(normalized_text)
    return re.sub(r' +', ' ', normalized_text)
