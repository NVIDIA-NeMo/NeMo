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


import json
import re
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Set, Tuple

EOS_TYPE = "EOS"
PUNCT_TYPE = "PUNCT"
PLAIN_TYPE = "PLAIN"
Instance = namedtuple('Instance', 'token_type un_normalized normalized')
known_types = [
    "PLAIN",
    "DATE",
    "CARDINAL",
    "LETTERS",
    "VERBATIM",
    "MEASURE",
    "DECIMAL",
    "ORDINAL",
    "DIGIT",
    "MONEY",
    "TELEPHONE",
    "ELECTRONIC",
    "FRACTION",
    "TIME",
    "ADDRESS",
]


def _load_kaggle_text_norm_file(file_path: str) -> List[Instance]:
    """
    https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
    Loads text file in the Kaggle Google text normalization file format: <semiotic class>\t<unnormalized text>\t<`self` if trivial class or normalized text>
    E.g. 
    PLAIN   Brillantaisia   <self>
    PLAIN   is      <self>
    PLAIN   a       <self>
    PLAIN   genus   <self>
    PLAIN   of      <self>
    PLAIN   plant   <self>
    PLAIN   in      <self>
    PLAIN   family  <self>
    PLAIN   Acanthaceae     <self>
    PUNCT   .       sil
    <eos>   <eos>

    Args:
        file_path: file path to text file

    Returns: flat list of instances 
    """
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts[0] == "<eos>":
                res.append(Instance(token_type=EOS_TYPE, un_normalized="", normalized=""))
            else:
                l_type, l_token, l_normalized = parts
                l_token = l_token.lower()
                l_normalized = l_normalized.lower()

                if l_type == PLAIN_TYPE:
                    res.append(Instance(token_type=l_type, un_normalized=l_token, normalized=l_token))
                elif l_type != PUNCT_TYPE:
                    res.append(Instance(token_type=l_type, un_normalized=l_token, normalized=l_normalized))
    return res


def load_files(file_paths: List[str], load_func=_load_kaggle_text_norm_file) -> List[Instance]:
    """
    Load given list of text files using the `load_func` function.

    Args: 
        file_paths: list of file paths
        load_func: loading function

    Returns: flat list of instances
    """
    res = []
    for file_path in file_paths:
        res.extend(load_func(file_path=file_path))
    return res


def clean_generic(text: str) -> str:
    """
    Cleans text without affecting semiotic classes.

    Args:
        text: string

    Returns: cleaned string
    """
    text = text.strip()
    text = text.lower()
    return text


def evaluate(preds: List[str], labels: List[str], input: Optional[List[str]] = None, verbose: bool = True) -> float:
    """
    Evaluates accuracy given predictions and labels. 

    Args:
        preds: predictions
        labels: labels
        input: optional, only needed for verbosity
        verbose: if true prints [input], golden labels and predictions

    Returns accuracy
    """
    acc = 0
    nums = len(preds)
    for i in range(nums):
        pred_norm = clean_generic(preds[i])
        label_norm = clean_generic(labels[i])
        if pred_norm == label_norm:
            acc = acc + 1
        else:
            if input:
                print(f"inpu: {json.dumps(input[i])}")
            print(f"gold: {json.dumps(label_norm)}")
            print(f"pred: {json.dumps(pred_norm)}")
    return acc / nums


def training_data_to_tokens(
    data: List[Instance], category: Optional[str] = None
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Filters the instance list by category if provided and converts it into a map from token type to list of un_normalized and normalized strings

    Args:
        data: list of instances
        category: optional semiotic class category name

    Returns Dict: token type -> (list of un_normalized strings, list of normalized strings)
    """
    result = defaultdict(lambda: ([], []))
    for instance in data:
        if instance.token_type != EOS_TYPE:
            if category is None or instance.token_type == category:
                result[instance.token_type][0].append(instance.un_normalized)
                result[instance.token_type][1].append(instance.normalized)
    return result


def training_data_to_sentences(data: List[Instance]) -> Tuple[List[str], List[str], List[Set[str]]]:
    """
    Takes instance list, creates list of sentences split by EOS_Token
    Args:
        data: list of instances
    Returns (list of unnormalized sentences, list of normalized sentences, list of sets of categories in a sentence)
    """
    # split data at EOS boundaries
    sentences = []
    sentence = []
    categories = []
    sentence_categories = set()

    for instance in data:
        if instance.token_type == EOS_TYPE:
            sentences.append(sentence)
            sentence = []
            categories.append(sentence_categories)
            sentence_categories = set()
        else:
            sentence.append(instance)
            sentence_categories.update([instance.token_type])
    un_normalized = [" ".join([instance.un_normalized for instance in sentence]) for sentence in sentences]
    normalized = [" ".join([instance.normalized for instance in sentence]) for sentence in sentences]
    return un_normalized, normalized, categories


def post_process_punctuation(text: str) -> str:
    """
    Normalized quotes and spaces

    Args:
        text: text

    Returns: text with normalized spaces and quotes
    """
    text = (
        text.replace('( ', '(')
        .replace(' )', ')')
        .replace('{ ', '{')
        .replace(' }', '}')
        .replace('[ ', '[')
        .replace(' ]', ']')
        .replace('  ', ' ')
        .replace('”', '"')
        .replace("’", "'")
        .replace("»", '"')
        .replace("«", '"')
        .replace("\\", "")
        .replace("„", '"')
        .replace("´", "'")
        .replace("’", "'")
        .replace('“', '"')
        .replace("‘", "'")
        .replace('`', "'")
        .replace('- -', "--")
    )

    for punct in "!,.:;?":
        text = text.replace(f' {punct}', punct)
    return text.strip()


def pre_process(text: str) -> str:
    """
    Optional text preprocessing before normalization (part of TTS TN pipeline)

    Args:
        text: string that may include semiotic classes

    Returns: text with spaces around punctuation marks
    """
    space_both = '[]'
    for punct in space_both:
        text = text.replace(punct, ' ' + punct + ' ')

    # remove extra space
    text = re.sub(r' +', ' ', text)
    return text


def check_installation():
    try:
        import pynini

        PYNINI_AVAILABLE = True

    except (ModuleNotFoundError, ImportError):
        PYNINI_AVAILABLE = False
    return PYNINI_AVAILABLE


def get_installation_msg():
    msg = "`pynini` is not installed ! \n Please run the `nemo_text_processing/setup.sh` script prior to usage of this toolkit."
    return msg


def load_file(file_path: str) -> List[str]:
    """
    Loads given text file with separate lines into list of string.

    Args: 
        file_path: file path

    Returns: flat list of string
    """
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            res.append(line)
    return res


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.

    Args:
        file_path: file path
        data: list of string
        
    """
    with open(file_path, 'w') as fp:
        for line in data:
            fp.write(line + '\n')
