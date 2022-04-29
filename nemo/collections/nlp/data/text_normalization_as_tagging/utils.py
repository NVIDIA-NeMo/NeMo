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


import re
from itertools import groupby
from typing import Dict, List

"""Utility functions for Thutmose Tagger."""


def get_token_list(text: str) -> List[str]:
    """Returns a list of tokens.

    This function expects that the tokens in the text are separated by space
    character(s). Example: "ca n't , touch". This is the case at least for the
    public DiscoFuse and WikiSplit datasets.

    Args:
        text: String to be split into tokens.
    """
    return text.split()


def yield_sources_and_targets(input_filename: str):
    """Reads and yields source lists and targets from the input file.

    Args:
        input_filename: Path to the input file.

    Yields:
        Tuple with (list of source texts, target text).
    """
    # The format expects a TSV file with the source on the first and the
    # target on the second column.
    with open(input_filename, 'r') as f:
        for line in f:
            source, target, semiotic_info = line.rstrip('\n').split('\t')
            yield source, target, semiotic_info


def read_label_map(path: str) -> Dict[str, int]:
    """Return label map read from the given path."""
    with open(path, 'r') as f:
        label_map = {}
        empty_line_encountered = False
        for tag in f:
            tag = tag.strip()
            if tag:
                label_map[tag] = len(label_map)
            else:
                if empty_line_encountered:
                    raise ValueError('There should be no empty lines in the middle of the label map ' 'file.')
                empty_line_encountered = True
        return label_map


def read_semiotic_classes(path: str) -> Dict[str, int]:
    """Return semiotic classes map read from the given path."""
    with open(path, 'r') as f:
        semiotic_classes = {}
        empty_line_encountered = False
        for tag in f:
            tag = tag.strip()
            if tag:
                semiotic_classes[tag] = len(semiotic_classes)
            else:
                if empty_line_encountered:
                    raise ValueError('There should be no empty lines in the middle of the label map ' 'file.')
                empty_line_encountered = True
        return semiotic_classes


def split_text_by_isalpha(s: str):
    """Split string into segments, so that alphabetic sequence is one segment"""
    for k, g in groupby(s, str.isalpha):
        yield ''.join(g)


def spoken_preprocessing(spoken: str) -> str:
    """Preprocess spoken input for Thuthmose tagger model.
    Attention!
    This function is used both during data preparation and during inference.
    If you change it, you should rerun data preparation and retrain the model.
    """
    spoken = spoken.casefold()
    spoken = spoken.replace('_trans', '').replace('_letter_latin', '').replace('_letter', '')

    #  "долларов сэ ш а"  => "долларов-сэ-ш-а"    #join into one token to simplify alignment
    spoken = re.sub(r" долларов сэ ш а", r" долларов-сэ-ш-а", spoken)
    spoken = re.sub(r" доллара сэ ш а", r" доллара-сэ-ш-а", spoken)
    spoken = re.sub(r" доллар сэ ш а", r" доллар-сэ-ш-а", spoken)
    spoken = re.sub(r" фунтов стерлингов", r" фунтов-стерлингов", spoken)
    spoken = re.sub(r" фунта стерлингов", r" фунта-стерлингов", spoken)
    spoken = re.sub(r" фунт стерлингов", r" фунт-стерлингов", spoken)
    spoken = re.sub(r" долларами сэ ш а", r" долларами-сэ-ш-а", spoken)
    spoken = re.sub(r" долларам сэ ш а", r" долларам-сэ-ш-а", spoken)
    spoken = re.sub(r" долларах сэ ш а", r" долларах-сэ-ш-а", spoken)
    spoken = re.sub(r" долларе сэ ш а", r" долларе-сэ-ш-а", spoken)
    spoken = re.sub(r" доллару сэ ш а", r" доллару-сэ-ш-а", spoken)
    spoken = re.sub(r" долларом сэ ш а", r" долларом-сэ-ш-а", spoken)
    spoken = re.sub(r" фунтами стерлингов", r" фунтами-стерлингов", spoken)
    spoken = re.sub(r" фунтам стерлингов", r" фунтам-стерлингов", spoken)
    spoken = re.sub(r" фунтах стерлингов", r" фунтах-стерлингов", spoken)
    spoken = re.sub(r" фунте стерлингов", r" фунте-стерлингов", spoken)
    spoken = re.sub(r" фунту стерлингов", r" фунту-стерлингов", spoken)
    spoken = re.sub(r" фунтом стерлингов", r" фунтом-стерлингов", spoken)

    return spoken
