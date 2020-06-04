# coding=utf-8
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

# Lint as: python3
"""Utility functions for LaserTagger."""

from __future__ import absolute_import, division, print_function

import json
from typing import Iterator, Mapping, Sequence, Text, Tuple


def get_token_list(text):
    """Returns a list of tokens.

  This function expects that the tokens in the text are separated by space
  character(s). Example: "ca n't , touch". This is the case at least for the
  public DiscoFuse and WikiSplit datasets.

  Args:
    text: String to be split into tokens.
  """
    return text.split()


def yield_sources_and_targets(input_file):
    """Reads and yields source lists and targets from the input file.

  Args:
    input_file: Path to the input file.

  Yields:
    Tuple with (list of source texts, target text).
  """

    for sources, target in _yield_fn(input_file):
        yield sources, target


def _yield_fn(input_file):
    # The format expects a TSV file with the source on the first and the
    # target on the second column.
    with open(input_file, 'r') as f:
        for line in f:
            source, target = line.rstrip('\n').split('\t')
            yield [source], target


def read_label_map(path):
    """Returns label map read from the given path."""
    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        else:
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
