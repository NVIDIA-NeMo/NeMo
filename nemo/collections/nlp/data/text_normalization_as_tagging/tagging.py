# Copyright 2019 The Google Research Authors.
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


"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/lasertagger/blob/master/tagging.py
"""

import re
from enum import Enum
from typing import List, Tuple

import nemo.collections.nlp.data.text_normalization_as_tagging.utils as utils


"""Classes representing a tag and a text editing task.

Tag corresponds to an edit operation, while EditingTask is a container for the
input that LaserTagger takes. EditingTask also has a method for realizing the
output text given the predicted tags.
"""


class SwapType(Enum):
    """Type of swap"""

    LONG_LEFT = 1  # token should be moved to the leftmost position of the whole semiotic span
    LONG_RIGHT = 2  # token should be moved to the rightmost position of the whole semiotic span
    SHORT_LEFT = 3  # token should be swapped with the left adjacent token
    SHORT_RIGHT = 4  # token should be swapped with the right adjacent token


class Token:
    """Class for the output token"""

    def __init__(self, inp: str, tag: str, out: str) -> None:
        self.inp = inp
        self.tag = tag
        self.out = out
        self.swap = None
        if self.out.endswith(">>"):
            self.swap = SwapType.LONG_RIGHT
        elif self.out.endswith("<<"):
            self.swap = SwapType.LONG_LEFT
        elif self.out.endswith(">"):
            self.swap = SwapType.SHORT_RIGHT
        elif self.out.endswith("<"):
            self.swap = SwapType.SHORT_LEFT

    @property
    def is_begin(self) -> bool:
        return self.out.startswith("_")

    @property
    def is_end(self) -> bool:
        return self.out.endswith("_")


class TagType(Enum):
    """Base tag which indicates the type of an edit operation."""

    # Keep the tagged token.
    KEEP = 1
    # Delete the tagged token.
    DELETE = 2


class Tag(object):
    """Tag that corresponds to a token edit operation.

    Attributes:
        tag_type: TagType of the tag.
        added_phrase: A phrase that's inserted before the tagged token (can be
            empty).
    """

    def __init__(self, tag: str) -> None:
        """Constructs a Tag object by parsing tag to tag_type and added_phrase.

        Args:
            tag: String representation for the tag which should have the following
                format "<TagType>|<added_phrase>" or simply "<TagType>" if no phrase
                is added before the tagged token. Examples of valid tags include "KEEP",
                "DELETE|and".

        Raises:
            ValueError: If <TagType> is invalid.
        """
        if '|' in tag:
            pos_pipe = tag.index('|')
            tag_type, added_phrase = tag[:pos_pipe], tag[pos_pipe + 1 :]
        else:
            tag_type, added_phrase = tag, ''
        try:
            self.tag_type = TagType[tag_type]
        except KeyError:
            raise ValueError('TagType should be KEEP or DELETE, not {}'.format(tag_type))
        self.added_phrase = added_phrase

    def __str__(self) -> str:
        if not self.added_phrase:
            return self.tag_type.name
        else:
            return '{}|{}'.format(self.tag_type.name, self.added_phrase)


class EditingTask(object):
    """Text-editing task.

    Attributes:
        source_tokens: Tokens of the source texts concatenated into a single list.
        first_tokens: The indices of the first tokens of each source text.
    """

    def __init__(self, source: str) -> None:
        """Initializes an instance of EditingTask.

        Args:
            source: string.
        """
        token_list = utils.get_token_list(source)
        # Tokens of the source texts concatenated into a single list.
        self.source_tokens = []
        # The indices of the first tokens of each source text.
        self.first_tokens = []
        self.first_tokens.append(len(self.source_tokens))
        self.source_tokens.extend(token_list)

    def realize_output(self, tags: List[Tag], semiotic_labels: List[str]) -> Tuple[str, str, str, str]:
        """Realize output text based on the source tokens and predicted tags.

        Args:
            tags: Predicted tags (one for each token in `self.source_tokens`).
            semiotic_labels: Predicted semiotic labels (one for each token in `self.source_tokens`).

        Returns:
            The realizer output text.

        Raises:
            ValueError: If the number of tags doesn't match the number of source
                tokens.
        """
        if len(tags) != len(self.source_tokens) or len(tags) != len(semiotic_labels):
            raise ValueError(
                'The number of tags ({}) should match the number of '
                'source tokens ({}) and semiotic labels({})'.format(
                    len(tags), len(self.source_tokens), len(semiotic_labels)
                )
            )

        sequence = []
        for inp_token, tag in zip(self.source_tokens, tags):
            if tag.added_phrase:
                sequence.append(Token(inp_token, tag.added_phrase, tag.added_phrase))
            elif tag.tag_type == TagType.KEEP:
                sequence.append(Token(inp_token, "<SELF>", inp_token))
            else:
                sequence.append(Token(inp_token, "<DELETE>", ""))
        if len(sequence) != len(semiotic_labels):
            raise ValueError(
                "Length mismatch: len(sequence)="
                + str(len(sequence))
                + "; len(semiotic_labels)="
                + str(len(semiotic_labels))
            )
        out_tokens_with_swap = [t.out for t in sequence]
        out_tags_with_swap = [t.tag for t in sequence]
        out_tags_without_swap = [t.tag for t in sequence]
        previous_semiotic_label_end = -1
        current_semiotic_label = ""
        for i in range(len(sequence)):
            if sequence[i].swap == SwapType.SHORT_LEFT or sequence[i - 1].swap == SwapType.SHORT_RIGHT:
                out_tokens_with_swap[i - 1], out_tokens_with_swap[i] = (
                    out_tokens_with_swap[i],
                    out_tokens_with_swap[i - 1],
                )
                out_tags_with_swap[i - 1], out_tags_with_swap[i] = out_tags_with_swap[i], out_tags_with_swap[i - 1]
            if semiotic_labels[i] != current_semiotic_label:
                previous_semiotic_label_end = i - 1
                current_semiotic_label = semiotic_labels[i]
            if sequence[i].swap == SwapType.LONG_LEFT:
                token = out_tokens_with_swap.pop(i)
                tag = out_tags_with_swap.pop(i)
                out_tokens_with_swap.insert(previous_semiotic_label_end + 1, token)
                out_tags_with_swap.insert(previous_semiotic_label_end + 1, tag)

        # detokenize
        output_tokens_str = " ".join(out_tokens_with_swap).replace("<", "").replace(">", "")
        output_tags_with_swap_str = " ".join(out_tags_with_swap)
        frags = re.split(r"(_[^ ][^_]+[^ ]_)", output_tokens_str)
        output_tokens = []
        for frag in frags:
            if frag.startswith("_") and frag.endswith("_"):
                output_tokens.append(frag.replace(" ", "").replace("_", ""))
            else:
                output_tokens.append(frag.strip().replace("_", ""))
        output_str = " ".join(output_tokens)
        output_str = re.sub(r" +", " ", output_str)
        return (
            output_str,
            " ".join(self.source_tokens),
            " ".join(out_tags_without_swap),
            output_tags_with_swap_str,
        )
