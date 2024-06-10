# Copyright 2019 The Google Research Authors.
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
from collections import OrderedDict
from os import path
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase

from nemo.utils.decorators import deprecated_warning

"""Build BERT Examples from asr hypothesis, customization candidates, target labels, span info.
"""


class BertExample(object):
    """Class for training and inference examples for BERT.

    Attributes:
        features: Feature dictionary.
    """

    def __init__(
        self,
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        input_ids_for_subwords: List[int],
        input_mask_for_subwords: List[int],
        segment_ids_for_subwords: List[int],
        character_pos_to_subword_pos: List[int],
        fragment_indices: List[Tuple[int, int, int]],
        labels_mask: List[int],
        labels: List[int],
        spans: List[Tuple[int, int, int]],
        default_label: int,
    ) -> None:
        """Inputs to the example wrapper

        Args:
            input_ids: indices of single characters (treated as subwords)
            input_mask: list of bools with 0s in place of input_ids to be masked
            segment_ids: list of ints from 0 to 10 to denote the text segment type (
                0 - for tokens of ASR hypothesis,
                1 - for tokens of the first candidate
                ...
                10 - for tokens of the tenth candidate
            )
            input_ids_for_subwords: indices of real subwords (as tokenized by bert tokenizer)
            input_mask_for_subwords: list of bools with 0s in place of input_ids_for_subwords to be masked
            segment_ids_for_subwords: same as segment_ids but for input_ids_for_subwords
            character_pos_to_subword_pos: list of size=len(input_ids), value=(position of corresponding subword in input_ids_for_subwords)
            fragment_indices: list of tuples (start_position, end_position, candidate_id), end is exclusive, candidate_id can be -1 if not set
            labels_mask: bool tensor with 0s in place of label tokens to be masked
            labels: indices of semiotic classes which should be predicted from each of the
                corresponding input tokens
            spans: list of tuples (class_id, start_position, end_position), end is exclusive, class is always 1(CUSTOM)
            default_label: The default label
        """
        # deprecation warning
        deprecated_warning("BertExample")

        input_len = len(input_ids)
        if not (
            input_len == len(input_mask)
            and input_len == len(segment_ids)
            and input_len == len(labels_mask)
            and input_len == len(labels)
            and input_len == len(character_pos_to_subword_pos)
        ):
            raise ValueError("All feature lists should have the same length ({})".format(input_len))

        input_len_for_subwords = len(input_ids_for_subwords)
        if not (
            input_len_for_subwords == len(input_mask_for_subwords)
            and input_len_for_subwords == len(segment_ids_for_subwords)
        ):
            raise ValueError(
                "All feature lists for subwords should have the same length ({})".format(input_len_for_subwords)
            )

        self.features = OrderedDict(
            [
                ("input_ids", input_ids),
                ("input_mask", input_mask),
                ("segment_ids", segment_ids),
                ("input_ids_for_subwords", input_ids_for_subwords),
                ("input_mask_for_subwords", input_mask_for_subwords),
                ("segment_ids_for_subwords", segment_ids_for_subwords),
                ("character_pos_to_subword_pos", character_pos_to_subword_pos),
                ("fragment_indices", fragment_indices),
                ("labels_mask", labels_mask),
                ("labels", labels),
                ("spans", spans),
            ]
        )
        self._default_label = default_label


class BertExampleBuilder(object):
    """Builder class for BertExample objects."""

    def __init__(
        self,
        label_map: Dict[str, int],
        semiotic_classes: Dict[str, int],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        """Initializes an instance of BertExampleBuilder.

        Args:
            label_map: Mapping from tags to tag IDs.
            semiotic_classes: Mapping from semiotic classes to their ids.
            tokenizer: Tokenizer object.
            max_seq_length: Maximum sequence length.
        """
        # deprecation warning
        deprecated_warning("BertExampleBuilder")

        self._label_map = label_map
        self._semiotic_classes = semiotic_classes
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        # one span usually covers one or more words and it only exists for custom phrases, so there are much less spans than characters.
        self._max_spans_length = max(4, int(max_seq_length / 20))
        self._pad_id = self._tokenizer.pad_token_id
        self._default_label = 0

    def build_bert_example(
        self, hyp: str, ref: str, target: Optional[str] = None, span_info: Optional[str] = None, infer: bool = False
    ) -> Optional[BertExample]:
        """Constructs a BERT Example.

        Args:
            hyp: Hypothesis text.
            ref: Candidate customization variants divided by ';'
            target:
                if infer==False, string of labels (each label is 1-based index of correct candidate) or 0.
                if infer==True, it can be None or string of labels (each label is 1-based index of some candidate). In inference this can be used to get corresponding fragments to fragment_indices.
            span_info:
                string of format "CUSTOM 6 20;CUSTOM 40 51", number of parts corresponds to number of targets. Can be empty if target is 0.
                If infer==False, numbers are correct start and end(exclusive) positions of the corresponding target candidate in the text.
                If infer==True, numbers are EXPECTED positions in the text. In inference this can be used to get corresponding fragments to fragment_indices.
            infer: inference mode
        Returns:
            BertExample, or None if the conversion from text to tags was infeasible

        Example (infer=False):
            hyp: "a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o"
            ref: "d i d i e r _ s a u m o n;a s t r o n o m i e;t r i s t a n _ g u i l l o t;t r i s t e s s e;m o n a d e;c h r i s t i a n;a s t r o n o m e r;s o l o m o n;d i d i d i d i d i;m e r c y"
            target: "1 3"
            span_info: "CUSTOM 12 23;CUSTOM 28 41"
        """
        if not ref.count(";") == 9:
            raise ValueError("Expect 10 candidates: " + ref)

        span_info_parts = []
        targets = []

        if len(target) > 0 and target != "0":
            span_info_parts = span_info.split(";")
            targets = list(map(int, target.split(" ")))
            if len(span_info_parts) != len(targets):
                raise ValueError(
                    "len(span_info_parts)="
                    + str(len(span_info_parts))
                    + " is different from len(target_parts)="
                    + str(len(targets))
                )

        tags = [0 for _ in hyp.split()]
        if not infer:
            for p, t in zip(span_info_parts, targets):
                c, start, end = p.split(" ")
                start = int(start)
                end = int(end)
                tags[start:end] = [t for i in range(end - start)]

        # get input features for characters
        (
            input_ids,
            input_mask,
            segment_ids,
            labels_mask,
            labels,
            _,
            _,
        ) = self._get_input_features(hyp=hyp, ref=ref, tags=tags)

        # get input features for words
        hyp_with_words = hyp.replace(" ", "").replace("_", " ")
        ref_with_words = ref.replace(" ", "").replace("_", " ")
        (
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            _,
            _,
            _,
            _,
        ) = self._get_input_features(hyp=hyp_with_words, ref=ref_with_words, tags=None)

        # used in forward to concatenate subword embeddings to character embeddings
        character_pos_to_subword_pos = self._map_characters_to_subwords(input_ids, input_ids_for_subwords)

        fragment_indices = []
        if infer:
            # used in inference to take argmax over whole fragments instead of separate characters to get more consistent predictions
            fragment_indices = self._get_fragment_indices(hyp, targets, span_info_parts)

        spans = []
        if not infer:
            # during training spans are used in validation step to calculate accuracy on whole custom phrases instead of separate characters
            spans = self._get_spans(span_info_parts)

        if len(input_ids) > self._max_seq_length or len(spans) > self._max_spans_length:
            print(
                "Max len exceeded: len(input_ids)=",
                len(input_ids),
                "; _max_seq_length=",
                self._max_seq_length,
                "; len(spans)=",
                len(spans),
                "; _max_spans_length=",
                self._max_spans_length,
            )
            return None

        example = BertExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            input_ids_for_subwords=input_ids_for_subwords,
            input_mask_for_subwords=input_mask_for_subwords,
            segment_ids_for_subwords=segment_ids_for_subwords,
            character_pos_to_subword_pos=character_pos_to_subword_pos,
            fragment_indices=fragment_indices,
            labels_mask=labels_mask,
            labels=labels,
            spans=spans,
            default_label=self._default_label,
        )
        return example

    def _get_spans(self, span_info_parts: List[str]) -> List[Tuple[int, int, int]]:
        """Converts span_info string into a list of (class_id, start, end) where start, end are coordinates of starting and ending(exclusive) tokens in input_ids of BertExample

        Example:
            span_info_parts: ["CUSTOM 37 41", "CUSTOM 47 52", "CUSTOM 42 46", "CUSTOM 0 7"]
            result: [(1, 38, 42), (1, 48, 53), (1, 43, 47), (1, 1, 8)]
        """
        result_spans = []

        for p in span_info_parts:
            if p == "":
                break
            c, start, end = p.split(" ")
            if c not in self._semiotic_classes:
                raise KeyError("class=" + c + " not found in self._semiotic_classes")
            cid = self._semiotic_classes[c]
            # +1 because this should be indexing on input_ids which has [CLS] token at beginning
            start = int(start) + 1
            end = int(end) + 1
            result_spans.append((cid, start, end))
        return result_spans

    def _get_fragment_indices(
        self, hyp: str, targets: List[int], span_info_parts: List[str]
    ) -> Tuple[List[Tuple[int, int, int]]]:
        """Build fragment indices for real candidates.
        This is used only at inference.
        After external candidate retrieval we know approximately, where the candidate is located in the text (from the positions of matched n-grams).
        In this function we
           1) adjust start/end positions to match word borders (possibly in multiple ways).
           2) generate content for fragment_indices tensor (it will be used during inference to average all predictions inside each fragment).

        Args:
            hyp: ASR-hypothesis where space separates single characters (real space is replaced to underscore).
            targets: list of candidate ids (only for real candidates, not dummy)
            span_info_parts: list of strings of format like "CUSTOM 12 25", corresponding to each of targets, with start/end coordinates in text.
        Returns:
            List of tuples (start, end, target) where start and end are positions in ASR-hypothesis, target is candidate_id.
            Note that returned fragments can be unsorted and can overlap, it's ok.
        Example:
            hyp: "a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o"
            targets: [1 2 3 4 6 7 9]
            span_info_parts: ["CUSTOM 12 25", "CUSTOM 0 10", "CUSTOM 27 42", ...], where numbers are EXPECTED start/end positions of corresponding target candidates in the text. These positions will be adjusted in this functuion.
            fragment_indices: [(1, 12, 2), (13, 24, 1), (13, 28, 1), ..., (29, 42, 3)]
        """

        fragment_indices = []

        letters = hyp.split()

        for target, p in zip(targets, span_info_parts):
            _, start, end = p.split(" ")
            start = int(start)
            end = min(int(end), len(hyp))  # guarantee that end is not outside length

            # Adjusting strategy 1: expand both sides to the nearest space.
            # Adjust start by finding the nearest left space or beginning of text. If start is already some word beginning, it won't change.
            k = start
            while k > 0 and letters[k] != '_':
                k -= 1
            adjusted_start = k if k == 0 else k + 1

            # Adjust end by finding the nearest right space. If end is already space or sentence end, it won't change.
            k = end
            while k < len(letters) and letters[k] != '_':
                k += 1
            adjusted_end = k

            # +1 because this should be indexing on input_ids which has [CLS] token at beginning
            fragment_indices.append((adjusted_start + 1, adjusted_end + 1, target))

            # Adjusting strategy 2: try to shrink to the closest space (from left or right or both sides).
            # For example, here the candidate "shippers" has a matching n-gram covering part of previous word
            # a b o u t _ o u r _ s h i p e r s _ b u t _ y o u _ k n o w
            # 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            expanded_fragment = "".join(letters[adjusted_start:adjusted_end])
            left_space_position = expanded_fragment.find("_")
            right_space_position = expanded_fragment.rfind("_")
            is_left_shrink = False
            is_right_shrink = False
            if left_space_position > -1 and left_space_position < len(expanded_fragment) / 2:
                # +1 because of CLS token, another +1 to put start position after found space
                fragment_indices.append((adjusted_start + 1 + left_space_position + 1, adjusted_end + 1, target))
                is_left_shrink = True
            if right_space_position > -1 and right_space_position > len(expanded_fragment) / 2:
                fragment_indices.append((adjusted_start + 1, adjusted_start + 1 + right_space_position, target))
                is_right_shrink = True
            if is_left_shrink and is_right_shrink:
                fragment_indices.append(
                    (adjusted_start + 1 + left_space_position + 1, adjusted_start + 1 + right_space_position, target)
                )

        return fragment_indices

    def _map_characters_to_subwords(self, input_ids: List[int], input_ids_for_subwords: List[int]) -> List[int]:
        """Maps each single character to the position of its corresponding subword.

        Args:
            input_ids: List of character token ids.
            input_ids_for_subwords: List of subword token ids.
        Returns:
            List of subword positions in input_ids_for_subwords. Its length is equal to len(input_ids)

        Example:
            input_ids: [101, 1037, 1055, 1056, 1054, 1051, 1050, ..., 1051, 102, 1040, ..., 1050, 102, 1037, ..., 1041, 102, ..., 102]
            input_ids_for_subwords: [101, 26357, 2106, 2666, 2061, 8202, 1998, 13012, 16643, 2319, 1043, 7174, 102, 2106, 3771, 7842, 2819, 2239, 102, ..., 102]
            result: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, ... , 45, 46, 46, 46, 46, 46, 47]
        """
        character_pos_to_subword_pos = [0 for _ in input_ids]

        ## '[CLS]', 'a', 's', 't', 'r', 'o', 'n', 'o', 'm', 'e', 'r', 's', '_', 'd', 'i', ..., 'l', 'o', '[SEP]', 'd', 'i', 'd', 'i', 'e', 'r', '_', 's', 'a', 'u', 'm', 'o', 'n', ..., '[SEP]'
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        ## '[CLS]', 'astronomers', 'did', '##ie', 'so', '##mon', 'and', 'tri', '##sti', '##an', 'g', '##llo', '[SEP]', 'did', '##ier', 'sa', '##um', '##on', '[SEP]', 'astro', '##no', '##mie', '[SEP]', 'tristan', 'gui', '##llo', '##t', '[SEP]', ..., '[SEP]', 'mercy', '[SEP]']
        tokens_for_subwords = self._tokenizer.convert_ids_to_tokens(input_ids_for_subwords)
        j = 0  # index for tokens_for_subwords
        j_offset = 0  # current letter index within subword
        for i in range(len(tokens)):
            character = tokens[i]
            subword = tokens_for_subwords[j]
            if character == "[CLS]" and subword == "[CLS]":
                character_pos_to_subword_pos[i] = j
                j += 1
                continue
            if character == "[SEP]" and subword == "[SEP]":
                character_pos_to_subword_pos[i] = j
                j += 1
                continue
            if character == "[CLS]" or character == "[SEP]" or subword == "[CLS]" or subword == "[SEP]":
                raise IndexError(
                    "character["
                    + str(i)
                    + "]="
                    + character
                    + "; subword["
                    + str(j)
                    + ";="
                    + subword
                    + "subwords="
                    + str(tokens_for_subwords)
                )
            # At this point we expect that
            #    subword either 1) is a normal first token of a word or 2) starts with "##" (not first word token)
            #    character either 1) is a normal character or 2) is a space character "_"
            if character == "_":
                character_pos_to_subword_pos[i] = j - 1  # space is assigned to previous subtoken
                continue
            if j_offset < len(subword):
                if character == subword[j_offset]:
                    character_pos_to_subword_pos[i] = j
                    j_offset += 1
                else:
                    raise IndexError(
                        "character mismatch:"
                        + "i="
                        + str(i)
                        + "j="
                        + str(j)
                        + "j_offset="
                        + str(j_offset)
                        + "; len(tokens)="
                        + str(len(tokens))
                        + "; len(subwords)="
                        + str(len(tokens_for_subwords))
                    )
            # if subword is finished, increase j
            if j_offset >= len(subword):
                j += 1
                j_offset = 0
                if j >= len(tokens_for_subwords):
                    break
                if tokens_for_subwords[j].startswith("##"):
                    j_offset = 2
        # check that all subword tokens are processed
        if j < len(tokens_for_subwords):
            raise IndexError(
                "j="
                + str(j)
                + "; len(tokens)="
                + str(len(tokens))
                + "; len(subwords)="
                + str(len(tokens_for_subwords))
            )
        return character_pos_to_subword_pos

    def _get_input_features(
        self, hyp: str, ref: str, tags: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[str], List[int]]:
        """Converts given ASR-hypothesis(hyp) and candidate string(ref) to features(token ids, mask, segment ids, etc).

        Args:
            hyp: Hypothesis text.
            ref: Candidate customization variants divided by ';'
            tags: List of labels corresponding to each token of ASR-hypothesis or None when building an example during inference.
        Returns:
            Features (input_ids, input_mask, segment_ids, labels_mask, labels, hyp_tokens, token_start_indices)

        Note that this method is called both for character-based example and for word-based example (to split to subwords).

        Character-based example:
            hyp:  "a s t r o n o m e r s _ d i d i e _ s o m o n _ a n d _ t r i s t i a n _ g l l o"
            ref:  "d i d i e r _ s a u m o n;a s t r o n o m i e;t r i s t a n _ g u i l l o t;t r i s t e s s e;m o n a d e;c h r i s t i a n;a s t r o n o m e r;s o l o m o n;d i d i d i d i d i;m e r c y"
            tags: "0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3"

            resulting token sequence:
                '[CLS]', 'a', 's', 't', 'r', 'o', 'n', 'o', 'm', 'e', 'r', 's', '_', 'd', 'i', ..., 'l', 'o', '[SEP]', 'd', 'i', 'd', 'i', 'e', 'r', '_', 's', 'a', 'u', 'm', 'o', 'n', ..., '[SEP]'

        Word-based example:
            hyp:  "astronomers didie somon and tristian gllo"
            ref:  "didier saumon;astronomie;tristan guillot;tristesse;monade;christian;astronomer;solomon;dididididi;mercy"
            tags: None (not used for word-based case)

            resulting token sequence:
                '[CLS]', 'astronomers', 'did', '##ie', 'so', '##mon', 'and', 'tri', '##sti', '##an', 'g', '##llo', '[SEP]', 'did', '##ier', 'sa', '##um', '##on', '[SEP]', 'astro', '##no', '##mie', '[SEP]', 'tristan', 'gui', '##llo', '##t', '[SEP]', ..., '[SEP]', 'mercy', '[SEP]']
        """

        labels_mask = []
        labels = []
        if tags is None:
            hyp_tokens, token_start_indices = self._split_to_wordpieces(hyp.split())
        else:
            hyp_tokens, labels, token_start_indices = self._split_to_wordpieces_with_labels(hyp.split(), tags)
        references = ref.split(";")
        all_ref_tokens = []
        all_ref_segment_ids = []
        for i in range(len(references)):
            ref_tokens, _ = self._split_to_wordpieces(references[i].split())
            all_ref_tokens.extend(ref_tokens + ["[SEP]"])
            all_ref_segment_ids.extend([i + 1] * (len(ref_tokens) + 1))

        input_tokens = ["[CLS]"] + hyp_tokens + ["[SEP]"] + all_ref_tokens  # ends with [SEP]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] + [0] * len(hyp_tokens) + [0] + all_ref_segment_ids
        if len(input_ids) != len(segment_ids):
            raise ValueError(
                "len(input_ids)="
                + str(len(input_ids))
                + " is different from len(segment_ids)="
                + str(len(segment_ids))
            )

        if tags:
            labels_mask = [0] + [1] * len(labels) + [0] + [0] * len(all_ref_tokens)
            labels = [0] + labels + [0] + [0] * len(all_ref_tokens)
        return (input_ids, input_mask, segment_ids, labels_mask, labels, hyp_tokens, token_start_indices)

    def _split_to_wordpieces_with_labels(
        self, tokens: List[str], labels: List[int]
    ) -> Tuple[List[str], List[int], List[int]]:
        """Splits tokens (and the labels accordingly) to WordPieces.

        Args:
            tokens: Tokens to be split.
            labels: Labels (one per token) to be split.

        Returns:
            3-tuple with the split tokens, split labels, and the indices of starting tokens of words
        """
        bert_tokens = []  # Original tokens split into wordpieces.
        bert_labels = []  # Label for each wordpiece.
        # Index of each wordpiece that starts a new token.
        token_start_indices = []
        for i, token in enumerate(tokens):
            # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
            token_start_indices.append(len(bert_tokens) + 1)
            pieces = self._tokenizer.tokenize(token)
            bert_tokens.extend(pieces)
            bert_labels.extend([labels[i]] * len(pieces))
        return bert_tokens, bert_labels, token_start_indices

    def _split_to_wordpieces(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """Splits tokens to WordPieces.

        Args:
            tokens: Tokens to be split.

        Returns:
            tuple with the split tokens, and the indices of the WordPieces that start a token.
        """
        bert_tokens = []  # Original tokens split into wordpieces.
        # Index of each wordpiece that starts a new token.
        token_start_indices = []
        for i, token in enumerate(tokens):
            # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
            token_start_indices.append(len(bert_tokens) + 1)
            pieces = self._tokenizer.tokenize(token)
            bert_tokens.extend(pieces)
        return bert_tokens, token_start_indices

    def read_input_file(
        self, input_filename: str, infer: bool = False
    ) -> Union[List['BertExample'], Tuple[List['BertExample'], Tuple[str, str]]]:
        """Reads in Tab Separated Value file and converts to training/inference-ready examples.

        Args:
            example_builder: Instance of BertExampleBuilder
            input_filename: Path to the TSV input file.
            infer: If true, input examples do not contain target info.

        Returns:
            examples: List of converted examples (BertExample).
               or
            (examples, hyps_refs): If infer==true, returns h
        """

        if not path.exists(input_filename):
            raise ValueError("Cannot find file: " + input_filename)
        examples = []  # output list of BertExample
        hyps_refs = []  # output list of tuples (ASR-hypothesis, candidate_str)
        with open(input_filename, 'r') as f:
            for line in f:
                if len(examples) % 1000 == 0:
                    logging.info("{} examples processed.".format(len(examples)))
                if infer:
                    parts = line.rstrip('\n').split('\t')
                    hyp, ref, target, span_info = parts[0], parts[1], None, None
                    if len(parts) == 4:
                        target, span_info = parts[2], parts[3]
                    try:
                        example = self.build_bert_example(hyp, ref, target=target, span_info=span_info, infer=infer)
                    except Exception as e:
                        logging.warning(str(e))
                        logging.warning(line)
                        continue
                    if example is None:
                        logging.info("cannot create example: ")
                        logging.info(line)
                        continue
                    hyps_refs.append((hyp, ref))
                    examples.append(example)
                else:
                    hyp, ref, target, semiotic_info = line.rstrip('\n').split('\t')
                    try:
                        example = self.build_bert_example(
                            hyp, ref, target=target, span_info=semiotic_info, infer=infer
                        )
                    except Exception as e:
                        logging.warning(str(e))
                        logging.warning(line)
                        continue
                    if example is None:
                        logging.info("cannot create example: ")
                        logging.info(line)
                        continue
                    examples.append(example)
        logging.info(f"Done. {len(examples)} examples converted.")
        if infer:
            return examples, hyps_refs
        return examples
