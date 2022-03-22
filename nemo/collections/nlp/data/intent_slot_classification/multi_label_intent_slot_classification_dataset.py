# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.intent_slot_classification import IntentSlotClassificationDataset
from nemo.collections.nlp.data.intent_slot_classification.intent_slot_classification_dataset import get_features
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType

__all__ = ['MultiLabelIntentSlotClassificationDataset']


class MultiLabelIntentSlotClassificationDataset(IntentSlotClassificationDataset):
    """
    Creates dataset to use for the task of multi-label joint intent
    and slot classification with pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    Args:
        input_file: file containing sentences + labels. The first line is header (sentence [tab] label)
            each line should be [sentence][tab][label] where label can be multiple labels separated by a comma
        slot_file: file containing slot labels, each line corresponding to slot labels for a sentence in input_file. No header.
        num_intents: total number of intents in dict.intents file
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as NemoBertTokenizer
        num_samples: number of samples you want to use for the dataset. If -1, use all dataset. Useful for testing.
        pad_label: pad value use for slot labels. by default, it's the neutral label.
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask.
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask.
        do_lower_case: convert query to lower case or not
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'intent_labels': [NeuralType(('B'), LabelsType())],
            'slot_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        input_file: str,
        slot_file: str,
        num_intents: int,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        pad_label: int = 128,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        do_lower_case: bool = False,
    ):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        with open(slot_file, 'r') as f:
            slot_lines = f.readlines()

        with open(input_file, 'r') as f:
            input_lines = f.readlines()[1:]

        assert len(slot_lines) == len(input_lines)

        dataset = list(zip(slot_lines, input_lines))

        if num_samples > 0:
            dataset = dataset[:num_samples]

        raw_slots, queries, raw_intents = [], [], []
        for slot_line, input_line in dataset:
            raw_slots.append([int(slot) for slot in slot_line.strip().split()])
            parts = input_line.strip().split("\t")[1:][0]
            parts = list(map(int, parts.split(",")))
            parts = [1 if label in parts else 0 for label in range(num_intents)]
            raw_intents.append(tuple(parts))
            tokens = input_line.strip().split("\t")[0].split()
            query = ' '.join(tokens)
            if do_lower_case:
                query = query.lower()
            queries.append(query)

        features = get_features(
            queries,
            max_seq_length,
            tokenizer,
            pad_label=pad_label,
            raw_slots=raw_slots,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
        )

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.all_intents = raw_intents
