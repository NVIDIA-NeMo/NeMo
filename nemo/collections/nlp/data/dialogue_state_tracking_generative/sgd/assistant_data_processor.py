# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/data_utils.py
"""


import os

from nemo.collections.nlp.data.dialogue_state_tracking_generative.sgd.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue_state_tracking_generative.sgd.input_example import DialogueInputExample

__all__ = ['DialogueAssistantDataProcessor']


class DialogueAssistantInputExample(DialogueInputExample):
    """
    Template for DialogueAssistantInputExample
    {
        
        "utterance": <utterance>,
        "labels": {
            "service": <service>,
            "intent": <intent>,
            "slots": {
                "<slot-name1>": [<slot-value1>, <slot-value2>],
                "<slot-name2>": [<slot-value2>],
            }
        },
        "label_positions":{
            "slots": {
                "<slot-name1>": { 
                    # note for the Assistant dataset, start and end are word positions rather than char position
                    # these are whitespace-delimited word positions rather than tokenization-specific sub-word tokens.
                    "exclusive_end": 3, 
                    "slot": "restaurant_name",
                    "start": 1 
                },
            }
        },
        "possible_labels": {
            "service": [<service1>, <service2>, ...],
            "intent": [<intent1>, <intent2>, ...],
            "slots": {
                # all slots for categorical variables
                # empty list for extractive slots
                # Assistant only support extractive slots
                "<slot-name1>": [],
                "<slot-name2>": [],
            }
        }
    }
    """


class DialogueAssistantDataProcessor(DialogueDataProcessor):
    """Data Processor for Assistant dialogues."""

    def __init__(self, data_dir: str, tokenizer: object):
        """
        Constructs SGDDataProcessor
        Args:
            data_dir: path to data directory
            tokenizer: tokenizer object
        """
        self.data_dir = data_dir
        self._tokenizer = tokenizer
        self.intents = self.open_file("dict.intents.csv")
        self.slots = self.open_file("dict.slots.csv")
        self.services = sorted(list(set([intent.split('_')[0] for intent in self.intents])))
        self.empty_slot_id = str(len(self.slots) - 1)

    def open_file(self, filename):
        """
        Reads file into a list
        """
        filename = os.path.join(self.data_dir, filename)
        with open(filename, "r", encoding="UTF-8") as f:
            lines = [i.strip() for i in f.readlines()]
        return lines

    def get_continuous_slots(self, slot_ids):
        """
        Extract continuous spans of slot_ids
        Args:
            Slot: list of int representing slot of each word token
            For instance, 54 54 54 54 54 54 54 54 18 54 44 44 54 46 46 54 12 
            Corresponds to "please set an alarm clock for my next meeting with the team at three pm next friday"
            Except for the empty_slot_id (54 in this case), we hope to extract the continuous spans of tokens,
            each containing a start position and an exclusive end position
            E.g {18: [9, 10], 44: [11,13], 46: [14, 16], 12: [17, 18]}
        """
        slot_id_stack = []
        position_stack = []
        for i, slot_id in enumerate(slot_ids):
            if not slot_id_stack or slot_id != slot_id_stack[-1]:
                slot_id_stack.append(slot_id)
                position_stack.append([])
            position_stack[-1].append(i)

        slot_id_to_start_and_exclusive_end = {
            slot_id_stack[i]: [position_stack[i][0], position_stack[i][-1] + 1]
            for i in range(len(position_stack))
            if slot_id_stack[i] != self.empty_slot_id
        }

        return slot_id_to_start_and_exclusive_end

    def get_dialog_examples(self, dataset_split: str):
        """
        Process raw files into DialogueInputExample
        Args: 
            dataset_split: {train, dev, test}
        For the assistant dataset, there is no explicit dev set (instead uses the test set as the dev set)
        Therefore, this function creates a dev set and a new train set from the train set.
        This is done by taking every 10th example and putting it into the dev set,
        with all other examples going into the new train set.
        """
        examples = []

        dataset_split_print = {"train": "train", "dev": "train", "test": "test"}

        raw_examples_intent = self.open_file("{}.tsv".format(dataset_split_print[dataset_split]))
        raw_examples_intent = raw_examples_intent[1:]
        raw_examples_slots = self.open_file("{}_slots.tsv".format(dataset_split_print[dataset_split]))

        if dataset_split in ["train", "dev"]:
            train_idx = []
            dev_idx = []
            for idx in range(len(raw_examples_intent)):
                if idx % 10 == 0:
                    dev_idx.append(idx)
                else:
                    train_idx.append(idx)

        if dataset_split == "train":
            raw_examples_intent = [raw_examples_intent[idx] for idx in train_idx]
            raw_examples_slots = [raw_examples_slots[idx] for idx in train_idx]
        elif dataset_split == "dev":
            raw_examples_intent = [raw_examples_intent[idx] for idx in dev_idx]
            raw_examples_slots = [raw_examples_slots[idx] for idx in dev_idx]

        for i in range(len(raw_examples_intent)):
            utterance, intent_id = raw_examples_intent[i].split('\t')
            slot_ids = raw_examples_slots[i].split()
            utterance_tokens = utterance.split()
            intent = self.intents[int(intent_id)]
            slot_id_to_start_and_exclusive_end = self.get_continuous_slots(slot_ids)
            slot_to_start_and_exclusive_end = {
                self.slots[int(slot_id)]: position for slot_id, position in slot_id_to_start_and_exclusive_end.items()
            }
            slot_to_words = {
                slot: ' '.join(utterance_tokens[position[0] : position[1]])
                for slot, position in slot_to_start_and_exclusive_end.items()
            }
            input_example = {
                "utterance": utterance,
                "labels": {"service": intent.split('_')[0], "intent": intent, "slots": slot_to_words},
                "label_positions": {
                    "slots": {
                        slot: {"start": position[0], "exclusive_end": position[1], "slot": slot,}
                        for slot, position in slot_to_start_and_exclusive_end.items()
                    }
                },
                "possible_labels": {
                    "service": self.services,
                    "intent": self.intents,
                    "slots": {
                        # this dataset does not support categorical slots (i.e. only extractive slots)
                        # therefore use empty list for all values
                        slot: []
                        for slot in self.slots
                    },
                },
            }
            example = DialogueInputExample(input_example)
            examples.append(example)
        return examples

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_dialog_examples("train")

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dialog_examples("dev")

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        return self.get_dialog_examples("test")
