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

import os

from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample
from nemo.utils.decorators import deprecated_warning

__all__ = ['DialogueAssistantDataProcessor']


class DialogueAssistantDataProcessor(DialogueDataProcessor):
    """Data Processor for Assistant dialogues."""

    def __init__(self, data_dir: str, tokenizer: object, cfg):
        """
        Constructs DialogueAssistantDataProcessor
        Args:
            data_dir: path to data directory
            tokenizer: tokenizer object
        """
        # deprecation warning
        deprecated_warning("DialogueAssistantDataProcessor")

        self.data_dir = data_dir
        self._tokenizer = tokenizer
        self.cfg = cfg
        self.intents = self.open_file("dict.intents.csv")
        if self.cfg.preprocess_intent_function == 'remove_domain':
            self.intents = [
                DialogueAssistantDataProcessor.normalize_zero_shot_intent(intent) for intent in self.intents
            ]
        self.slots = self.open_file("dict.slots.csv")
        (
            bio_slot_ids_to_unified_slot_ids,
            unified_slots,
        ) = DialogueAssistantDataProcessor.map_bio_format_slots_to_unified_slots(self.slots)
        self.slots = unified_slots

        self.bio_slot_ids_to_unified_slot_ids = bio_slot_ids_to_unified_slot_ids
        self.services = sorted(list(set([intent.split('_')[0] for intent in self.intents])))
        self.empty_slot_id = [str(idx) for idx, slot_name in enumerate(self.slots) if slot_name == "O"][0]

    @staticmethod
    def normalize_zero_shot_intent(label):
        label = label.split('.')[1]
        if label == 'nomatch':
            return 'no match'
        else:
            return label.replace('_', ' ')

    def open_file(self, filename):
        """
        Reads file into a list
        """
        filename = os.path.join(self.data_dir, filename)
        with open(filename, "r", encoding="UTF-8") as f:
            lines = [i.strip() for i in f.readlines()]
        return lines

    @staticmethod
    def get_continuous_slots(slot_ids, empty_slot_id, bio_slot_ids_to_unified_slot_ids):
        """
        Extract continuous spans of slot_ids

        To accomodate slots with distinct labels for B-label1 and I-label1,
        slot_id = self.bio_slot_ids_to_unified_slot_ids[slot_id] is called to map them both to label1

        Args:
            Slot: list of int representing slot of each word token
            For instance, 54 54 54 54 54 54 54 54 18 54 44 44 54 46 46 54 12
            Corresponds to "please set an alarm clock for my next meeting with the team at three pm next friday"
            Except for the empty_slot_id (54 in this case), we hope to extract the continuous spans of tokens,
            each containing a start position and an exclusive end position
            E.g {18: [9, 10], 44: [11, 13], 46: [14, 16], 12: [17, 18]}
        """
        slot_id_stack = []
        position_stack = []
        for i in range(len(slot_ids)):
            slot_id = slot_ids[i]

            slot_id = bio_slot_ids_to_unified_slot_ids[slot_id]

            if not slot_id_stack or slot_id != slot_id_stack[-1]:
                slot_id_stack.append(slot_id)
                position_stack.append([])
            position_stack[-1].append(i)

        slot_id_to_start_and_exclusive_end = {
            slot_id_stack[i]: [position_stack[i][0], position_stack[i][-1] + 1]
            for i in range(len(position_stack))
            if slot_id_stack[i] != empty_slot_id
        }

        return slot_id_to_start_and_exclusive_end

    @staticmethod
    def map_bio_format_slots_to_unified_slots(slots):
        """
        maps BIO format slots to unified slots (meaning that B-alarm_time and I-alarm_time both map to alarm_time)
        called even slots does not contain BIO, for unified interface
        in that case slots == unified_slots and bio_slot_ids_to_unified_slot_ids is an identity mapping i.e. {"0": "0", "1": "1"}
        """
        bio_slot_ids_to_unified_slot_ids = {}
        unified_slots = []
        unified_idx = -1
        for idx, slot in enumerate(slots):
            if slot.replace('I-', '').replace('B-', '') not in unified_slots:
                unified_idx += 1
                unified_slots.append(slot.replace('I-', '').replace('B-', ''))
            bio_slot_ids_to_unified_slot_ids[str(idx)] = str(unified_idx)
        return bio_slot_ids_to_unified_slot_ids, unified_slots

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
        # removes header of tsv file
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
            slot_id_to_start_and_exclusive_end = DialogueAssistantDataProcessor.get_continuous_slots(
                slot_ids, self.empty_slot_id, self.bio_slot_ids_to_unified_slot_ids
            )

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
                        slot: {
                            "start": position[0],
                            "exclusive_end": position[1],
                            "slot": slot,
                        }
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
