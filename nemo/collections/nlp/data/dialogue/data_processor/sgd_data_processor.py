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
import collections
import json
import os
import pickle
import re
from typing import List

from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample
from nemo.collections.nlp.data.dialogue.sgd.schema import Schema
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning
from nemo.utils.get_rank import is_global_rank_zero

__all__ = ['DialogueSGDDataProcessor']

FILE_RANGES = {
    "sgd_single_domain": {"train": range(1, 44), "dev": range(1, 8), "test": range(1, 12)},
    "sgd_multi_domain": {"train": range(44, 128), "dev": range(8, 21), "test": range(12, 35)},
    "sgd_all": {"train": range(1, 128), "dev": range(1, 21), "test": range(1, 35)},
    "sgd_all_single": {"train": range(1, 128), "dev": range(1, 8), "test": range(1, 12)},
    "multiwoz": {"train": range(1, 18), "dev": range(1, 3), "test": range(1, 3)},
    "debug_sample": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
}


class DialogueSGDDataProcessor(DialogueDataProcessor):
    """Data Processor for SGD dialogues.

    More information at https://arxiv.org/abs/1909.05855

    ***Downloading the dataset***
        #   git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git

    ***Data format***
    SGD data comes with a JSON schema file and dialogue files for each dataset split.

    In the following we will show an example for a service entry in the schema file.
    * service_name
    * description
    * slots
        * name
        * description
        * is_categorical
        * possible values
    * intents
        * name
        * description
        * required_slots (not used)
        * is_transactional (not used)
        * optional_slots (not used)
        * result_slots (not used)


    In the following we will show an example for a dialogue.
    * dialogue_id
    * services
    * turns
        * frames
            * actions
                * act
                * slot
                * values
            * service
            * slots
                * exclusive_end
                * slot
                * start
            * state
                * active_intent
                * requeste_slots
                * slot_values
        * speaker - [USER, SYSTEM]
        * utterance

    """

    def __init__(
        self,
        data_dir: str,
        dialogues_example_dir: str,
        tokenizer: object,
        cfg=None,
    ):
        """
        Constructs DialogueSGDDataProcessor
        Args:
            data_dir: path to data directory
            dialogues_example_dir: path to store processed dialogue examples
            tokenizer: tokenizer object
            cfg: cfg container for dataset
        """
        # deprecation warning
        deprecated_warning("DialogueSGDDataProcessor")

        self.data_dir = data_dir
        self.cfg = cfg

        self._task_name = self.cfg.task_name  # e.g. "sgd_single_domain"
        self._subsample = self.cfg.subsample

        all_schema_json_paths = []
        for dataset_split in ['train', 'test', 'dev']:
            all_schema_json_paths.append(os.path.join(self.cfg.data_dir, dataset_split, "schema.json"))
        self.schemas = Schema(all_schema_json_paths)

        self.schema_config = {
            "MAX_NUM_CAT_SLOT": self.cfg.max_num_cat_slot,
            "MAX_NUM_NONCAT_SLOT": self.cfg.max_num_noncat_slot,
            "MAX_NUM_VALUE_PER_CAT_SLOT": self.cfg.max_value_per_cat_slot,
            "MAX_NUM_INTENT": self.cfg.max_num_intent,
            "NUM_TASKS": self.cfg.num_tasks,
            "MAX_SEQ_LENGTH": self.cfg.max_seq_length,
        }

        train_file_range = FILE_RANGES[self._task_name]["train"]
        dev_file_range = FILE_RANGES[self._task_name]["dev"]
        test_file_range = FILE_RANGES[self._task_name]["test"]

        self._file_ranges = {
            "train": train_file_range,
            "dev": dev_file_range,
            "test": test_file_range,
        }

        self._seen_services = {
            "train": set(),
            "dev": set(),
            "test": set(),
        }

        self._tokenizer = tokenizer

        self._dialogues_example_dir = dialogues_example_dir

        self.dial_files = {}

        # slots_relation_list.np would contain the candidate list of slots for each (service, slot) which would be
        # looked into when a switch between two services happens in the dialogue and we can not find any value for a slot in the current user utterance.
        # This file would get generated from the dialogues in the training set.
        self.slots_relation_file = os.path.join(
            dialogues_example_dir, f"{self._task_name}_train_slots_relation_list.np"
        )
        for dataset in ["train", "dev", "test"]:
            # Process dialogue files
            dial_file = f"{self._task_name}_{dataset}_examples.json"
            dial_file = os.path.join(dialogues_example_dir, dial_file)
            self.dial_files[(self._task_name, dataset)] = dial_file

            dialog_paths = DialogueSGDDataProcessor.get_dialogue_files(data_dir, dataset, self._task_name)
            dialogs = DialogueSGDDataProcessor.load_dialogues(dialog_paths)
            for dialog in dialogs:
                self._seen_services[dataset].update(set(dialog['services']))

        if is_global_rank_zero():
            overwrite_dial_files = not self.cfg.use_cache
            self.save_dialog_examples(overwrite_dial_files=overwrite_dial_files)

    def save_dialog_examples(self, overwrite_dial_files: bool):
        """
        Preprocesses dialogues and saves to disk.
        Args:
            overwrite_dial_files: whether or not to overwrite saved file if already exists
        """
        for dataset in ["train", "dev", "test"]:
            dial_file = self.dial_files[(self._task_name, dataset)]
            if not os.path.exists(dial_file) or overwrite_dial_files:
                logging.info(f"Start generating the dialogue examples for {dataset} dataset.")
                if not os.path.exists(self._dialogues_example_dir):
                    os.makedirs(self._dialogues_example_dir)
                dial_examples, slots_relation_list = self._generate_dialog_examples(
                    dataset, self.schemas, self._subsample
                )

                with open(dial_file, "w", encoding="UTF-8") as f:
                    json.dump([i.data for i in dial_examples], f)

                if dataset == "train":
                    with open(self.slots_relation_file, "wb") as f:
                        pickle.dump(slots_relation_list, f)
                    logging.info(f"The slot carry-over list for train set is stored at {self.slots_relation_file}")

                logging.info(f"The dialogue examples for {dataset} dataset saved at {dial_file}")
                logging.info(f"Finish generating the dialogue examples for {dataset} dataset.")

    # common interface for Data Processor
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_dialog_examples("train")

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dialog_examples("dev")

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        return self.get_dialog_examples("test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_dialog_examples(self, dataset_split: str) -> List[object]:
        """
        Loads preprocessed dialogue examples from disk.
        Args:
            dataset_split: dataset split
        Returns:
            dial_examples:  list of InputExample's.
        """
        if (self._task_name, dataset_split) not in self.dial_files or not os.path.exists(
            self.dial_files[(self._task_name, dataset_split)]
        ):
            raise ValueError(
                f"{dataset_split} dialogue examples were not processed for {self._task_name} task. Re-initialize SGDDataProcessor and add {dataset_split} dataset split to datasets arg."
            )
        dial_file = self.dial_files[(self._task_name, dataset_split)]
        logging.info(f"Loading dialogue examples from {dial_file}.")

        with open(dial_file, "rb") as f:
            dial_examples = json.load(f)
            dial_examples = [DialogueInputExample(i) for i in dial_examples]
        if not os.path.exists(self.slots_relation_file):
            raise ValueError(
                f"Slots relation file {self.slots_relation_file} does not exist. It is needed for the carry-over mechanism of state tracker for switches between services."
            )
        if os.path.getsize(self.slots_relation_file) > 0:
            with open(self.slots_relation_file, "rb") as f:
                self.schemas._slots_relation_list = pickle.load(f)
            logging.info(
                f"Loaded the slot relation list for value carry-over between services from {self.slots_relation_file}."
            )

        return dial_examples

    def get_seen_services(self, dataset_split: str):
        """
        Returns list of seen services, i.e. both in given and training split
        Args:
            dataset_split: data split
        Returns:
            seen_services: list of seen services
        """
        seen_services = self._seen_services[dataset_split]
        return seen_services

    def _generate_dialog_examples(self, dataset_split: str, schemas: object, subsample: bool):
        """
        Returns a list of `InputExample`s of the data splits' dialogues.
        Args:
            dataset_split: data split, can be "train", "dev", or "test".
            schemas: schema for all services of all datasets
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
        """
        logging.info(f'Creating examples and slot relation list from the dialogues started...')
        dialog_paths = [
            os.path.join(self.data_dir, dataset_split, "dialogues_{:03d}.json".format(i))
            for i in self._file_ranges[dataset_split]
        ]
        dialogs = DialogueSGDDataProcessor.load_dialogues(dialog_paths)

        examples = []
        slot_carryover_candlist = collections.defaultdict(int)
        for dialog_idx, dialog in enumerate(dialogs):
            if dialog_idx % 1000 == 0:
                logging.info(f'Processed {dialog_idx} dialogues.')
            examples.extend(
                self._create_examples_from_dialog(dialog, schemas, dataset_split, slot_carryover_candlist, subsample)
            )

        slots_relation_list = collections.defaultdict(list)
        for slots_relation, relation_size in slot_carryover_candlist.items():
            if relation_size > 0:
                slots_relation_list[(slots_relation[0], slots_relation[1])].append(
                    (slots_relation[2], slots_relation[3], relation_size)
                )
                slots_relation_list[(slots_relation[2], slots_relation[3])].append(
                    (slots_relation[0], slots_relation[1], relation_size)
                )

        return examples, slots_relation_list

    def _create_examples_from_dialog(
        self, dialog: dict, schemas: object, dataset_split: str, slot_carryover_candlist: dict, subsample: bool
    ):
        """
        Create examples for every turn in the dialogue.
        Args:
            dialog: dialogue example
            schemas: schema for all services of all datasets
            dataset_split: data split
            slot_carryover_candlist: a dictionary to keep and count the number of carry-over cases between two slots from two different services
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
        """
        dialog_id = dialog["dialogue_id"]
        prev_states = {}
        examples = []
        for turn_idx, turn in enumerate(dialog["turns"]):
            # Generate an example for every frame in every user turn.
            if turn["speaker"] == "USER":
                user_utterance = turn["utterance"]
                user_frames = {f["service"]: f for f in turn["frames"]}
                if self.cfg.system_utterance == 'prev_turn':
                    if turn_idx > 0:
                        system_turn = dialog["turns"][turn_idx - 1]
                        system_utterance = system_turn["utterance"]
                        system_frames = {f["service"]: f for f in system_turn["frames"]}
                    else:
                        system_utterance = ""
                        system_frames = {}
                else:  # takes the system utterance of the next turn
                    system_turn = dialog["turns"][turn_idx + 1]
                    system_utterance = system_turn["utterance"]
                    system_frames = {f["service"]: f for f in system_turn["frames"]}

                turn_id = "{}-{}-{:02d}".format(dataset_split, dialog_id, turn_idx)
                turn_examples, prev_states, slot_carryover_values = self._create_examples_from_turn(
                    turn_id,
                    system_utterance,
                    user_utterance,
                    system_frames,
                    user_frames,
                    prev_states,
                    schemas,
                    subsample,
                )
                examples.extend(turn_examples)

                for value, slots_list in slot_carryover_values.items():
                    if value in ["True", "False"]:
                        continue
                    if len(slots_list) > 1:
                        for service1, slot1 in slots_list:
                            for service2, slot2 in slots_list:
                                if service1 == service2:
                                    continue
                                if service1 > service2:
                                    service1, service2 = service2, service1
                                    slot1, slot2 = slot2, slot1
                                slot_carryover_candlist[(service1, slot1, service2, slot2)] += 1
        return examples

    def _get_state_update(self, current_state: dict, prev_state: dict) -> dict:
        """
        Updates dialogue state
        Args:
            current_state: slot values pairs for the current dialogue turn
            prev_state: slot values pairs for the previous dialogue turns
        Returns:
            state_update: slot values pairs that are added/updated during the current dialogue turn
        """
        state_update = dict(current_state)
        for slot, values in current_state.items():
            if slot in prev_state and prev_state[slot][0] in values:
                # Remove the slot from state if its value didn't change.
                state_update.pop(slot)
        return state_update

    @staticmethod
    def convert_camelcase_to_lower(label):
        """Converts camelcase to lowercase with spaces e.g. 'HelloWorld' --> 'hello world'"""
        if label.lower() == "none":
            return "none"
        label = label.split("_")[0]
        tokens = re.findall('[A-Z][^A-Z]*', label)
        return ' '.join([token.lower() for token in tokens])

    def preprocess_intent(self, intent, schemas, service):
        if self.cfg.preprocess_intent_function == 'default':
            return intent
        elif self.cfg.preprocess_intent_function == 'lowercase':
            return DialogueSGDDataProcessor.convert_camelcase_to_lower(intent)
        elif self.cfg.preprocess_intent_function == 'description':
            return schemas.get_service_schema(service).intent_descriptions[intent]
        else:
            raise ValueError(
                'Only default, lowercase and description are allowed for model.dataset.preprocess_intent_function for SGD task'
            )

    def _create_examples_from_turn(
        self,
        turn_id: int,
        system_utterance: str,
        user_utterance: str,
        system_frames: dict,
        user_frames: dict,
        prev_states: dict,
        schemas: object,
        subsample: bool,
    ):
        """
        Creates an example for each frame in the user turn.
        Args:
            turn_id: turn number
            system_utterance: last system utterance
            user_utterance: lst user utterance
            system_frames: all system utterances and slot - slot value pairs
            user_frames: all user utterances and slot - slot value pairs
            prev_states: slot - slot value pairs from the previous turns
            schemas: schema for all services of all datasets
            subsample: whether to balance postive and negative samples in the dataset
        Returns:
            examples: a list of `InputExample`s.
            prev_states: updated dialogue state e.g. {'Restaurants_1': {'city': ['San Jose'], 'cuisine': ['American']}}
        """
        system_user_utterance = system_utterance + ' ' + user_utterance
        states = {}

        examples = []
        slot_carryover_values = collections.defaultdict(list)

        for service, user_frame in user_frames.items():

            state = user_frame["state"]["slot_values"]
            state_update = self._get_state_update(state, prev_states.get(service, {}))
            states[service] = state
            system_frame = system_frames.get(service, None)
            dataset_split, dialog_id, turn_id_ = turn_id.split('-')
            dialog_id_1, dialog_id_2 = dialog_id.split('_')
            example_id = f"{turn_id}-{service}"
            example_id_num = [
                int(dialog_id_1),
                int(dialog_id_2),
                int(turn_id_),
                schemas.get_service_id(service),
            ]
            intent = user_frames[service]["state"]['active_intent']
            all_possible_slots = schemas.get_service_schema(service).slots
            categorical_slots = schemas.get_service_schema(service).categorical_slots
            one_example = {
                "example_id": example_id,
                "example_id_num": example_id_num,
                "utterance": user_utterance,
                "system_utterance": system_utterance,
                "system_slots": (
                    {slot["slot"]: slot for slot in system_frame["slots"]} if system_frame is not None else None
                ),
                "system_actions": system_frame["actions"] if system_frame is not None else None,
                "labels": {
                    "service": service,
                    "intent": self.preprocess_intent(intent, schemas, service),
                    "slots": {slot: state[slot] for slot in state_update},
                },
                "label_positions": {"slots": {slot["slot"]: slot for slot in user_frames[service]["slots"]}},
                "possible_labels": {
                    "service": schemas.services,
                    "intent": [
                        self.preprocess_intent(intent, schemas, service)
                        for intent in schemas.get_service_schema(service).intents
                    ],
                    "slots": {
                        slot: (
                            schemas.get_service_schema(service).get_categorical_slot_values(slot)
                            if slot in categorical_slots
                            else []
                        )
                        for slot in all_possible_slots
                    },
                },
                "description": {
                    "service": schemas.get_service_schema(service).description,
                    "intent": schemas.get_service_schema(service).intent_descriptions[intent],
                    "slots": {
                        slot: schemas.get_service_schema(service).slot_descriptions[slot] for slot in state_update
                    },
                },
            }

            examples.append(DialogueInputExample(one_example))

            if service not in prev_states and int(turn_id_) > 0:
                for slot_name, values in state_update.items():
                    for value in values:
                        slot_carryover_values[value].append((service, slot_name))
                for prev_service, prev_slot_value_list in prev_states.items():
                    if prev_service == service:
                        continue
                    if prev_service in state:
                        prev_slot_value_list = state[prev_service]
                    for prev_slot_name, prev_values in prev_slot_value_list.items():
                        for prev_value in prev_values:
                            slot_carryover_values[prev_value].append((prev_service, prev_slot_name))

        return examples, states, slot_carryover_values

    def _find_subword_indices(
        self,
        slot_values: dict,
        utterance: str,
        char_slot_spans: dict,
        alignments: List[int],
        subwords: List[str],
        bias: int,
    ) -> dict:
        """
        Find indices for subwords corresponding to slot values.
        Args:
            slot_values: slot - slot value pairs
            utterance: utterance
            char_slot_spans: char - slot spans
            alignments: alignments
            subwords: subtokens mapping
            bias: offset
        Returns:
            span_boundaries: span boundaries
        """
        span_boundaries = {}
        for slot, values in slot_values.items():
            # Get all values present in the utterance for the specified slot.
            value_char_spans = {}
            for slot_span in char_slot_spans:
                if slot_span["slot"] == slot:
                    value = utterance[slot_span["start"] : slot_span["exclusive_end"]]
                    start_tok_idx = alignments[slot_span["start"]]
                    end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
                    if 0 <= start_tok_idx < len(subwords):
                        end_tok_idx = min(end_tok_idx, len(subwords) - 1)
                        value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
            for v in values:
                if v in value_char_spans:
                    span_boundaries[slot] = value_char_spans[v]
                    break
        return span_boundaries

    @classmethod
    def load_dialogues(cls, dialog_json_filepaths: List[str]) -> List[dict]:
        """
        Obtain the list of all dialogues from specified json files.
        Args:
            dialog_json_filepaths: list of json files
        Returns:
            dialogs: the list of all dialogues
        """
        dialogs = []
        for dialog_json_filepath in sorted(dialog_json_filepaths):
            with open(dialog_json_filepath, 'r', encoding="UTF-8") as f:
                dialogs.extend(json.load(f))
                f.close()
        return dialogs

    @classmethod
    def get_dialogue_files(cls, data_dir: str, dataset_split: str, task_name: str):
        """
        Obtain the list of all dialogue json files
        Args:
            data_dir: path to the data folder
            dataset_split: data split
            task_name: SGD task name, see keys of the FILE_RANGES
        Returns:
            dialog: the list of all dialogue json files paths
        """
        return [
            os.path.join(data_dir, dataset_split, 'dialogues_{:03d}.json'.format(fid))
            for fid in FILE_RANGES[task_name][dataset_split]
        ]
