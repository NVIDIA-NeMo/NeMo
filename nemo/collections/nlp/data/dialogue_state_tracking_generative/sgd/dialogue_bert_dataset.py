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
https://github.com/google-research/google-research/blob/master/schema_guided_dst
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np

from nemo.collections.nlp.data.data_utils import get_stats
from nemo.collections.nlp.data.dialogue_state_tracking_generative.sgd.input_example import SGDInputExample
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['DialogueSGDBERTDataset', 'DialogueBERTDataset']


class DialogueBERTDataset(Dataset):

    """
    Creates a dataset to use for the task of joint intent
    and slot classification with pretrained model.

    For a dataset to use during inference without labels, see
    IntentSlotDataset.
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
            'intent_labels': NeuralType(('B'), LabelsType()),
            'slot_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for dialogues
            tokenizer: tokenizer to split text into sub-word tokens
            cfg: config dict for dataset
        """
        self.cfg = cfg
        self.all_possible_labels = dialogues_processor.intents
        self.label_to_label_id = {self.all_possible_labels[i]: i for i in range(len(self.all_possible_labels))}
        self.all_possible_slots = dialogues_processor.slots
        self.slot_name_to_slot_id = {self.all_possible_slots[i]: i for i in range(len(self.all_possible_slots))}
        self.empty_slot_name = self.all_possible_slots[-1]

        self.features = dialogues_processor.get_dialog_examples(dataset_split)
        self.features = self.features if self.cfg.num_samples == -1 else self.features[: self.cfg.num_samples]

        queries = [feature.data["utterance"] for feature in self.features]
        if self.cfg.do_lowercase:
            queries = [query.lower() for query in queries]
        intents = [self.label_to_label_id[feature.data["labels"]["intent"]] for feature in self.features]
        word_level_slots = [self.convert_slot_position_to_slot_ids(feature.data) for feature in self.features]

        features = DialogueBERTDataset.get_features(
            queries,
            self.cfg.max_seq_length,
            tokenizer,
            pad_label=self.cfg.pad_label,
            word_level_slots=word_level_slots,
            ignore_extra_tokens=self.cfg.ignore_extra_tokens,
            ignore_start_end=self.cfg.ignore_start_end,
        )

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.all_slots = features[5]
        self.all_intents = intents

    def convert_slot_position_to_slot_ids(self, feature):
        slot_ids = [self.slot_name_to_slot_id[self.empty_slot_name] for i in range(len(feature["utterance"].split()))]
        slot_name_to_positions = feature["label_positions"]["slots"]

        for slot_name in slot_name_to_positions:
            slot_id = self.slot_name_to_slot_id[slot_name]
            start = slot_name_to_positions[slot_name]["start"]
            exclusive_end = slot_name_to_positions[slot_name]["exclusive_end"]
            for to_replace_position in range(start, min(exclusive_end, len(slot_ids))):
                slot_ids[to_replace_position] = slot_id

        return slot_ids

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            self.all_intents[idx],
            np.array(self.all_slots[idx]),
        )

    @staticmethod
    def truncate_and_pad(
        max_seq_length,
        ignore_start_end,
        with_label,
        pad_label,
        tokenizer,
        all_slots,
        all_subtokens,
        all_input_mask,
        all_loss_mask,
        all_subtokens_mask,
        all_input_ids,
        all_segment_ids,
    ):

        too_long_count = 0

        for i, subtokens in enumerate(all_subtokens):
            if len(subtokens) > max_seq_length:
                subtokens = [tokenizer.cls_token] + subtokens[-max_seq_length + 1 :]
                all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
                all_loss_mask[i] = [1 - ignore_start_end] + all_loss_mask[i][-max_seq_length + 1 :]
                all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

                if with_label:
                    all_slots[i] = [pad_label] + all_slots[i][-max_seq_length + 1 :]
                too_long_count += 1

            all_input_ids.append([tokenizer.tokens_to_ids(t) for t in subtokens])

            if len(subtokens) < max_seq_length:
                extra = max_seq_length - len(subtokens)
                all_input_ids[i] = all_input_ids[i] + [0] * extra
                all_loss_mask[i] = all_loss_mask[i] + [0] * extra
                all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
                all_input_mask[i] = all_input_mask[i] + [0] * extra

                if with_label:
                    all_slots[i] = all_slots[i] + [pad_label] * extra

            all_segment_ids.append([0] * max_seq_length)

        logging.info(f'{too_long_count} are longer than {max_seq_length}')
        return (
            all_slots,
            all_subtokens,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_input_ids,
            all_segment_ids,
        )

    @staticmethod
    def get_features(
        queries,
        max_seq_length,
        tokenizer,
        pad_label=128,
        word_level_slots=None,
        ignore_extra_tokens=False,
        ignore_start_end=False,
    ):
        """
        Convert queries (utterance, intent label and slot labels) to BERT input format 
        """

        all_subtokens = []
        all_loss_mask = []
        all_subtokens_mask = []
        all_segment_ids = []
        all_input_ids = []
        all_input_mask = []
        sent_lengths = []
        all_slots = []

        with_label = word_level_slots is not None

        for i, query in enumerate(queries):
            words = query.strip().split()
            subtokens = [tokenizer.cls_token]
            loss_mask = [1 - ignore_start_end]
            subtokens_mask = [0]
            if with_label:
                slots = [pad_label]

            for j, word in enumerate(words):
                word_tokens = tokenizer.text_to_tokens(word)

                # to handle emojis that could be neglected during tokenization
                if len(word.strip()) > 0 and len(word_tokens) == 0:
                    word_tokens = [tokenizer.ids_to_tokens(tokenizer.unk_id)]

                subtokens.extend(word_tokens)
                # mask all sub-word tokens except the first token in a word
                # use the label for the first sub-word token as the label for the entire word to eliminate need for disambiguation
                loss_mask.append(1)
                loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))

                subtokens_mask.append(1)
                subtokens_mask.extend([0] * (len(word_tokens) - 1))

                if with_label:
                    slots.extend([word_level_slots[i][j]] * len(word_tokens))

            subtokens.append(tokenizer.sep_token)
            loss_mask.append(1 - ignore_start_end)
            subtokens_mask.append(0)
            sent_lengths.append(len(subtokens))
            all_subtokens.append(subtokens)
            all_loss_mask.append(loss_mask)
            all_subtokens_mask.append(subtokens_mask)
            all_input_mask.append([1] * len(subtokens))
            if with_label:
                slots.append(pad_label)
                all_slots.append(slots)
        max_seq_length_data = max(sent_lengths)
        max_seq_length = min(max_seq_length, max_seq_length_data) if max_seq_length > 0 else max_seq_length_data
        logging.info(f'Setting max length to: {max_seq_length}')
        get_stats(sent_lengths)

        # truncate and pad samples
        (
            all_slots,
            all_subtokens,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_input_ids,
            all_segment_ids,
        ) = DialogueBERTDataset.truncate_and_pad(
            max_seq_length,
            ignore_start_end,
            with_label,
            pad_label,
            tokenizer,
            all_slots,
            all_subtokens,
            all_input_mask,
            all_loss_mask,
            all_subtokens_mask,
            all_input_ids,
            all_segment_ids,
        )

        # log examples for debugging
        logging.debug("*** Some Examples of Processed Data ***")
        for i in range(min(len(all_input_ids), 5)):
            logging.debug("i: %s" % (i))
            logging.debug("subtokens: %s" % " ".join(list(map(str, all_subtokens[i]))))
            logging.debug("loss_mask: %s" % " ".join(list(map(str, all_loss_mask[i]))))
            logging.debug("input_mask: %s" % " ".join(list(map(str, all_input_mask[i]))))
            logging.debug("subtokens_mask: %s" % " ".join(list(map(str, all_subtokens_mask[i]))))
            if with_label:
                logging.debug("slots_label: %s" % " ".join(list(map(str, all_slots[i]))))

        return (all_input_ids, all_segment_ids, all_input_mask, all_loss_mask, all_subtokens_mask, all_slots)


class DialogueSGDBERTDataset(Dataset):
    '''
    Dataset Class 
        1. Performs Model-dependent (but Data-independent) operations (tokenization etc)
        2. This can allow the same model preprocessing for multiple datasources
        3. Users can configurate which labels to use for modelling 
            (e.g. intent classification, slot filling or both together etc)
    '''

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, schemas, schema_config, cfg):
        """ Constructor
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for SGD dialogues
        """
        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.schema_config = schema_config
        self.dialogues_processor = dialogues_processor
        self.cfg = cfg
        self.subsample = self.dialogues_processor._subsample

        dial_file = f"{dialogues_processor._task_name}_{dataset_split}_examples_bert.processed"
        self.dial_file = os.path.join(self.cfg.data_dir, dial_file)
        if self.cfg.use_cache and os.path.exists(self.dial_file):
            self.load_features()
        else:
            self.process_features()
            self.save_features()

    def load_features(self):
        with open(self.dial_file, "rb") as f:
            self.features = np.load(f, allow_pickle=True)

    def process_features(self):
        self.features = []
        self.raw_features = self.dialogues_processor.get_dialog_examples(self.dataset_split)
        for idx in range(len(self.raw_features)):
            self.bert_process_one_sample(idx)

    def save_features(self):
        with open(self.dial_file, "wb") as f:
            np.save(f, self.features)

    def _tokenize(self, utterance: str):
        """
        Tokenize the utterance

        Args:
            utterance: A string containing the utterance to be tokenized.

        Returns:
            bert_tokens: A list of tokens obtained by word-piece tokenization of the
                utterance.
            alignments: A dict mapping indices of characters corresponding to start
                and end positions of words (not subwords) to corresponding indices in
                bert_tokens list.
            inverse_alignments: A list of size equal to bert_tokens. Each element is a
                tuple containing the index of the starting and inclusive ending
                character of the word corresponding to the subword. This list is used
                during inference to map word-piece indices to spans in the original
                utterance.
        """
        # utterance = tokenization.convert_to_unicode(utterance)

        # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
        # direct concatenation of all the tokens in the sequence will be the
        # original string.
        tokens = DialogueSGDBERTDataset._naive_tokenize(utterance)
        # ['I', ' ', 'am', ' ', 'feeling', ' ', 'hungry', ' ', 'so', ' ', 'I', ' ', 'would', ' ', 'like', ' ', 'to', ' ', 'find', ' ', 'a', ' ', 'place', ' ', 'to', ' ', 'eat', '.']
        # Filter out empty tokens and obtain aligned character index for each token.
        alignments = {}
        char_index = 0
        bert_tokens = (
            []
        )  # ['I', 'am', 'feeling', 'hungry', 'so', 'I', 'would', 'like', 'to', 'find', 'a', 'place', 'to', 'eat', '.']
        # These lists store inverse alignments to be used during inference.
        bert_tokens_start_chars = []
        bert_tokens_end_chars = []
        for token in tokens:
            if token.strip():
                subwords = self.tokenizer.text_to_tokens(token)
                # Store the alignment for the index of starting character and the
                # inclusive ending character of the token.
                alignments[char_index] = len(bert_tokens)
                bert_tokens_start_chars.extend([char_index] * len(subwords))
                bert_tokens.extend(subwords)
                # The inclusive ending character index corresponding to the word.
                inclusive_char_end = char_index + len(token) - 1
                alignments[inclusive_char_end] = len(bert_tokens) - 1
                bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
            char_index += len(token)
        inverse_alignments = list(zip(bert_tokens_start_chars, bert_tokens_end_chars))
        return bert_tokens, alignments, inverse_alignments

    @classmethod
    def _naive_tokenize(cls, s: str):
        """
        Tokenizes a string, separating words, spaces and punctuations.
        Args:
            s: a string
        Returns:
            seq_tok: list of words, spaces and punctuations from the string
        """
        # Spaces and punctuation marks are all retained, i.e. direct concatenation
        # of all the tokens in the sequence will be the original string.
        seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
        return seq_tok

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        ex = self.features[idx]

        return (
            np.array(ex.example_id_num),
            np.array(ex.example_id_num[-1]),  # service_id
            np.array(ex.utterance_ids),
            np.array(ex.utterance_segment),
            np.array(ex.utterance_mask, dtype=np.long),
            np.array(ex.intent_status, dtype=np.float32),
            np.array(ex.requested_slot_status, dtype=np.float32),
            np.array(ex.categorical_slot_status),
            np.array(ex.categorical_slot_value_status, dtype=np.float32),
            np.array(ex.noncategorical_slot_status),
            np.array(ex.noncategorical_slot_value_start),
            np.array(ex.noncategorical_slot_value_end),
            np.array(ex.start_char_idx),  # noncat_alignment_start
            np.array(ex.end_char_idx),  # noncat_alignment_end
            np.array(ex.task_mask),  # noncat_alignment_end
        )

    def bert_process_one_sample(self, idx):
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

        ex = self.raw_features[idx].data
        example_id_num = ex["example_id_num"]
        example_id = ex["example_id"]
        user_utterance = ex["utterance"]
        system_utterance = ex["system_utterance"]
        service = ex["labels"]["service"]
        schemas = self.schemas
        state_update = ex["labels"]["slots"]
        system_slots = ex["system_slots"]

        user_tokens, user_alignments, user_inv_alignments = self._tokenize(user_utterance)
        system_tokens, system_alignments, system_inv_alignments = self._tokenize(system_utterance)
        system_user_utterance = system_utterance + ' ' + user_utterance
        system_user_tokens, system_user_alignments, system_user_inv_alignments = self._tokenize(system_user_utterance)
        examples = []

        base_example = SGDInputExample(schema_config=self.schema_config, tokenizer=self.tokenizer)
        base_example.service_schema = self.schemas.get_service_schema(service)
        base_example.service_id = example_id_num[-1]

        base_example.example_id = example_id
        base_example.example_id_num = example_id_num

        for model_task in range(self.schema_config["NUM_TASKS"]):
            if model_task == 0:
                for intent_id, intent in enumerate(schemas.get_service_schema(service).intents):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1
                    task_example.intent_id = intent_id
                    task_example.example_id += f"-{model_task}-{intent_id}-0"
                    task_example.example_id_num.extend([model_task, intent_id, 0])
                    intent_description = (
                        intent + " " + self.schemas.get_service_schema(service).intent_descriptions[intent]
                    )
                    intent_tokens, intent_alignments, intent_inv_alignments = self._tokenize(intent_description)
                    task_example.add_utterance_features(
                        intent_tokens,
                        intent_inv_alignments,
                        system_user_tokens,
                        system_user_inv_alignments,
                        intent_description,
                        system_user_utterance,
                    )

                    task_example.add_intents(ex)
                    examples.append(task_example)

            if model_task == 1:
                for slot_id, slot in enumerate(schemas.get_service_schema(service).slots):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1
                    task_example.requested_slot_id = slot_id
                    task_example.example_id += f"-{model_task}-{slot_id}-0"
                    task_example.example_id_num.extend([model_task, slot_id, 0])
                    slot_description = slot + " " + self.schemas.get_service_schema(service).slot_descriptions[slot]
                    slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                    task_example.add_utterance_features(
                        slot_tokens,
                        slot_inv_alignments,
                        user_tokens,
                        user_inv_alignments,
                        slot_description,
                        user_utterance,
                    )

                    task_example.add_requested_slots(ex)
                    examples.append(task_example)

            if model_task == 2:
                off_slots = []
                on_slots = []
                for slot_id, slot in enumerate(schemas.get_service_schema(service).categorical_slots):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1

                    # assert task_example.task_mask == [0, 0, 1, 0, 0, 0]
                    task_example.categorical_slot_id = slot_id
                    task_example.example_id += f"-{model_task}-{slot_id}-0"
                    task_example.example_id_num.extend([model_task, slot_id, 0])
                    slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                    slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                    task_example.add_utterance_features(
                        slot_tokens,
                        slot_inv_alignments,
                        system_user_tokens,
                        system_user_inv_alignments,
                        slot_description,
                        system_user_utterance,
                    )
                    task_example.add_categorical_slots(state_update)

                    if task_example.categorical_slot_status == 0:
                        off_slots.append(task_example)
                    else:
                        on_slots.append(task_example)
                        examples.append(task_example)
                    old_example = task_example

                    for value_id, value in enumerate(
                        schemas.get_service_schema(service).get_categorical_slot_values(slot)
                    ):
                        if self.dataset_split != 'train' or task_example.categorical_slot_status == 1:
                            task_example = old_example.make_copy_of_categorical_features()
                            task_example.task_mask[3] = 1
                            # assert task_example.task_mask == [0, 0, 0, 1, 0, 0]
                            task_example.categorical_slot_id = slot_id
                            task_example.categorical_slot_value_id = value_id
                            task_example.example_id = base_example.example_id + f"-3-{slot_id}-{value_id}"
                            task_example.example_id_num = base_example.example_id_num + [3, slot_id, value_id]
                            slot_description = slot + " " + value  # add slot description
                            slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                            task_example.add_utterance_features(
                                slot_tokens,
                                slot_inv_alignments,
                                system_user_tokens,
                                system_user_inv_alignments,
                                slot_description,
                                system_user_utterance,
                            )
                            task_example.add_categorical_slots(state_update)
                            assert task_example.categorical_slot_status == old_example.categorical_slot_status
                            examples.append(task_example)

                if self.dataset_split == 'train' and self.subsample:
                    num_on_slots = len(on_slots)
                    examples.extend(
                        np.random.choice(off_slots, replace=False, size=min(max(num_on_slots, 1), len(off_slots)))
                    )
                else:
                    examples.extend(off_slots)

            if model_task == 4:  # noncat slot status
                off_slots = []
                on_slots = []
                for slot_id, slot in enumerate(schemas.get_service_schema(service).non_categorical_slots):
                    task_example = base_example.make_copy()
                    task_example.task_mask[model_task] = 1
                    # assert task_example.task_mask == [0, 0, 0, 0, 1, 0]
                    task_example.noncategorical_slot_id = slot_id
                    task_example.example_id += f"-{model_task}-{slot_id}-0"
                    task_example.example_id_num.extend([model_task, slot_id, 0])
                    slot_description = slot + " " + schemas.get_service_schema(service).slot_descriptions[slot]
                    slot_tokens, slot_alignments, slot_inv_alignments = self._tokenize(slot_description)
                    task_example.add_utterance_features(
                        slot_tokens,
                        slot_inv_alignments,
                        system_user_tokens,
                        system_user_inv_alignments,
                        slot_description,
                        system_user_utterance,
                    )

                    user_span_boundaries = self._find_subword_indices(
                        state_update,
                        user_utterance,
                        ex["label_positions"]["slots"],
                        user_alignments,
                        user_tokens,
                        2 + len(slot_tokens) + len(system_tokens),
                    )

                    if system_slots is not None:
                        system_span_boundaries = self._find_subword_indices(
                            state_update,
                            system_utterance,
                            system_slots,
                            system_alignments,
                            system_tokens,
                            2 + len(slot_tokens),
                        )
                    else:
                        system_span_boundaries = {}

                    task_example.add_noncategorical_slots(state_update, user_span_boundaries, system_span_boundaries)
                    if task_example.noncategorical_slot_status == 0:
                        off_slots.append(task_example)
                    else:
                        on_slots.append(task_example)
                        examples.append(task_example)

                    if self.dataset_split != 'train' or task_example.noncategorical_slot_status == 1:
                        task_example = task_example.make_copy_of_non_categorical_features()
                        task_example.task_mask[5] = 1
                        # assert task_example.task_mask == [0, 0, 0, 0, 0, 1]
                        task_example.example_id = base_example.example_id + f"-5-{slot_id}-0"
                        task_example.example_id_num = base_example.example_id_num + [5, slot_id, 0]
                        examples.append(task_example)

                if self.dataset_split == 'train' and self.subsample:
                    num_on_slots = len(on_slots)
                    examples.extend(
                        np.random.choice(off_slots, replace=False, size=min(max(num_on_slots, 1), len(off_slots)))
                    )
                else:
                    examples.extend(off_slots)

        for example in examples:
            self.features.append(example)

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
            for key, slot_span in char_slot_spans.items():
                # print(key, slot, slot_span, char_slot_spans)
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
