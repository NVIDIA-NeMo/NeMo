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

from typing import List

from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample
from nemo.utils import logging

__all__ = [
    'SGDInputExample',
    'STR_DONTCARE',
    'STATUS_OFF',
    'STATUS_ACTIVE',
    'STATUS_DONTCARE',
]


class DialogueSGDInputExample(DialogueInputExample):

    """
    Template for DialogueSGDInputExample

    Meant as a descriptor rather than to be instantiated

    Please instantiate using the base class 'DialogueInputExample' 

    {
        "example_id": <example_id>,
        "example_id_num": <example_id_num>,
        "utterance": <utterance>,
        "system_utterance": <system_utterance>,
        "system_slots": None or {
                    "<slot-name1>": {
                        "exclusive_end": 46,
                        "slot": "restaurant_name",
                        "start": 34
            },
        "system_actions": None or [{
                "act": "INFORM",
                "canonical_values": [
                  "2019-03-02"
                ],
                "slot": "date",
                "values": [
                  "March 2nd"
                ]
              }, ...]
        "labels": {
            "service": <service>,
            "intent": <intent>,
            "slots": {
                #only non-empty slots
                #most slot values are list of length 1 
                #but there are some of length 2 as both are accepted
                #e.g. 1930 and 7:30 pm
                "<slot-name1>": [<slot-value1>, <slot-value2>],
                "<slot-name2>": [<slot-value2>],
            }
        },
        "label_positions":{
            "slots": {
                "<slot-name1>": {
                    "exclusive_end": 46,
                    "slot": "restaurant_name",
                    "start": 34
              },
            }
        },
        "possible_labels": {
            "service": [<service1>, <service2>, ...],
            "intent": [<intent1>, <intent2>, ...],
            "slots": {
                #all slots including empty
                "<slot-name1>": [<slot-value1>, <slot-value2>, ...],
                "<slot-name2>": [<slot-value1>, <slot-value2>, ...],
            }
        },
        "description": {
            "service": <service description>,
            "intent": <intent description>,
            "slots": {
                #only non-empty slots
                "<slot-name1>": <slot-name1 description>,
                "<slot-name2>": <slot-name2 description>,
            }
        }
    }

    """


STR_DONTCARE = "dontcare"

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2


class SGDInputExample(object):
    """An example for training/inference."""

    def __init__(
        self,
        schema_config: dict,
        tokenizer: object,
        service_schema: object = None,
        example_id: str = "NONE",
        example_id_num: List[int] = [],
    ):
        """
        Constructs an InputExample.
        Args:
            schema_config: configuration
            tokenizer: tokenizer object
            service_schema: A ServiceSchema object wrapping the schema for the service
                corresponding to this example.
            example_id: Unique identifier for the example, like: 'train-1_00000-00-Restaurants_1'
            example_id_num: dialogue_id and turn_id combined and service id combined into a list of ints,
                like: [1, 0, 0, 18]
        """
        self.schema_config = schema_config
        self.service_schema = service_schema
        self.service_id = None
        if service_schema:
            self.service_id = service_schema.service_id
        self.example_id = example_id
        self.example_id_num = example_id_num
        self._max_seq_length = schema_config["MAX_SEQ_LENGTH"]
        self._tokenizer = tokenizer
        if self._tokenizer is None:
            raise ValueError("Must specify tokenizer")

        self.user_utterance = ''
        self.system_utterance = ''
        # The id of each subword in the vocabulary for BERT.
        self.utterance_ids = [0] * self._max_seq_length
        # Denotes the identity of the sequence. Takes values 0 (schema description) and 1 (system and user utterance).
        self.utterance_segment = [0] * self._max_seq_length
        # Mask which takes the value 0 for padded tokens and 1 otherwise.
        self.utterance_mask = [0] * self._max_seq_length
        # Start and inclusive end character indices in the original utterance
        # corresponding to the tokens. This is used to obtain the character indices
        # from the predicted subword indices during inference.
        # NOTE: A positive value indicates the character indices in the schema description
        # whereas a negative value indicates the character indices in the
        # utterance. The indices are offset by 1 to prevent ambiguity in the
        # 0 index, which could be in either the schema description or utterance by the
        # above convention. Now the 0 index corresponds to padded tokens.
        self.start_char_idx = [0] * self._max_seq_length
        self.end_char_idx = [0] * self._max_seq_length

        # Id of categorical slot present in the example or 0 if not present.
        self.categorical_slot_id = 0
        # Id of non categorical slot present in the example or 0 if not present.
        self.noncategorical_slot_id = 0
        # The status of categorical slot in the example.
        self.categorical_slot_status = STATUS_OFF
        # The status of non categorical slot in the example.
        self.noncategorical_slot_status = STATUS_OFF
        # Masks out tasks not represented by example
        self.task_mask = [0] * schema_config["NUM_TASKS"]

        # The index of the starting subword corresponding to the slot span
        # for a non-categorical slot value.
        self.noncategorical_slot_value_start = 0
        # The index of the ending (inclusive) subword corresponding to the slot span
        # for a non-categorical slot value.
        self.noncategorical_slot_value_end = 0

        # Id of categorical slot value present in the example or 0 if not present.
        self.categorical_slot_value_id = 0
        # The status of categorical slot value in the example.
        self.categorical_slot_value_status = STATUS_OFF
        # Id of requested slot present in the example or 0 if not present.
        self.requested_slot_id = 0
        # Takes value 1 if the corresponding slot is requested, 0 otherwise.
        self.requested_slot_status = STATUS_OFF

        # ID of intent present in the example.
        self.intent_id = 0
        # Takes value 1 if the intent is active, 0 otherwise.
        self.intent_status = STATUS_OFF

    @property
    def readable_summary(self):
        """Get a readable dict that summarizes the attributes of an InputExample."""
        seq_length = sum(self.utterance_mask)
        utt_toks = self._tokenizer.ids_to_tokens(self.utterance_ids[:seq_length])
        utt_tok_mask_pairs = list(zip(utt_toks, self.utterance_segment[:seq_length]))
        active_intent = (
            self.service_schema.get_intent_from_id(self.intent_id) if self.intent_status == STATUS_ACTIVE else ""
        )
        slot_values_in_state = {}
        if self.categorical_slot_status == STATUS_ACTIVE:
            slot_values_in_state[
                self.service_schema.get_categorical_slot_from_id(self.categorical_slot_id)
            ] = self.service_schema.get_categorical_slot_value_from_id(
                self.categorical_slot_id, self.categorical_slot_value_id
            )
        elif self.categorical_slot_status == STATUS_DONTCARE:
            slot_values_in_state[
                self.service_schema.get_categorical_slot_from_id(self.categorical_slot_id)
            ] = STR_DONTCARE
        if self.noncategorical_slot_status == STATUS_ACTIVE:
            slot = self.service_schema.get_non_categorical_slot_from_id(self.noncategorical_slot_id)
            start_id = self.noncategorical_slot_value_start[slot]
            end_id = self.noncategorical_slot_value_end[slot]
            # Token list is consisted of the subwords that may start with "##". We
            # remove "##" to reconstruct the original value. Note that it's not a
            # strict restoration of the original string. It's primarily used for
            # debugging.
            # ex. ["san", "j", "##ose"] --> "san jose"
            readable_value = " ".join(utt_toks[start_id : end_id + 1]).replace(" ##", "")
            slot_values_in_state[slot] = readable_value
        elif self.noncategorical_slot_status == STATUS_DONTCARE:
            slot = self.service_schema.get_non_categorical_slot_from_id(self.noncategorical_slot_id)
            slot_values_in_state[slot] = STR_DONTCARE

        summary_dict = {
            "utt_tok_mask_pairs": utt_tok_mask_pairs,
            "utt_len": seq_length,
            "categorical_slot_id": self.categorical_slot_id,
            "noncategorical_slot_id": self.noncategorical_slot_id,
            "intent_id": self.intent_id,
            "service_name": self.service_schema.service_name,
            "active_intent": active_intent,
            "slot_values_in_state": slot_values_in_state,
        }
        return summary_dict

    def add_utterance_features(
        self, system_tokens, system_inv_alignments, user_tokens, user_inv_alignments, system_utterance, user_utterance
    ):
        """Add utterance related features input to InputExample.

        Note: this method modifies the system tokens and user_tokens in place to
        make their total length <= the maximum input length for BERT model.

        Args:
          system_tokens: a list of strings which represents schema description.
          system_inv_alignments: a list of tuples which denotes the start and end
            charater of the tpken that a bert token originates from in the original
            schema description.
          user_tokens: a list of strings which represents utterance.
          user_inv_alignments: a list of tuples which denotes the start and end
            charater of the token that a bert token originates from in the original
            system and user utterance.
        """
        # Input sequence length for utterance BERT encoder
        max_utt_len = self._max_seq_length

        # Modify lengths of schema description & utterance so that length of total utt
        # (including cls_token, setp_token, sep_token) is no more than max_utt_len
        is_too_long = truncate_seq_pair(system_tokens, user_tokens, max_utt_len - 3)
        if is_too_long:
            logging.debug(
                f'Utterance sequence truncated in example id - {self.example_id} from {len(system_tokens) + len(user_tokens)}.'
            )

        # Construct the tokens, segment mask and valid token mask which will be
        # input to BERT, using the tokens for schema description (sequence A) and
        # system and user utterance (sequence B).
        utt_subword = []
        utt_seg = []
        utt_mask = []
        start_char_idx = []
        end_char_idx = []

        utt_subword.append(self._tokenizer.cls_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(system_tokens):
            utt_subword.append(subword)
            utt_seg.append(0)
            utt_mask.append(1)
            st, en = system_inv_alignments[subword_idx]
            start_char_idx.append(-(st + 1))
            end_char_idx.append(-(en + 1))

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(0)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        for subword_idx, subword in enumerate(user_tokens):
            utt_subword.append(subword)
            utt_seg.append(1)
            utt_mask.append(1)
            st, en = user_inv_alignments[subword_idx]
            start_char_idx.append(st + 1)
            end_char_idx.append(en + 1)

        utt_subword.append(self._tokenizer.sep_token)
        utt_seg.append(1)
        utt_mask.append(1)
        start_char_idx.append(0)
        end_char_idx.append(0)

        utterance_ids = self._tokenizer.tokens_to_ids(utt_subword)

        # Zero-pad up to the BERT input sequence length.
        while len(utterance_ids) < max_utt_len:
            utterance_ids.append(0)
            utt_seg.append(0)
            utt_mask.append(0)
            start_char_idx.append(0)
            end_char_idx.append(0)
        self.utterance_ids = utterance_ids
        self.utterance_segment = utt_seg
        self.utterance_mask = utt_mask
        self.start_char_idx = start_char_idx
        self.end_char_idx = end_char_idx

        self.user_utterance = user_utterance
        self.system_utterance = system_utterance

    def make_copy(self):
        """Make a copy of the current example with utterance features."""
        new_example = SGDInputExample(
            schema_config=self.schema_config,
            service_schema=self.service_schema,
            example_id=self.example_id,
            example_id_num=self.example_id_num.copy(),
            tokenizer=self._tokenizer,
        )
        return new_example

    def make_copy_of_categorical_features(self):
        """Make a copy of the current example with utterance and categorical features."""
        new_example = self.make_copy()

        new_example.categorical_slot_status = self.categorical_slot_status
        return new_example

    def make_copy_of_non_categorical_features(self):
        """Make a copy of the current example with utterance features and non categorical features."""
        new_example = self.make_copy()
        new_example.noncategorical_slot_id = self.noncategorical_slot_id
        new_example.noncategorical_slot_status = self.noncategorical_slot_status
        new_example.utterance_ids = list(self.utterance_ids)
        new_example.utterance_segment = list(self.utterance_segment)
        new_example.utterance_mask = list(self.utterance_mask)
        new_example.start_char_idx = list(self.start_char_idx)
        new_example.end_char_idx = list(self.end_char_idx)
        new_example.user_utterance = self.user_utterance
        new_example.system_utterance = self.system_utterance
        new_example.noncategorical_slot_status = self.noncategorical_slot_status
        new_example.noncategorical_slot_value_start = self.noncategorical_slot_value_start
        new_example.noncategorical_slot_value_end = self.noncategorical_slot_value_end
        return new_example

    def add_categorical_slots(self, state_update: dict):
        """Add features for categorical slots.
        Args:
            state_update: slot value pairs of the state update
        """

        categorical_slots = self.service_schema.categorical_slots
        if not categorical_slots:
            return
        slot = categorical_slots[self.categorical_slot_id]
        values = state_update.get(slot, [])

        if not values:
            self.categorical_slot_status = STATUS_OFF
        elif values[0] == STR_DONTCARE:
            self.categorical_slot_status = STATUS_DONTCARE
        else:
            self.categorical_slot_status = STATUS_ACTIVE
            self.categorical_slot_value_status = (
                self.categorical_slot_value_id == self.service_schema.get_categorical_slot_value_id(slot, values[0])
            )

    def add_noncategorical_slots(self, state_update: dict, system_span_boundaries: dict, user_span_boundaries: dict):
        """Add features for non-categorical slots.
        Args:
            state_update: slot value pairs of state update
            system_span_boundaries: span boundaries of schema description
            user_span_boundaries: span boundaries of utterance 
        """

        noncategorical_slots = self.service_schema.non_categorical_slots
        slot = noncategorical_slots[self.noncategorical_slot_id]

        values = state_update.get(slot, [])
        if not values:
            self.noncategorical_slot_status = STATUS_OFF
        elif values[0] == STR_DONTCARE:
            self.noncategorical_slot_status = STATUS_DONTCARE
        else:
            self.noncategorical_slot_status = STATUS_ACTIVE
            # Add indices of the start and end tokens for the first encountered
            # value. Spans in user utterance are prioritized over the system
            # utterance. If a span is not found, the slot value is ignored.
            if slot in user_span_boundaries:
                start, end = user_span_boundaries[slot]
            elif slot in system_span_boundaries:
                start, end = system_span_boundaries[slot]
            else:
                # A span may not be found because the value was cropped out or because
                # the value was mentioned earlier in the dialogue. Since this model
                # only makes use of the last two utterances to predict state updates,
                # it will fail in such cases.
                logging.debug(
                    f'"Slot values {str(values)} not found in user or system utterance in example with id - {self.example_id}.'
                )
                start = 0
                end = 0
            self.noncategorical_slot_value_start = start
            self.noncategorical_slot_value_end = end

    def add_requested_slots(self, frame: dict):
        """Add requested slots to InputExample
        Args:
            frame: frame object from which requested slots are extracted
        """
        all_slots = self.service_schema.slots
        slot = all_slots[self.requested_slot_id]
        if slot in frame["labels"]["slots"]:
            self.requested_slot_status = STATUS_ACTIVE

    def add_intents(self, frame):
        """Add intents to InputExample
        Args:
            frame: frame object from which intents are extracted
        """
        all_intents = self.service_schema.intents
        intent = all_intents[self.intent_id]
        if intent == frame["labels"]["intent"]:
            self.intent_status = STATUS_ACTIVE


# Modified from run_classifier._truncate_seq_pair in the public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def truncate_seq_pair(tokens_a: List[int], tokens_b: List[int], max_length: int) -> bool:
    """Truncate a seq pair in place so that their total length <= max_length.
    Args:
        tokens_a: first token sequence
        tokens_b: second token sequence
        max_length: truncated sequence length
    Returns:
        is_too_long: whether combined sequences exceed maximum sequence length
    """
    is_too_long = False
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        is_too_long = True
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return is_too_long
