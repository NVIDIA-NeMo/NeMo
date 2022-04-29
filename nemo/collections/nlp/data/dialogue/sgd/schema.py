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
Wrappers for schemas of different services.
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/schema.py
"""

import json
from typing import List, Optional, Union

from nemo.utils import logging

__all__ = ['Schema']


class ServiceSchema(object):
    """A wrapper for schema for a service."""

    def __init__(self, schema_json: dict, service_id: Optional[int] = None):
        """
        Constructor for ServiceSchema.
        Args:
            schema_json: schema json dict
            service_id: service ID
        """
        self._service_name = schema_json["service_name"]
        self._description = schema_json["description"]
        self._schema_json = schema_json
        self._service_id = service_id

        # Construct the vocabulary for intents, slots, categorical slots,
        # non-categorical slots and categorical slot values.
        self._intents = ["NONE"] + sorted(i["name"] for i in schema_json["intents"])
        self._intent_descriptions = {i["name"]: i["description"] for i in schema_json["intents"]}
        self._intent_descriptions["NONE"] = "none"
        self._slots = sorted(s["name"] for s in schema_json["slots"])
        self._slots_descriptions = {s["name"]: s["description"] for s in schema_json["slots"]}
        self._categorical_slots = sorted(
            s["name"] for s in schema_json["slots"] if s["is_categorical"] and s["name"] in self.state_slots
        )
        self._non_categorical_slots = sorted(
            s["name"] for s in schema_json["slots"] if not s["is_categorical"] and s["name"] in self.state_slots
        )
        slot_schemas = {s["name"]: s for s in schema_json["slots"]}
        categorical_slot_values = {}
        categorical_slot_value_ids = {}
        categorical_slot_ids = {}
        non_categorical_slot_ids = {}
        for slot_id, slot in enumerate(self._categorical_slots):
            slot_schema = slot_schemas[slot]
            values = sorted(slot_schema["possible_values"])
            categorical_slot_values[slot] = values
            value_ids = {value: idx for idx, value in enumerate(values)}
            categorical_slot_value_ids[slot] = value_ids
            categorical_slot_ids[slot] = slot_id

        for slot_id, slot in enumerate(self._non_categorical_slots):
            non_categorical_slot_ids[slot] = slot_id

        self._categorical_slot_values = categorical_slot_values
        self._categorical_slot_value_ids = categorical_slot_value_ids

        self._categorical_slot_ids = categorical_slot_ids
        self._non_categorical_slot_ids = non_categorical_slot_ids

    @property
    def schema_json(self) -> dict:
        """Returns schema json dictionary"""
        return self._schema_json

    @property
    def state_slots(self) -> set:
        """Set of slots which are permitted to be in the dialogue state."""
        state_slots = set()
        for intent in self._schema_json["intents"]:
            state_slots.update(intent["required_slots"])
            state_slots.update(intent["optional_slots"])
        return state_slots

    @property
    def service_name(self):
        return self._service_name

    @property
    def service_id(self):
        return self._service_id

    @property
    def description(self):
        return self._description

    @property
    def slots(self):
        return self._slots

    @property
    def intents(self):
        return self._intents

    @property
    def intent_descriptions(self):
        return self._intent_descriptions

    @property
    def slot_descriptions(self):
        return self._slots_descriptions

    @property
    def categorical_slots(self):
        return self._categorical_slots

    @property
    def non_categorical_slots(self):
        return self._non_categorical_slots

    @property
    def categorical_slot_values(self):
        return self._categorical_slot_values

    def get_categorical_slot_values(self, slot):
        return self._categorical_slot_values[slot]

    def get_slot_from_id(self, slot_id):
        return self._slots[slot_id]

    def get_intent_from_id(self, intent_id):
        return self._intents[intent_id]

    def get_categorical_slot_from_id(self, slot_id):
        return self._categorical_slots[slot_id]

    def get_non_categorical_slot_from_id(self, slot_id):
        return self._non_categorical_slots[slot_id]

    def get_categorical_slot_value_from_id(self, slot_id, value_id):
        slot = self._categorical_slots[slot_id]
        return self._categorical_slot_values[slot][value_id]

    def get_categorical_slot_value_id(self, slot, value):
        return self._categorical_slot_value_ids[slot][value]

    def get_categorical_slot_id(self, slot):
        return self._categorical_slot_ids[slot]

    def get_non_categorical_slot_id(self, slot):
        return self._non_categorical_slot_ids[slot]


class Schema(object):
    """Wrapper for schemas for all services in a dataset."""

    def __init__(self, schema_json_paths: Union[str, List[str]]):
        """
        schema_json_paths: list of .json path to schema files of a single str with path to the json file.
        """
        # Load the schema from the json file.
        if isinstance(schema_json_paths, str):
            with open(schema_json_paths, "r") as f:
                all_schemas = json.load(f)
                f.close()
        else:
            # load multiple schemas from the list of the json files
            all_schemas = []
            completed_services = []
            for schema_json_path in schema_json_paths:
                with open(schema_json_path, "r") as f:
                    schemas = json.load(f)
                    f.close()
                    logging.debug("Num of services in %s: %s", schema_json_path, len(schemas))

                for service in schemas:
                    if service['service_name'] not in completed_services:
                        completed_services.append(service['service_name'])
                        all_schemas.append(service)

        self._services = sorted(schema["service_name"] for schema in all_schemas)
        self._services_vocab = {v: k for k, v in enumerate(self._services)}
        self._services_id_to_vocab = {v: k for k, v in self._services_vocab.items()}
        service_schemas = {}
        for schema in all_schemas:
            service = schema["service_name"]
            service_schemas[service] = ServiceSchema(schema, service_id=self.get_service_id(service))

        self._service_schemas = service_schemas
        self._schemas = all_schemas
        self._slots_relation_list = {}

    def get_service_id(self, service: str):
        return self._services_vocab[service]

    def get_service_from_id(self, service_id: int):
        return self._services[service_id]

    def get_service_schema(self, service: str):
        return self._service_schemas[service]

    @property
    def services(self):
        return self._services

    def save_to_file(self, file_path):
        """
        Saves schema object to file
        Args:
            file_path: path to store schema object at
        """
        with open(file_path, "w") as f:
            json.dump(self._schemas, f, indent=2)
