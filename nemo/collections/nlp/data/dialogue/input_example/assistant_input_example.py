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

from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample


class DialogueAssistantInputExample(DialogueInputExample):
    """
    Template for DialogueAssistantInputExample

    Meant as a descriptor rather than to be instantiated

    Please instantiate using the base class 'DialogueInputExample'

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
