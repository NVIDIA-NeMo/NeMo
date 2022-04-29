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


class DialogueDesignInputExample(DialogueInputExample):
    """
    Template for DialogueDesignInputExample

    Meant as a descriptor rather than to be instantiated

    Please instantiate using the base class 'DialogueInputExample'

    {
        "utterance": <utterance>,
        "system_utterance": <system_utterance>,
        "labels": {
            "service": <service>,
            "intent": <intent_description>,
            "slots": {
                <slot-name1>: '',
                <slot-name2>: '',
            },  # dataset does not contain ground truth slot values
        },
        "possible_labels": {
            'intent': [<intent1>, <intent2>, ...],
            "service": [<service1>, <service2>, ...],
            "slots": {
                "<slot-name1>": [<slot-value1>, <slot-value2>, ...],
                "<slot-name2>": [<slot-value1>, <slot-value2>, ...],
            }
        },
        "description": {
            "service": <service>,
            "intent": <intent_description>,
            "slots": {
                "<slot-name1>": "<slot-question1>",
                "<slot-name2>": "<slot-question2>",
            }
        },
    }
    """
