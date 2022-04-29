# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


class MellonQAInputExample(DialogueInputExample):
    """
    Template for MellonQAInputExample

    Meant as a descriptor rather than to be instantiated

    Please instantiate using the base class 'DialogueInputExample'

    {
        "utterance": <utterance>,
        "labels": {
            "example_id": <example_id>,
            "response": <response>,
            "fluent_response": <fluent_response>, # written version of the response that is more fluent
            "passage": <passage>, # passage which supports generating the response (answer) to the utterance (question)
        }
    }
    """
