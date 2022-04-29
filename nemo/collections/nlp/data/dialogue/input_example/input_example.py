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

__all__ = ['DialogueInputExample']


class DialogueInputExample(object):
    """
    Generic Dialogue Input Example
    Uses data: dict as a flexible interface to support various input types.
    This ranges from classification labels, to complex nested labels such as those in SGD

    {
        "utterance": <utterance>,
        "labels": { 
            "intent": <intent>,
            "slots": { ... },
        }
    }
    """

    def __init__(self, data: dict):
        self.data = data

    def __repr__(self):
        return self.data

    def __str__(self):
        return self.data
