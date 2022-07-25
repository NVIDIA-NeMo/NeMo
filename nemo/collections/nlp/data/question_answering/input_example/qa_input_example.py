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

from dataclasses import dataclass
from typing import List


@dataclass
class QAExample(object):
    """ A single training/test example for a QA dataset, as loaded from disk """

    qas_id: str  # The example's unique identifier
    question_text: str
    context_text: str
    context_id: int
    answer_text: str
    start_position_character: int  # The character position of the start of the answer, 0 indexed
    title: str
    answers: List[
        str
    ] = None  # None by default, this is used during evaluation. Holds answers as well as their start positions
    is_impossible: bool = False  # False by default, set to True if the example has no possible answer
