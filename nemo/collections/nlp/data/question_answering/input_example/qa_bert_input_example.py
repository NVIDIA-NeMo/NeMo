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
from typing import Dict, List, Optional


@dataclass
class BERTQAInputExample(object):
    """ A single set of features of a QA example for BERT-like model """

    unique_id: int
    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]
    example_index: int = None
    doc_span_index: int = None
    tokens: List[str] = None
    token_to_orig_map: Dict[int, int] = None
    token_is_max_context: Dict[int, bool] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    is_impossible: Optional[int] = None
