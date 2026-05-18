# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class S2SRequestOptions:
    """Per-stream options for S2S inference.

    Attached to the first ``Frame`` of each stream via the ``options``
    field so that the pipeline can read per-stream configuration at the
    start of every new audio file / Triton sequence.
    """

    system_prompt: str | None = None
    fc_random_ack_enabled: bool | None = None
    tool_reminder_enabled: bool | None = None
    tool_response_text: str | None = None
    gt_reference: str | None = None
    gt_tool_response: str | None = None
    audio_duration_sec: float | None = None
    user_segments: List[dict] | None = None
    audio_filepath: str | None = None
