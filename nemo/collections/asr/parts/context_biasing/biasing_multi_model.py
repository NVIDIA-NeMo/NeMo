# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Stub — BiasingRequestItemConfig is used as an optional type hint in
# ASRRequestOptions.biasing_cfg. Context biasing is not used for S2S voice chat.
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class BiasingRequestItemConfig:
    """Minimal stub for context-biasing config used in ASRRequestOptions."""
    boosted_lm_words: List[str] = field(default_factory=list)
    boosted_lm_score: float = 0.0
    context_graph: Optional[Any] = None

    def is_empty(self) -> bool:
        return not self.boosted_lm_words and self.context_graph is None
