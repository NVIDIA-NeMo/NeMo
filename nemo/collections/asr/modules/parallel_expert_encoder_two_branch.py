# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Compatibility imports for the canonical two-branch Parallel Expert Encoder.

The released public class remains defined in :mod:`parallel_expert_encoder`.
This module exists so exported configs from the stacked integration continue to
resolve to that exact same class object rather than a duplicate implementation.
"""

from nemo.collections.asr.modules.parallel_expert_encoder import *  # noqa: F401,F403
from nemo.collections.asr.modules.parallel_expert_encoder import (  # noqa: F401
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)

__all__ = ["ParallelExpertEncoder", "ParallelExpertEncoderPT"]
