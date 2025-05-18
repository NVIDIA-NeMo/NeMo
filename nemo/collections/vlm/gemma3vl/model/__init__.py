# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.vlm.gemma3vl.model.base import Gemma3VLConfig, Gemma3VLModel
from nemo.collections.vlm.gemma3vl.model.gemma3vl import Gemma3VLConfig4B, Gemma3VLConfig12B, Gemma3VLConfig27B
from nemo.collections.vlm.gemma3vl.model.vision import Gemma3VLMultimodalProjectorConfig, Gemma3VLVisionConfig

__all__ = [
    "Gemma3VLVisionConfig",
    "Gemma3VLMultimodalProjectorConfig",
    "Gemma3VLConfig",
    "Gemma3VLConfig4B",
    "Gemma3VLConfig12B",
    "Gemma3VLConfig27B",
    "Gemma3VLModel",
]
