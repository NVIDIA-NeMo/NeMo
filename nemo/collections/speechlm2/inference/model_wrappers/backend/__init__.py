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

"""Per-component inference backends.

``DuplexLLM`` and ``DuplexTTS`` are the two contracts the VoiceChat wrapper
runs its frame loop against; ``pytorch/`` and ``vllm/`` hold one
implementation of each. The vLLM implementations are not re-exported here so
that importing this package stays free of vLLM.
"""

from nemo.collections.speechlm2.inference.model_wrappers.backend.eartts import DuplexTTS
from nemo.collections.speechlm2.inference.model_wrappers.backend.llm import DuplexLLM
from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.eartts import (
    PyTorchEarTTS,
    TTSGenerationResult,
)
from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.llm import PyTorchLLM

__all__ = [
    'DuplexLLM',
    'DuplexTTS',
    'PyTorchEarTTS',
    'TTSGenerationResult',
    'PyTorchLLM',
]
