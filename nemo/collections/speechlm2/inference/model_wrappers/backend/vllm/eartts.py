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

"""vLLM-Omni backend for the TTS (EarTTS) component of NemotronVoiceChat.

Implements :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.eartts.DuplexTTS`.
The PyTorch sibling lives in ``backend/pytorch/eartts.py``.

This PR stubs the class: construction raises so a ``vllm_omni`` engine
selection cannot silently fall through to native. The implementation is the
parent commit on ``duplex-vllm-omni-on-main``.
"""

from typing import Any

import torch

from nemo.collections.speechlm2.inference.model_wrappers.backend.eartts import DuplexTTS
from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import VLLM_OMNI, reject_unimplemented_vllm


class VllmEarTTS(DuplexTTS):
    """Runs EarTTS in a vLLM-Omni engine, one text token per step.

    Not implemented in this PR. Same contract as :class:`PyTorchEarTTS`, so the
    wrapper's frame loop does not branch on engine type.
    """

    def __init__(self, device: torch.device):
        """
        Args:
            device: Device the native audio codec decodes on, so the codes this
                backend returns land where the codec expects them.
        """
        del device
        reject_unimplemented_vllm("native", VLLM_OMNI)

    def step(self, state: Any, current_frame_idx: int, request_id: str) -> torch.Tensor:
        """One EarTTS step -- see ``DuplexTTS.step``."""
        del state, current_frame_idx, request_id
        reject_unimplemented_vllm("native", VLLM_OMNI)
