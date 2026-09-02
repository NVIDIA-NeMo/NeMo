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

"""vLLM-Omni backend for the LLM component of NemotronVoiceChat.

Implements :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.llm.DuplexLLM`.
The PyTorch sibling lives in ``backend/pytorch/llm.py``.

This PR stubs the class: construction raises so a ``vllm_omni`` engine
selection cannot silently fall through to native. The implementation is the
parent commit on ``duplex-vllm-omni-on-main``.
"""

from typing import Any

import torch

from nemo.collections.speechlm2.inference.model_wrappers.backend.llm import DuplexLLM, LlmStepResult
from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import VLLM_OMNI, reject_unimplemented_vllm


class VllmLLM(DuplexLLM):
    """Runs Nemotron in a vLLM-Omni engine, one acoustic frame per step.

    Not implemented in this PR. The native frame loop already calls
    :meth:`DuplexLLM.step` without inspecting the engine type, so landing the
    runtime later does not reshape the per-frame path.
    """

    def __init__(self) -> None:
        reject_unimplemented_vllm(VLLM_OMNI, "native")

    def step(
        self,
        frame_embedding: torch.Tensor,
        state: Any,
        *,
        frame_offset: int,
        current_frame_idx: int,
        has_prompt: bool,
        return_debug: bool = False,
        sampling_params: dict[str, float] | None = None,
        debug_logger: Any = None,
    ) -> LlmStepResult:
        """One Nemotron step -- see ``DuplexLLM.step``."""
        del frame_embedding, state, frame_offset, current_frame_idx, has_prompt
        del return_debug, sampling_params, debug_logger
        reject_unimplemented_vllm(VLLM_OMNI, "native")
