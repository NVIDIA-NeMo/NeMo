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

"""Per-frame contract for the VoiceChat LLM component.

Implemented by :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.llm.PyTorchLLM`
and :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.llm.VllmLLM`.
The wrapper picks one at construction and then runs the same frame loop, so
nothing above this line inspects which engine is in use.

The contract is deliberately just :meth:`DuplexLLM.step`. Cache creation,
prompt prefill and request abort are native-only concepts -- vLLM does those
inside the engine and the session -- so they are not part of it; the wrapper
calls them on the native object at stream setup, where it knows it has one.
Adding them here as no-op defaults would advertise methods that only mean
something for one implementation.

The same rule applies to the *parameters*, not just the method list: anything
only one backend can act on is per-stream state on ``StreamingDecodeState``
rather than an argument, so the signature does not advertise capabilities an
implementation has to discard.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LlmStepResult:
    """One frame's LLM output, in the shape both backends can fill.

    A dataclass rather than a dict so the optional channels are discoverable
    and callers stop probing with ``"text_logits" in ans``. The auxiliary
    tokens are ``None`` when the checkpoint has no such head; the logits are
    ``None`` unless the backend has them *and* ``return_debug`` asked (vLLM
    keeps its logits inside the engine, so it never fills them).
    """

    predicted_token: torch.Tensor
    asr_predicted_token: torch.Tensor | None = None
    function_predicted_token: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    asr_logits: torch.Tensor | None = None
    function_logits: torch.Tensor | None = None


class DuplexLLM(ABC):
    """Produces one frame's text (and optional ASR/function) tokens."""

    @abstractmethod
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
        """Advance one 80 ms frame.

        Args:
            frame_embedding: Encoded audio for this frame, shape ``(B, 1, H)``.
            state: The stream's ``StreamingDecodeState``. Implementations read
                committed history (``gen_text`` and friends) from it and update
                their own per-stream fields -- caches,
                ``input_embeds_history`` -- in place.
            frame_offset: Index of this frame within the current chunk.
            current_frame_idx: Index of this frame within the whole stream.
            has_prompt: Whether a system prompt is already in the LLM state.
            return_debug: Ask for logits in the result when the backend has them.
            sampling_params: Per-stream sampling overrides, for backends that
                can still apply them at this point.
            debug_logger: Receives the per-frame LLM input.
        """
        raise NotImplementedError
