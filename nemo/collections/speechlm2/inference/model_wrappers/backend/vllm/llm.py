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

Implements :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.llm.DuplexLLM`
against the per-stream ``OmniStreamingSession``; the PyTorch sibling lives in
``backend/pytorch/llm.py``.
"""

from typing import Any

import torch

from nemo.collections.speechlm2.inference.model_wrappers.backend.llm import DuplexLLM, LlmStepResult
from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm import require_session


class VllmLLM(DuplexLLM):
    """Runs Nemotron in a vLLM-Omni engine, one acoustic frame per step.

    Stateless itself: the engine is process-scoped and owned by
    ``OmniRuntime``, and everything request-scoped lives in the session that
    the pipeline attached to the decode state at prefill.
    """

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
        """One Nemotron step -- see ``DuplexLLM.step``.

        Nemotron builds its own duplex input embedding from the acoustic frame,
        so there is no ``build_input_embedding`` and no history replay here;
        ``frame_offset`` and ``has_prompt`` do not apply. Per-stream sampling
        was fixed when the session was created, and logits stay inside the
        engine, so ``return_debug`` cannot be honoured either -- the result's
        logit fields stay None.

        The previous frame's committed text token is fed back explicitly. That
        is what carries a forced-turn-taking rewrite into Nemotron's own
        history, the role ``gen_text`` plays for the PyTorch backend.
        """
        del frame_offset, has_prompt, sampling_params, return_debug

        session = require_session(state)
        if debug_logger is not None:
            debug_logger.log_input_embeds(frame_embedding)

        prev_text_token = None
        if current_frame_idx > 0:
            prev_text_token = int(state.gen_text[0, current_frame_idx - 1].item())

        tokens = session.step_llm(frame_embedding.reshape(-1), prev_text_token=prev_text_token)
        return LlmStepResult(
            predicted_token=tokens.text,
            asr_predicted_token=tokens.asr,
            function_predicted_token=tokens.function,
        )
