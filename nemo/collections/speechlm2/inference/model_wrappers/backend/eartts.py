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

"""Per-frame contract for the VoiceChat TTS (EarTTS) component.

Implemented by :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.eartts.PyTorchEarTTS`
and :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.eartts.VllmEarTTS`.

Separate from :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.llm.DuplexLLM`
because the two components are chosen independently, and because an LLM-shaped
call signature never fitted EarTTS: it consumes a text token and emits acoustic
codes.

Note what is *not* here. ``inference_force_speech_silence_on_eos`` is applied by
both EarTTS implementations internally, on the acoustic input of the step whose
text token is EOS, so it needs no place in this contract. The audio codec runs
natively for every backend and stays in the wrapper.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class DuplexTTS(ABC):
    """Turns one committed text token into one frame of acoustic codes."""

    @abstractmethod
    def step(
        self,
        state: Any,
        current_frame_idx: int,
        request_id: str,
    ) -> torch.Tensor:
        """Generate this frame's acoustic codes.

        Reads the committed text token from ``state.gen_text`` rather than
        taking it as an argument, so a caller that rewrote it (forced
        turn-taking) does not have to remember to pass the new value.

        Returns:
            Codes shaped ``(B, T, num_quantizers)``, which is what the native
            audio codec consumes.
        """
        raise NotImplementedError
