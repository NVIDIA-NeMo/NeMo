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

Implements :class:`~nemo.collections.speechlm2.inference.model_wrappers.backend.eartts.DuplexTTS`
against the per-stream ``OmniStreamingSession``; the PyTorch sibling lives in
``backend/pytorch/eartts.py``.
"""

from typing import Any

import torch

from nemo.collections.speechlm2.inference.model_wrappers.backend.eartts import DuplexTTS
from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm import require_session


class VllmEarTTS(DuplexTTS):
    """Runs EarTTS in a vLLM-Omni engine, one text token per step.

    Stateless itself, like its LLM counterpart: the engine belongs to
    ``OmniRuntime`` and the request belongs to the session on the decode state.
    With classifier-free guidance enabled, the session's conditional and
    unconditional requests are kept in lockstep by the custom scheduler, so one
    submission still yields one acoustic frame.
    """

    def __init__(self, device: torch.device):
        """
        Args:
            device: Device the native audio codec decodes on, so the codes this
                backend returns land where the codec expects them.
        """
        self.device = device

    def step(self, state: Any, current_frame_idx: int, request_id: str) -> torch.Tensor:
        """One EarTTS step -- see ``DuplexTTS.step``.

        ``inference_force_speech_silence_on_eos`` is not applied here: the
        converted EarTTS substitutes codec silence itself when the incoming
        text token is EOS, matching what DuplexEARTTS does natively. It has no
        flag for it, so it cannot honour a ``False`` setting; the wrapper
        reports that at load time.
        """
        del request_id  # The session already owns this stream's request ids.

        session = require_session(state)
        text_token = int(state.gen_text[:, current_frame_idx].item())
        session.step_tts(text_token)
        audio_chunks = session.drain_audio_codes()
        if not audio_chunks:
            raise RuntimeError("vLLM EarTTS produced no audio codes for the submitted text token")
        # The native codec helpers consume [B, T, num_quantizers].
        return torch.cat(audio_chunks, dim=0).to(self.device, dtype=torch.long).unsqueeze(0)
