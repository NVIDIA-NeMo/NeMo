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

"""Streaming decode of EarTTS acoustic codes into a waveform.

Used for every LLM/TTS engine pairing: the engines emit codes, not audio.
"""

import torch

from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.utils import logging


class AudioCodec:
    """Streaming decode of acoustic codes into a waveform."""

    def __init__(self, tts_model, device: torch.device):
        """
        Args:
            tts_model: The ``DuplexEARTTS`` that owns ``audio_codec`` and the
                control-code table.
            device: Device the codec decodes on.
        """
        self.tts_model = tts_model
        self.device = device

    def create_state(self, max_len: int) -> tuple[torch.Tensor, object]:
        """Per-stream codec state: ``(subword_mask, codec_cache)``."""
        from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache

        subword_mask = torch.ones((1, max_len), device=self.device, dtype=torch.bool)
        return subword_mask, CausalConv1dCache()

    def decode(self, new_codes: list[torch.Tensor], cache) -> torch.Tensor | None:
        """Decode this chunk's accumulated codes into a waveform.

        Args:
            new_codes: One ``(B, T, num_quantizers)`` tensor per frame.
            cache: The stream's ``CausalConv1dCache``, updated in place.

        Returns:
            The decoded waveform, or *None* when no codes were produced.
        """
        if not new_codes:
            return None

        with fp32_precision(), torch.no_grad():
            codes = torch.cat(new_codes, dim=1)
            codes = self._replace_control_codes(codes)
            code_len = torch.tensor([codes.shape[1]], dtype=torch.long, device=self.device)
            decoded_audio, _ = self.tts_model.audio_codec.decode(codes, code_len, cache=cache)
        return decoded_audio

    def _replace_control_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Substitute codec silence for the model's control codes.

        Only checkpoints that define a control-code table need this; older ones
        have none, and then the codes pass through unchanged.
        """
        control_codes = getattr(self.tts_model, "_control_codes", None)
        if control_codes is None:
            return codes
        from nemo.collections.speechlm2.models.duplex_ear_tts import replace_control_speech_codes

        return replace_control_speech_codes(
            codes,
            control_codes,
            getattr(self.tts_model, "codec_silence_tokens", None),
        )

    def log_configuration(self) -> None:
        """Record the codec's sample rate and frame rate once, at load time."""
        logging.info(
            f"Audio codec ready: target_fps={self.tts_model.target_fps}, "
            f"sample_rate={self.tts_model.target_sample_rate}"
        )
