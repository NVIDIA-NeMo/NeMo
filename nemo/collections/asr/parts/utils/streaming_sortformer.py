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

import copy
import math
from typing import TYPE_CHECKING

import torch

from nemo.collections.asr.parts.preprocessing.features import normalize_batch

if TYPE_CHECKING:
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

__all__ = ['SortformerStreamingSession']


class SortformerStreamingSession:
    """Stateful raw-audio session for a streaming Sortformer model.

    A session owns the feature and speaker-cache state for one mono audio stream. Incoming waveform chunks may have
    arbitrary lengths. The session waits until a complete model chunk and its configured right context are available,
    then returns only the newly committed speaker probabilities. Call :meth:`reset` before reusing a session for a new
    stream.

    Args:
        model: Streaming ``SortformerEncLabelModel`` in evaluation mode.
    """

    def __init__(self, model: 'SortformerEncLabelModel'):
        if not model.streaming_mode:
            raise ValueError("SortformerStreamingSession requires a model with streaming_mode=True")
        if model.training:
            raise ValueError("SortformerStreamingSession requires an evaluation model; call model.eval() first")

        self.model = model
        self.device = model.device
        self._normalization = model.preprocessor.featurizer.normalize
        self._preprocessor = copy.deepcopy(model.preprocessor).to(self.device).eval()
        self._preprocessor.featurizer.normalize = None
        self._preprocessor.featurizer.dither = 0.0
        self._preprocessor.featurizer.pad_to = 0

        self._hop_length = self._preprocessor.hop_length
        self._n_fft = self._preprocessor.featurizer.n_fft
        self._stft_margin_frames = math.ceil((self._n_fft // 2 + 1) / self._hop_length) + 1
        self._chunk_frames = model.sortformer_modules.chunk_len * model.encoder.subsampling_factor
        self._left_context_frames = model.sortformer_modules.chunk_left_context * model.encoder.subsampling_factor
        self._right_context_frames = model.sortformer_modules.chunk_right_context * model.encoder.subsampling_factor
        self.reset()

    @torch.inference_mode()
    def diarize_step(self, audio_chunk: torch.Tensor, is_final: bool = False) -> torch.Tensor:
        """Consume a mono waveform chunk and return newly committed speaker probabilities.

        Args:
            audio_chunk: Float waveform with shape ``(num_samples,)`` or ``(1, num_samples)`` at the model sample rate.
            is_final: Flush the remaining audio and close the stream. No more audio can be supplied until ``reset``.

        Returns:
            Tensor with shape ``(1, new_frames, num_speakers)``. ``new_frames`` can be zero while the session waits
            for a complete chunk and its right context.
        """
        if self._finalized:
            raise RuntimeError("This streaming session is finalized; call reset() before supplying more audio")
        if not isinstance(is_final, bool):
            raise TypeError(f"is_final must be a boolean, got {type(is_final).__name__}")

        audio_chunk = self._validate_audio_chunk(audio_chunk)
        if audio_chunk.numel() > 0:
            self._audio_buffer = torch.cat([self._audio_buffer, audio_chunk])
            self._received_samples += audio_chunk.numel()

        available_frames = self._available_feature_frames(is_final=is_final)
        emitted = []
        while self._next_feature_frame < available_frames:
            central_end = min(self._next_feature_frame + self._chunk_frames, available_frames)
            if not is_final and central_end + self._right_context_frames > available_frames:
                break

            feature_start = max(0, self._next_feature_frame - self._left_context_frames)
            feature_end = min(central_end + self._right_context_frames, available_frames)
            processed_signal = self._extract_feature_window(feature_start, feature_end)
            processed_signal_length = torch.tensor([feature_end - feature_start], dtype=torch.long, device=self.device)
            empty_preds = processed_signal.new_zeros((1, 0, self.model.sortformer_modules.n_spk))
            self.streaming_state, chunk_preds = self.model.forward_streaming_step(
                processed_signal=processed_signal.transpose(1, 2),
                processed_signal_length=processed_signal_length,
                streaming_state=self.streaming_state,
                total_preds=empty_preds,
                left_offset=self._next_feature_frame - feature_start,
                right_offset=feature_end - central_end,
            )
            emitted.append(chunk_preds)
            self._next_feature_frame = central_end

        self._compact_audio_buffer()

        if is_final:
            self._finalized = True

        if emitted:
            return torch.cat(emitted, dim=1)
        return torch.zeros(
            (1, 0, self.model.sortformer_modules.n_spk),
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )

    def reset(self) -> None:
        """Clear buffered audio and model state so this session can process a new stream."""
        self.streaming_state = self.model.sortformer_modules.init_streaming_state(
            batch_size=1,
            async_streaming=self.model.async_streaming,
            device=self.device,
        )
        self._audio_buffer = torch.empty(0, dtype=torch.float32, device=self.device)
        self._audio_buffer_start = 0
        self._received_samples = 0
        self._next_feature_frame = 0
        self._finalized = False

    def _available_feature_frames(self, is_final: bool) -> int:
        sample_count = torch.tensor(self._received_samples, device=self.device)
        offline_frames = int(self._preprocessor.featurizer.get_seq_len(sample_count).item())
        if is_final:
            return max(0, offline_frames)

        stable_samples = self._received_samples - self._n_fft // 2
        if stable_samples < 0:
            return 0
        stable_frames = stable_samples // self._hop_length + 1
        return max(0, min(offline_frames, stable_frames))

    def _compact_audio_buffer(self) -> None:
        first_needed_frame = max(
            0,
            self._next_feature_frame - self._left_context_frames - self._stft_margin_frames,
        )
        first_needed_sample = first_needed_frame * self._hop_length
        drop_samples = first_needed_sample - self._audio_buffer_start
        if drop_samples > 0:
            self._audio_buffer = self._audio_buffer[drop_samples:].clone()
            self._audio_buffer_start = first_needed_sample

    def _extract_feature_window(self, feature_start: int, feature_end: int) -> torch.Tensor:
        segment_start_frame = max(0, feature_start - self._stft_margin_frames)
        segment_start_sample = segment_start_frame * self._hop_length
        segment_end_sample = min(
            self._received_samples,
            (feature_end + self._stft_margin_frames) * self._hop_length,
        )
        buffer_start = segment_start_sample - self._audio_buffer_start
        buffer_end = segment_end_sample - self._audio_buffer_start
        audio_signal = self._audio_buffer[buffer_start:buffer_end].unsqueeze(0)
        audio_signal_length = torch.tensor([audio_signal.shape[1]], dtype=torch.long, device=self.device)
        features, _ = self._preprocessor(input_signal=audio_signal, length=audio_signal_length)

        local_start = feature_start - segment_start_frame
        local_end = local_start + feature_end - feature_start
        if features.shape[2] < local_end:
            raise RuntimeError(
                "Streaming preprocessor returned fewer feature frames than required: "
                f"needed {local_end}, got {features.shape[2]}"
            )
        features = features[:, :, local_start:local_end]
        feature_length = torch.tensor([features.shape[2]], dtype=torch.long, device=self.device)
        if self._normalization:
            features, _, _ = normalize_batch(features, feature_length, self._normalization)
        return features

    def _validate_audio_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        if not isinstance(audio_chunk, torch.Tensor):
            raise TypeError(f"audio_chunk must be a torch.Tensor, got {type(audio_chunk).__name__}")
        if audio_chunk.ndim == 2 and audio_chunk.shape[0] == 1:
            audio_chunk = audio_chunk.squeeze(0)
        if audio_chunk.ndim != 1:
            raise ValueError(
                "audio_chunk must contain one mono stream with shape (num_samples,) or (1, num_samples); "
                f"got {tuple(audio_chunk.shape)}"
            )
        return audio_chunk.detach().to(device=self.device, dtype=torch.float32)
