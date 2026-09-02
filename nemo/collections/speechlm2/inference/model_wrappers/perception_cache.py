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

"""
Cache-aware perception encoder for streaming S2S inference.

Provides incremental mel-spectrogram encoding so that only new audio needs
to be processed each step instead of re-encoding the entire buffer.

Optional CUDA-graph replay of the encoder step is
``StreamingEncoder.set_streaming_cuda_graphs``. Subsequent chunks pass
``keep_all_outputs=False`` so that helper can capture; for
``att_context_style=chunked_limited`` that flag is a no-op versus ``True``
(``valid_out_len`` already equals lookahead + 1).
"""

import copy
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from nemo.utils import logging


@dataclass
class PerceptionCacheState:
    """Cache state for streaming perception inference.

    Holds the cache tensors for the ASR encoder used in the perception module.
    This enables cache-aware streaming inference without needing the full audio buffer.
    """

    cache_last_channel: torch.Tensor | None = None
    cache_last_time: torch.Tensor | None = None
    cache_last_channel_len: torch.Tensor | None = None

    def is_initialized(self) -> bool:
        """Check if the cache has been initialized."""
        return None not in [self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len]


class PerceptionCacheManager:
    """Manages cache-aware streaming perception encoding.

    Encapsulates preprocessor setup, encoder-cache state, and the incremental
    encoding step. Created by the inference wrapper when
    ``use_perception_cache=True``. When ``use_cudagraph=True``, attaches
    ``encoder.set_streaming_cuda_graphs`` so subsequent encoder steps replay
    from a CUDA graph; adapter and projection stay eager.
    """

    def __init__(self, model, device: torch.device, dtype: torch.dtype, use_cudagraph: bool = False):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.use_cudagraph = use_cudagraph

        self.streaming_cfg = None
        self.preprocessor = None
        self.subsampling_factor = None
        self.input_features = None
        self.sampling_frames = None

    def setup(self) -> bool:
        """Setup cache-aware streaming for the perception encoder.

        Returns:
            True if setup succeeded, False if the encoder doesn't support streaming.
        """
        from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder

        perception = self.model.stt_model.perception
        encoder = perception.encoder

        if not isinstance(encoder, StreamingEncoder):
            logging.warning("Perception encoder does not support streaming. Disabling perception cache.")
            return False

        if encoder.streaming_cfg is None:
            encoder.setup_streaming_params()

        self.streaming_cfg = encoder.streaming_cfg

        cfg = copy.deepcopy(perception.cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0

        self.preprocessor = perception.from_config_dict(cfg.preprocessor)
        self.preprocessor.to(self.device)

        self.subsampling_factor = encoder.subsampling_factor
        self.input_features = encoder._feat_in

        if hasattr(encoder, "pre_encode") and hasattr(encoder.pre_encode, "get_sampling_frames"):
            self.sampling_frames = encoder.pre_encode.get_sampling_frames()
        else:
            self.sampling_frames = None

        logging.info("Perception cache setup complete:")
        logging.info(
            f"   Streaming config: chunk_size={self.streaming_cfg.chunk_size}, "
            f"shift_size={self.streaming_cfg.shift_size}"
        )
        logging.info(f"   Pre-encode cache size: {self.streaming_cfg.pre_encode_cache_size}")
        logging.info(f"   Subsampling factor: {self.subsampling_factor}")

        if self.use_cudagraph:
            encoder.set_streaming_cuda_graphs(enabled=True)
            logging.info("   Streaming encoder CUDA graphs enabled (set_streaming_cuda_graphs)")

        return True

    def get_initial_state(self, batch_size: int = 1) -> PerceptionCacheState:
        """Get initial cache state for perception encoder."""
        encoder = self.model.stt_model.perception.encoder
        cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
            batch_size=batch_size
        )

        return PerceptionCacheState(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def step(
        self,
        audio_input: torch.Tensor,
        frame_idx: int,
        num_frames_per_chunk: int,
        perception_cache: PerceptionCacheState,
    ) -> tuple[torch.Tensor, PerceptionCacheState]:
        """
        Perform cache-aware perception encoding for streaming inference.

        Note: "chunk" in this method (chunk_size, mel_chunk, etc.) follows NeMo's
        cache-aware streaming encoder API and is measured in mel-spectrogram time-steps,
        not audio samples or seconds.

        This method computes the full mel spectrogram from the audio buffer, then slices
        it appropriately based on the frame index. It supports processing multiple
        "base steps" in a single call, where each base step processes (lookahead + 1) frames.

        Processing logic per sub-step:
        - First sub-step (sub_frame_idx == 0): take first chunk_size_first columns,
          prepend zeros for pre_encode_cache
        - Subsequent sub-steps (sub_frame_idx > 0): take chunk_size columns starting from
          (shift_size_first + (step_number-1)*shift_size), prepend pre_encode_cache_size
          columns from mel spec

        The method loops over sub-steps, running the encoder for each and concatenating
        the outputs. This allows num_frames_per_chunk to be a multiple of (lookahead + 1).

        Args:
            audio_input: Audio buffer tensor [B, T] (full buffer with all samples)
            frame_idx: Current frame index in the stream
            num_frames_per_chunk: Number of 80ms frames to process. Must be a multiple
                of (lookahead + 1), i.e., encoder._cfg.att_context_size[1] + 1
            perception_cache: Current cache state containing encoder caches

        Returns:
            Tuple of (encoded_output [B, T_out, D], updated_perception_cache)
            where T_out = num_frames_per_chunk (one output frame per input frame)
        """
        perception = self.model.stt_model.perception
        encoder = perception.encoder
        streaming_cfg = self.streaming_cfg

        audio_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
        processed_signal, _ = self.preprocessor(
            input_signal=audio_input,
            length=audio_len,
        )

        if isinstance(streaming_cfg.chunk_size, list):
            chunk_size_first = streaming_cfg.chunk_size[0]
            chunk_size = streaming_cfg.chunk_size[1]
        else:
            chunk_size_first = streaming_cfg.chunk_size
            chunk_size = streaming_cfg.chunk_size

        if isinstance(streaming_cfg.shift_size, list):
            shift_size_first = streaming_cfg.shift_size[0]
            shift_size = streaming_cfg.shift_size[1]
        else:
            shift_size_first = streaming_cfg.shift_size
            shift_size = streaming_cfg.shift_size

        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            pre_encode_cache_size_first = streaming_cfg.pre_encode_cache_size[0]
            pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
        else:
            pre_encode_cache_size_first = streaming_cfg.pre_encode_cache_size
            pre_encode_cache_size = streaming_cfg.pre_encode_cache_size

        cache_last_channel = perception_cache.cache_last_channel
        cache_last_time = perception_cache.cache_last_time
        cache_last_channel_len = perception_cache.cache_last_channel_len

        base_step_size = encoder._cfg.att_context_size[1] + 1
        if num_frames_per_chunk % base_step_size != 0:
            raise ValueError(
                f"num_frames_per_chunk must be a multiple of (lookahead + 1) = {base_step_size}. "
                f"Got num_frames_per_chunk={num_frames_per_chunk}"
            )
        num_sub_steps = num_frames_per_chunk // base_step_size

        encoded_chunks = []

        for sub_step in range(num_sub_steps):
            sub_frame_idx = frame_idx + (sub_step * base_step_size)
            is_first_sub_step = sub_frame_idx == 0

            if is_first_sub_step:
                cur_chunk_size = chunk_size_first
                cur_pre_encode_cache_size = pre_encode_cache_size_first
                drop_extra_pre_encoded = 0

                mel_chunk = processed_signal[:, :, :cur_chunk_size]

                if cur_pre_encode_cache_size > 0:
                    zeros_pad = torch.zeros(
                        (processed_signal.size(0), self.input_features, cur_pre_encode_cache_size),
                        device=self.device,
                        dtype=processed_signal.dtype,
                    )
                    mel_chunk = torch.cat([zeros_pad, mel_chunk], dim=-1)
            else:
                cur_chunk_size = chunk_size
                cur_pre_encode_cache_size = pre_encode_cache_size
                drop_extra_pre_encoded = streaming_cfg.drop_extra_pre_encoded

                mel_T = processed_signal.shape[-1]

                step_number = sub_frame_idx // base_step_size
                chunk_start = shift_size_first + (step_number - 1) * shift_size
                chunk_end = chunk_start + cur_chunk_size

                offset = chunk_size - shift_size_first
                if chunk_end > mel_T - offset:
                    sub_steps_remaining = num_sub_steps - 1 - sub_step
                    chunk_end = mel_T - offset - sub_steps_remaining * shift_size
                    chunk_start = chunk_end - cur_chunk_size

                main_chunk = processed_signal[:, :, chunk_start:chunk_end]

                cache_start = max(0, chunk_start - cur_pre_encode_cache_size)
                cache_mel = processed_signal[:, :, cache_start:chunk_start]

                if cache_mel.shape[-1] < cur_pre_encode_cache_size:
                    zeros_pad = torch.zeros(
                        (cache_mel.size(0), cache_mel.size(1), cur_pre_encode_cache_size - cache_mel.shape[-1]),
                        device=self.device,
                        dtype=cache_mel.dtype,
                    )
                    cache_mel = torch.cat([zeros_pad, cache_mel], dim=-1)

                mel_chunk = torch.cat([cache_mel, main_chunk], dim=-1)

            chunk_lengths = torch.tensor([mel_chunk.shape[-1]], dtype=torch.long, device=self.device)

            # keep_all_outputs=False so set_streaming_cuda_graphs can capture
            # subsequent steps. For chunked_limited VoiceChat this is a no-op vs
            # True: valid_out_len already equals lookahead + 1.
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = encoder.cache_aware_stream_step(
                processed_signal=mel_chunk,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )

            encoded_adapted, _ = perception.modality_adapter(audio_signal=encoded, length=encoded_len)
            encoded_chunk = perception.proj(encoded_adapted.transpose(1, 2))

            encoded_chunks.append(encoded_chunk)

        if len(encoded_chunks) > 1:
            encoded_chunk = torch.cat(encoded_chunks, dim=1)
        else:
            encoded_chunk = encoded_chunks[0]

        new_perception_cache = PerceptionCacheState(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        return encoded_chunk, new_perception_cache
