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

import torch

from nemo.collections.speechlm2.inference.model_wrappers.decode_state import InferenceStepResult, StreamingDecodeState


class S2SContextManager:
    """Manages the lifecycle of model-level decode state for S2S streaming inference.

    Each active stream gets a :class:`StreamingDecodeState` that holds LLM KV
    caches, TTS KV caches, perception cache, codec cache, token workspaces
    (``gen_text``, ``gen_asr_text``), and ``frame_idx``.  These are created by
    the model wrapper's ``create_decode_state()`` and mutated in-place by
    ``infer_one_step()``.

    This class is kept separate from the pipeline-level output accumulator
    (:class:`S2SStreamingOutput`) so that the heavy GPU tensors inside
    ``StreamingDecodeState`` can be released as soon as a stream finishes,
    independently of how long the caller holds on to the accumulated text
    and audio results.

    :meth:`get_context` lazily creates a context on first access,
    :meth:`reset_streams` destroys contexts at end-of-stream, and
    :meth:`reset` destroys all contexts.  A stream ID may be reused
    after its context has been destroyed.
    """

    def __init__(
        self,
        s2s_model,
        max_len: int,
    ):
        self.s2s_model = s2s_model
        self.max_len = max_len
        self.device = getattr(self.s2s_model, "device", torch.device("cpu"))
        self.dtype = getattr(self.s2s_model, "dtype", torch.float32)

        self._contexts: dict[int, StreamingDecodeState] = {}

    def reset(self) -> None:
        """Release all contexts and start fresh."""
        self._contexts.clear()

    @property
    def active_stream_ids(self) -> set[int]:
        """Stream IDs that currently have an active decode context."""
        return set(self._contexts.keys())

    def _create_context(self) -> StreamingDecodeState:
        """Allocate a fresh context backed by the realtime inference model."""
        if not hasattr(self.s2s_model, "create_decode_state"):
            raise RuntimeError("s2s_model must provide create_decode_state(max_len)")
        return self.s2s_model.create_decode_state(self.max_len)

    def get_context(self, stream_ids: list[int]) -> StreamingDecodeState:
        """Return the decode context for the given stream IDs, creating if needed."""
        if len(stream_ids) == 0:
            return self._create_context()
        if len(stream_ids) != 1:
            raise NotImplementedError("get_context currently supports batch_size == 1")

        stream_id = stream_ids[0]
        if stream_id not in self._contexts:
            self._contexts[stream_id] = self._create_context()

        return self._contexts[stream_id]

    def get_context_for_stream(self, stream_id: int) -> StreamingDecodeState | None:
        """Return the decode context for a single stream, or *None* if absent."""
        return self._contexts.get(stream_id)

    def update_context(
        self,
        stream_ids: list[int],
        step_result: InferenceStepResult,
        num_frames: int,
    ) -> None:
        """Advance frame counter and set subword mask after an inference step.

        All cache and tensor mutations (dynamic_cache, past_key_values, code,
        perception_cache, codec_cache, gen_text, gen_asr_text, etc.) are
        already applied in-place on the ``StreamingDecodeState`` by
        ``infer_one_step``.  This method only bumps ``frame_idx`` and marks
        the subword mask for the newly generated frames.
        """
        if len(stream_ids) == 0:
            return
        if len(stream_ids) != 1:
            raise NotImplementedError("update_context currently supports batch_size == 1")

        stream_id = stream_ids[0]
        context = self._contexts.get(stream_id)
        if context is None:
            raise RuntimeError(f"Stream {stream_id} is not registered in the context manager")

        start_idx = context.frame_idx
        end_idx = start_idx + num_frames
        if end_idx > context.gen_text.shape[1]:
            raise RuntimeError(
                "Context maximum length exceeded. Consider increasing `streaming.max_len` in the configuration."
            )

        context.frame_idx = end_idx

        if context.subword_mask is not None:
            context.subword_mask[:, start_idx:end_idx] = True

    def reset_streams(self, stream_ids: list[int], eos_flags: list[bool]) -> None:
        """Release contexts for streams that signalled end-of-stream."""
        if len(stream_ids) != len(eos_flags):
            raise ValueError("stream_ids and eos_flags must have the same length")
        for stream_id, eos_flag in zip(stream_ids, eos_flags):
            if eos_flag and stream_id in self._contexts:
                del self._contexts[stream_id]
