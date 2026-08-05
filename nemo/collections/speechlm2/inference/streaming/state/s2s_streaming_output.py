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

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.collections.speechlm2.parts.text_utils import tokens_to_str
from nemo.utils import logging


@dataclass
class S2SStreamingOutput:
    """Pipeline-level output accumulator for a single S2S stream.

    Collects the generated audio chunks, agent text, and ASR (user) text
    produced by successive ``generate_step()`` calls, concatenating lazily
    to avoid O(n^2) copying.  One instance exists per active stream and is
    stored in the pipeline's ``_state_pool``.

    At end-of-stream, :meth:`finalize_tokens` snapshots the raw token-ID
    tensors from the :class:`StreamingDecodeState` (before it is destroyed)
    and decodes them into the *finalized fields* (``text_with_timestamps``,
    ``raw_text``, etc.) in a single call.  The same object is then returned
    by ``run()`` as the final per-stream result.

    This class is **not** the model-level decode state (KV caches, token
    workspaces, perception cache).  That role belongs to
    :class:`~nemo.collections.speechlm2.inference.model_wrappers.decode_state.StreamingDecodeState`,
    which lives inside the :class:`S2SContextManager` and is mutated
    in-place by the model wrapper's ``infer_one_step()``.

    Typical lifecycle::

        # Created on is_first frame
        output = S2SStreamingOutput(device=..., dtype=..., output_sample_rate=...)

        # Each step appends incremental results
        output.append_step_output(audio_chunk, text="", asr_text="Hi")
        output.append_step_output(audio_chunk, text="", asr_text="")
        output.append_step_output(audio_chunk, text="Hello", asr_text="")

        # At end-of-stream, snapshot token tensors and decode text fields
        output.finalize_tokens(ctx.gen_text, ctx.gen_asr_text, ctx.frame_idx,
                               tokenizer=tok, pad_id=pad_id)

        # Final results read via properties / fields
        wav = output.audio_buffer        # single concatenated tensor (CPU)
        txt = output.output_text_str     # joined string
        ts  = output.text_with_timestamps  # decoded with turn-taking markers
    """

    # Required init metadata
    device: torch.device
    dtype: torch.dtype
    output_sample_rate: int

    # Per-stream request options (system prompt, sampling overrides, etc.)
    options: S2SRequestOptions = field(default_factory=S2SRequestOptions)

    # Audio chunks accumulated each step; use the ``audio_buffer`` property
    # to get a single concatenated tensor (lazy, O(n) instead of O(n^2)).
    _audio_chunks: list[torch.Tensor] = field(default_factory=list, repr=False)
    _total_audio_samples: int = field(default=0, repr=False)

    # Text parts accumulated each step; use the ``output_text_str`` /
    # ``output_asr_text_str`` properties to get joined strings.
    _text_parts: list[str] = field(default_factory=list, repr=False)
    _asr_text_parts: list[str] = field(default_factory=list, repr=False)

    # Per-step debug data (logits, embeddings, etc.) when collect_debug is on.
    debug_data: list[dict] = field(default_factory=list, repr=False)

    # -- Finalized fields (populated by finalize_tokens() at end-of-stream) --
    token_text: torch.Tensor | None = None
    token_asr_text: torch.Tensor | None = None
    token_length: int | None = None
    text_with_timestamps: str | None = None
    asr_text_with_timestamps: str | None = None
    raw_text: str | None = None
    raw_asr_text: str | None = None
    audio_filepath: str | None = None

    @property
    def audio_buffer(self) -> torch.Tensor:
        """Concatenated audio from all steps, or loaded from disk after finalization.

        During streaming this concatenates in-memory chunks lazily.
        After ``run()`` finishes, the chunks are cleared to save memory
        and the audio is re-loaded from ``audio_filepath`` on demand.
        """
        if self._audio_chunks:
            return torch.cat(self._audio_chunks, dim=-1)
        if self.audio_filepath is not None:
            import soundfile as sf

            audio_np, _ = sf.read(self.audio_filepath, dtype="float32")
            return torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        return torch.empty((1, 0), device=self.device, dtype=self.dtype)

    @property
    def output_text_str(self) -> str:
        """Accumulated agent response text. Joined lazily to avoid O(n^2) copies."""
        return "".join(self._text_parts)

    @property
    def output_asr_text_str(self) -> str:
        """Accumulated ASR (user) text. Joined lazily to avoid O(n^2) copies."""
        return "".join(self._asr_text_parts)

    def reset(self) -> None:
        """Reset all accumulated outputs to initial state."""
        self._audio_chunks.clear()
        self._total_audio_samples = 0
        self._text_parts.clear()
        self._asr_text_parts.clear()
        self.debug_data.clear()
        self.token_text = None
        self.token_asr_text = None
        self.token_length = None
        self.text_with_timestamps = None
        self.asr_text_with_timestamps = None
        self.raw_text = None
        self.raw_asr_text = None
        self.audio_filepath = None

    def append_step_output(
        self,
        audio: torch.Tensor | None,
        text: str | None = None,
        asr_text: str | None = None,
    ) -> None:
        """Append generated audio and optional text from one inference step."""
        if audio is None:
            return
        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor")

        append_tensor = audio
        if append_tensor.dim() > 1:
            append_tensor = append_tensor.reshape(1, -1)
        elif append_tensor.dim() == 1:
            append_tensor = append_tensor.unsqueeze(0)
        self._audio_chunks.append(append_tensor.to(self.device, dtype=self.dtype))
        self._total_audio_samples += int(append_tensor.shape[-1])

        if isinstance(text, str) and text:
            self._text_parts.append(text)

        if isinstance(asr_text, str) and asr_text:
            self._asr_text_parts.append(asr_text)

    def finalize_tokens(
        self,
        gen_text: torch.Tensor,
        gen_asr_text: torch.Tensor,
        total_frames: int,
        tokenizer,
        pad_id: int,
    ) -> None:
        """Snapshot token-ID tensors from the decode context and decode them into text fields.

        Must be called at end-of-stream, before the :class:`StreamingDecodeState`
        is destroyed.
        """
        self.token_text = gen_text[:, :total_frames].clone().cpu()
        self.token_asr_text = gen_asr_text[:, :total_frames].clone().cpu()
        self.token_length = total_frames

        lengths = torch.tensor([total_frames], dtype=torch.long)

        def _to_str(tokens, **kwargs):
            return tokens_to_str(tokens, lengths, tokenizer=tokenizer, pad_id=pad_id, **kwargs)[0]

        self.text_with_timestamps = _to_str(self.token_text, eval_text_turn_taking=True)
        self.asr_text_with_timestamps = _to_str(self.token_asr_text, eval_text_turn_taking=True)
        self.raw_text = _to_str(self.token_text, keep_pad=True)
        self.raw_asr_text = _to_str(self.token_asr_text, keep_pad=True)

    def clear_audio_buffer(self) -> None:
        """Free in-memory audio chunks.

        Only drops the chunk list; ``_total_audio_samples`` is preserved
        so that CTM timing and other metadata remain valid after the
        waveform data has been written to disk.
        """
        self._audio_chunks.clear()
