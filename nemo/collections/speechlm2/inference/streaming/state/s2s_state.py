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

import threading
from dataclasses import dataclass, field
from typing import List, Any, Optional

import torch
from nemo.collections.asr.inference.utils.text_segment import Word


@dataclass
class S2SStreamingState:
	"""
	State for streaming speech generation.

	This dataclass stores streaming tensors and counters used during
	incremental generation. It keeps initialization metadata so it can be
	reset to a clean state on demand.
	"""
	# Initialization metadata (required)
	device: torch.device
	dtype: torch.dtype
	max_len: int
	num_audio_codebooks: int
	output_sample_rate: int

	# Runtime tensors (initialized in __post_init__)
	audio_buffer: torch.Tensor = field(init=False)

	# Accumulated text output
	output_text_str: str = ""
	output_text_tokens: List[str] = field(default_factory=list)
	# Accumulated ASR text output
	output_asr_text_str: str = ""
	output_asr_text_tokens: List[str] = field(default_factory=list)
	# Accumulated function channel text output
	output_function_text_str: str = ""
	# Accumulated words with timings
	output_words: List[Word] = field(default_factory=list)
	# Final token tensors saved from the context before it is destroyed.
	# Used for post-hoc tokens_to_str / tokens_to_str_raw conversion.
	final_gen_text: Optional[torch.Tensor] = None
	final_gen_asr_text: Optional[torch.Tensor] = None
	final_total_frames: int = 0
	# RNNT user transcription (finalized at stream end)
	final_rnnt_text: Optional[str] = None
	# FC timing data (set from context at stream end)
	fc_timing: Optional[Any] = None
	# ------------------------------------------------------------------
	# TTS PAD-silence substitution — per-stream "agent idle" flag.
	#
	# Purpose:
	#   Drives the silence-substitution decision at the three TTS call sites
	#   in nemotron_voicechat_inference_wrapper.py. Answers the question
	#   "is the agent currently NOT in an open turn?" — when True, a PAD on
	#   the agent text channel corresponds to genuine silence (no turn is
	#   open, TTS has nothing to render), so it's safe to substitute silence
	#   codes. When False, a PAD might be part of the TTS rendering tail
	#   inside an open turn (Branch-B training: dense content tokens then
	#   PAD trail until EOS at the audio-end frame), so silencing would
	#   chop in-turn audio.
	#
	# Lifecycle:
	#   - session start: True (no agent turn yet — see also default field value)
	#   - BOS emitted on agent text channel: False (turn opened — don't substitute)
	#   - EOS emitted on agent text channel: True (turn closed — substitute)
	#   - PAD or any other token: no change
	#   - implicit turn end (abort_event during FC async):
	#     mark_agent_idle() — guarantees the flag isn't stuck at False
	#
	# IMPORTANT: do NOT flip this flag from cleanup_after_response() — that
	# method is called once per Triton step (every 3-frame chunk), not just at
	# session end. Flipping here corrupts in-turn TTS state. The Site-3 state
	# machine in the wrapper is the sole owner of this flag during normal
	# operation; only the abort_event hook touches it outside of Site 3.
	#
	# This field is exposed only via the locked helpers below; do not assign
	# directly. (Direct assignment is technically safe in CPython for a bool,
	# but the helpers also ensure correct fallback behavior.)
	# ------------------------------------------------------------------
	agent_idle: bool = True
	# Per-instance lock guarding agent_idle. While CPython's GIL makes bool
	# read/write atomic, the lock guards CHECK-THEN-USE patterns from the TTS
	# substitution gate (Caveat 3 — race between FC-async background thread
	# reads and main-thread BOS/EOS writes). Initialized via default_factory
	# so each S2SStreamingState instance gets its own lock — no cross-stream
	# contention.
	_agent_idle_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Allocate tensors lazily based on provided metadata."""
		with torch.no_grad():
			# Empty 2D buffer: shape (1, 0). Will be appended over time.
			self.audio_buffer = torch.empty((1, 0), device=self.device, dtype=self.dtype)

	def reset(self) -> None:
		"""Reset all tensors and counters to their initial state."""
		with torch.no_grad():
			self.audio_buffer = torch.empty((1, 0), device=self.device, dtype=self.dtype)
			self.output_text_str = ""
			self.output_text_tokens.clear()
			self.output_asr_text_str = ""
			self.output_asr_text_tokens.clear()
			self.output_function_text_str = ""
			self.output_words.clear()
			self.final_gen_text = None
			self.final_gen_asr_text = None
			self.final_total_frames = 0
		# Reset agent_idle under the lock for thread-safety with TTS reads.
		with self._agent_idle_lock:
			self.agent_idle = True

	def update_state(self, processed_frames: torch.Tensor, output_text_tokens: Any = None, output_text: str | None = None, output_asr_text: str | None = None) -> None:
		"""Append new audio to the right of the buffer; token/text args are accepted for API compatibility."""
		if processed_frames is None:
			return
		if not isinstance(processed_frames, torch.Tensor):
			raise TypeError("processed_frames must be a torch.Tensor")
		with torch.no_grad():
			# Ensure 2D [1, T] layout by flattening extra dims
			append_tensor = processed_frames
			if append_tensor.dim() > 1:
				append_tensor = append_tensor.reshape(1, -1)
			elif append_tensor.dim() == 1:
				append_tensor = append_tensor.unsqueeze(0)
			prior_samples = int(self.audio_buffer.shape[-1])
			appended_samples = int(append_tensor.shape[-1])
			self.audio_buffer = torch.cat([self.audio_buffer, append_tensor.to(self.device, dtype=self.dtype)], dim=-1)

		# Accumulate text output if provided and create a Word with naive timing
		if isinstance(output_text, str) and output_text:
			self.output_text_tokens.append(output_text)
			# Directly concatenate - spacing is already handled by tokenizer (Ġ → space)
			self.output_text_str += output_text
			try:
				if appended_samples > 0 and self.output_sample_rate > 0:
					start_t = float(prior_samples) / float(self.output_sample_rate)
					end_t = float(prior_samples + appended_samples) / float(self.output_sample_rate)
					self.output_words.append(Word(text=output_text, start=start_t, end=end_t, conf=1.0))
			except Exception:
				pass
		
		if isinstance(output_asr_text, str) and output_asr_text:
			self.output_asr_text_tokens.append(output_asr_text)
			self.output_asr_text_str += output_asr_text

	@property
	def speech_frames(self) -> List[torch.Tensor]:
		"""Backward-compatible view for code expecting a list of chunks."""
		return [self.audio_buffer]

	def get_output_text(self) -> str:
		"""Return accumulated text as a single string."""
		return self.output_text_str

	def get_output_asr_text(self) -> str:
		"""Return accumulated ASR text as a single string."""
		return self.output_asr_text_str

	def get_output_function_text(self) -> str:
		"""Return accumulated function channel text as a single string."""
		return self.output_function_text_str

	def get_output_words(self) -> List[Word]:
		"""Return accumulated words with timings."""
		return list(self.output_words)

	def save_token_tensors(self, gen_text: torch.Tensor, gen_asr_text: torch.Tensor, total_frames: int,
						   gen_function_text: torch.Tensor = None) -> None:
		"""Snapshot the full token-ID tensors from the context before it is destroyed."""
		with torch.no_grad():
			self.final_gen_text = gen_text[:, :total_frames].clone().cpu()
			self.final_gen_asr_text = gen_asr_text[:, :total_frames].clone().cpu()
			self.final_total_frames = total_frames
			self.final_gen_function_text = (
				gen_function_text[:, :total_frames].clone().cpu()
				if gen_function_text is not None else None
			)

	def get_token_tensors(self) -> Optional[tuple]:
		"""Return (gen_text, gen_asr_text, total_frames[, gen_function_text]) or None if not saved."""
		if self.final_gen_text is None:
			return None
		return self.final_gen_text, self.final_gen_asr_text, self.final_total_frames, getattr(self, 'final_gen_function_text', None)

	def cleanup_after_response(self) -> None:
		"""Clear transient audio; keep token workspaces allocated.

		Do NOT touch `agent_idle` here — this method is called once per
		Triton step (every 3-frame chunk) from base_pipeline.py, NOT just at
		session end. Flipping `agent_idle` here caused in-turn PAD frames
		to be silenced (which corrupts the TTS feedback loop via prev_audio_tokens
		and cuts off audio mid-word). The Site-3 state machine in the wrapper
		(BOS → False, EOS → True, read from gen_text) is the sole owner of
		this flag during normal operation.
		"""
		with torch.no_grad():
			self.audio_buffer = torch.empty((1, 0), device=self.device, dtype=self.dtype)

	# ------------------------------------------------------------------
	# agent_idle helpers (Option B + Caveats 2/3 hardening).
	# ------------------------------------------------------------------
	def mark_agent_idle(self) -> None:
		"""Idempotent setter: agent has no open turn → agent_idle=True.

		Call from places where the agent turn ends WITHOUT a natural EOS on
		the agent text channel — e.g. user barge-in, abort_event, session
		boundary. Safe to call repeatedly.
		"""
		with self._agent_idle_lock:
			self.agent_idle = True

	def mark_agent_active(self) -> None:
		"""Idempotent setter: agent has an open turn → agent_idle=False.

		Normally set automatically by Site 3 on BOS, but exposed for
		explicit hooks (e.g., force-turn-taking injecting BOS).
		"""
		with self._agent_idle_lock:
			self.agent_idle = False

	def is_agent_idle(self) -> bool:
		"""Locked read for use by TTS substitution gate."""
		with self._agent_idle_lock:
			return self.agent_idle
