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

import os
import time

import torch
import librosa
from typing import List, Optional
from torch import Tensor
import soundfile as sf
from omegaconf import DictConfig
import math

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedFrameStreamer
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.utils.progressbar import ProgressBar
from nemo.collections.speechlm2.inference.pipelines.s2s_pipeline_interface import S2SPipelineInterface
from nemo.collections.speechlm2.inference.streaming.state.s2s_state import S2SStreamingState
from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import NemotronVoicechatInferenceWrapper, tokens_to_str_raw
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.inference.streaming.state.s2s_context_manager import S2SContextManager
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions
from nemo.collections.speechlm2.inference.utils.pipeline_utils import PipelineOutput
from nemo.utils import logging


class StreamingS2SPipeline(S2SPipelineInterface):
	"""
	Streaming S2S pipeline.
	"""

	def __init__(self, cfg: DictConfig, s2s_model: NemotronVoicechatInferenceWrapper):
		# ------------------------------------------------------------------
		# Model & device
		# ------------------------------------------------------------------
		self.s2s_model = s2s_model
		self.device = self.s2s_model.device

		# ------------------------------------------------------------------
		# Streaming configuration
		# ------------------------------------------------------------------
		self.streaming_cfg = cfg.get("streaming", {})
		self.input_sample_rate = getattr(self.streaming_cfg, "input_sample_rate", 16000)
		self.output_sample_rate = getattr(self.streaming_cfg, "output_sample_rate", 22050)
		self.batch_size = getattr(self.streaming_cfg, "batch_size", 1)
		self.max_len = getattr(self.streaming_cfg, "max_len", 200)
		

		# ------------------------------------------------------------------
		# Chunk & buffer sizes
		# Terminology: "frame" = 80ms audio unit, "chunk" = 1 or more frames
		# A chunk is the amount of audio that is processed per inference step.
		# ------------------------------------------------------------------
		self.chunk_size_in_secs = getattr(self.streaming_cfg, "chunk_size_in_secs", 0.08)
		# Check if self.chunk_size_in_secs is a multiple of 0.08.
		# Because of quirks of floating point arithmetic, the remainder could be either ~0 or ~0.08,
		# so we check for both cases.
		remainder = self.chunk_size_in_secs % 0.08
		if not (math.isclose(remainder, 0, abs_tol=1e-9) or math.isclose(remainder, 0.08, abs_tol=1e-9)):
			raise ValueError(f"Chunk size must be a multiple of 0.08s, but got {self.chunk_size_in_secs}")

		self.num_frames_per_chunk = int(self.chunk_size_in_secs / 0.08)

		# Buffer size determines how much audio is passed to the perception encoder
		# Default: 5.68 seconds (71 * 0.08). This is the minimum valid buffer size without the perception cache.
		# i.e. att_context_size[0] + att_context_size[1] + 1 frames = 70+0+1 = 71 frames = 5.68 seconds
		self.buffer_size_in_secs = getattr(self.streaming_cfg, "buffer_size_in_secs", 71 * 0.08)

		self.att_context_size = getattr(self.streaming_cfg, "att_context_size", [70,0])

		# ------------------------------------------------------------------
		# bufferer – reused from ASR utilities
		# ------------------------------------------------------------------
		self.bufferer = BatchedAudioBufferer(
			sample_rate=self.input_sample_rate,
			buffer_size_in_secs=self.buffer_size_in_secs,
		)

		# ------------------------------------------------------------------
		# System prompt configuration
		# ------------------------------------------------------------------
		s2s_cfg = cfg.get("s2s", {})
		self.system_prompt: Optional[str] = getattr(s2s_cfg, "system_prompt", None)
		if self.system_prompt:
			logging.info(f"System prompt configured: {self.system_prompt[:100]}{'...' if len(self.system_prompt) > 100 else ''}")

		# Context manager
		self.context_manager = S2SContextManager(
			s2s_model=self.s2s_model,
			num_slots=self.batch_size,
			max_len=self.max_len,
		)

		# Output directory for generated files
		self.output_dir = getattr(cfg, "output_dir", "./generated")

		# Parse and validate request type early, with a safe default
		req_type_cfg = getattr(self.streaming_cfg, "request_type", "frame")

		# Parse and validate the request type; only 'frame' is supported for s2s.
		self.request_type = RequestType.from_str(req_type_cfg)
		if self.request_type is not RequestType.FRAME:
			raise ValueError(f"Request type {self.request_type} is not supported for s2s.")

		self._stream_has_prompt: bool = False

		# ------------------------------------------------------------------
		# Input audio padding (silence appended after real audio)
		# ------------------------------------------------------------------
		self.pad_audio_to_sec: float | None = cfg.get("pad_audio_to_sec", None)
		self.pad_silence_ratio: float | None = cfg.get("pad_silence_ratio", None)
		self.pad_audio_by_sec: float | None = cfg.get("pad_audio_by_sec", None)
		if sum(x is not None for x in [self.pad_audio_to_sec, self.pad_silence_ratio, self.pad_audio_by_sec]) > 1:
			raise ValueError("Set at most one of: pad_audio_to_sec, pad_silence_ratio, pad_audio_by_sec")

		super().__init__()

	# --------------------------------  ----------------------------------
	# State helpers
	# ------------------------------------------------------------------
	def create_state(self) -> S2SStreamingState:
		"""Create new empty state."""
		num_audio_codebooks = getattr(self.s2s_model.model, "_num_codebooks", 1)
		dtype = getattr(self.s2s_model, "compute_dtype", torch.float32)
		state = S2SStreamingState(
			device=self.device,
			dtype=dtype,
			max_len=self.max_len,
			num_audio_codebooks=num_audio_codebooks,
			output_sample_rate=self.output_sample_rate,
		)
		return state


	# ------------------------------------------------------------------
	# Output helpers
	# ------------------------------------------------------------------
	def log_output(self, frames: List[Frame], audio_wave: Tensor, ready_feats: List[bool], text_pieces: List[str], asr_text_pieces: List[str] = None):
		"""Append generated audio waveform and text to per-stream state."""
		for idx, frame in enumerate(frames):
			if not ready_feats[idx]:
				continue
			state = self.get_or_create_state(frame.stream_id)
			# audio_wave is [B, S]; take sample idx
			sample_audio = audio_wave[idx:idx+1, ...]
			# Determine text piece for this index
			piece = None
			if text_pieces and idx < len(text_pieces):
				candidate = text_pieces[idx]
				if isinstance(candidate, str) and candidate:
					piece = candidate
			
			# Determine ASR text piece
			asr_piece = None
			if asr_text_pieces and idx < len(asr_text_pieces):
				candidate = asr_text_pieces[idx]
				if isinstance(candidate, str) and candidate:
					asr_piece = candidate

			state.update_state(sample_audio, output_text=piece, output_asr_text=asr_piece)


	def inner_generate_step(self, frames: List[Frame], buffers: List[Tensor], left_paddings: List[int], ready_feats: List[bool]):
		"""Generate speech for chunks in *batch* using a shared ContextManager."""
		if len(frames) == 0:
			return

		stream_ids = [f.stream_id for f in frames]
		eos_flags = [f.is_last for f in frames]
		bos_flags = [f.is_first for f in frames]

		logging.debug(f"stream_ids={stream_ids} bos_flags={bos_flags} eos_flags={eos_flags}")

		if len(frames) != 1:
			raise NotImplementedError("NemotronVoicechatInferenceWrapper currently supports batch_size == 1")

		# If this is the first audio frame and prefill was already done via a
		# zero-length prefill frame, skip context init -- it's already set up.
		# Otherwise (no system prompt), create a fresh context_manager.
		has_prompt = False
		if bos_flags[0]:
			if self._stream_has_prompt:
				logging.debug(f"Prefill already done for stream {stream_ids[0]}, skipping context init")
			else:
				logging.debug(f"No prefill for stream {stream_ids[0]}, creating fresh context_manager")
				self.context_manager = S2SContextManager(
					s2s_model=self.s2s_model,
					num_slots=self.batch_size,
					max_len=self.max_len,
				)

		has_prompt = self._stream_has_prompt
		self._stream_has_prompt = False
		
		request_id = self._request_id_for_stream(stream_ids[0])
		
		context, _ = self.context_manager.get_context(stream_ids)

		audio_buffer = buffers[0]
		if audio_buffer.dim() == 1:
			audio_buffer = audio_buffer.unsqueeze(0)
		audio_buffer = audio_buffer.to(self.s2s_model.device, dtype=self.s2s_model.dtype)
		
		# Trim the buffer to exclude left padding (zeros at the beginning before buffer is filled)
		left_pad = left_paddings[0]
		if left_pad > 0:
			audio_buffer = audio_buffer[:, left_pad:]

		result = self.s2s_model.infer_one_step(
			audio_input=audio_buffer,
			num_frames_per_chunk=self.num_frames_per_chunk,
			frame_idx=context.frame_idx,
			gen_text=context.gen_text,
			audio_toks_buffer=context.audio_toks_buffer,
			input_embeds_history=context.input_embeds_history,
			dynamic_cache=context.dynamic_cache,
			past_key_values=context.past_key_values,
			code=context.code,
			subword_mask=context.subword_mask,
			gen_asr_text=context.gen_asr_text,
			gen_function_text=context.gen_function_text,
			request_id=request_id,
			perception_cache=context.perception_cache,
			has_prompt=has_prompt,
			codec_cache=context.codec_cache,
		)

		# Persist updated cache & clean finished streams
		self.context_manager.update_context(stream_ids, result, self.num_frames_per_chunk)

		# Save full token tensors to state before the context is destroyed,
		# so we can run tokens_to_str / tokens_to_str_raw post-hoc.
		for stream_id, eos_flag in zip(stream_ids, eos_flags):
			if eos_flag:
				ctx = self.context_manager.slot_contexts[
					self.context_manager.streamidx2slotidx[stream_id]
				]
				if ctx is not None:
					state = self.get_or_create_state(stream_id)
					state.save_token_tensors(ctx.gen_text, ctx.gen_asr_text, ctx.frame_idx,
											 gen_function_text=ctx.gen_function_text)

		self.context_manager.reset_slots(stream_ids, eos_flags)
		
		# Explicitly clean up bufferer and state for finished streams
		for stream_id, eos_flag in zip(stream_ids, eos_flags):
			if eos_flag:
				logging.debug(f"Ending stream {stream_id} - cleaning up bufferer and context")
				self.bufferer.rm_bufferer(stream_id)
				self._abort_stream_request(stream_id)
				# Note: We keep the state in _state_pool until finalization to save audio
				# It will be cleaned up in close_session()
		
		# Log audio and attach text to state
		self.log_output(frames, result["decoded_audio_new"], ready_feats, result["predicted_text_strs"], result.get("asr_predicted_text_strs"))

	def prefill_for_new_stream(self, stream_id: int, system_prompt: str | None = None) -> bool:
		"""Prepare the pipeline for a new stream by resetting context and prefilling the system prompt.

		This is the public API for prefill-only calls (e.g. from the Triton backend)
		that need to initialize TTS speaker embeddings and/or inject a system prompt
		into the LLM KV cache *without* processing any audio.

		Args:
			stream_id: Unique identifier for the new stream.
			system_prompt: System prompt text. If *None*, falls back to
				the YAML-configured ``self.system_prompt``.

		Returns:
			True if a system prompt was prefilled, False otherwise.
		"""
		t0 = time.time()
		if system_prompt is None:
			system_prompt = self.system_prompt

		self.context_manager = S2SContextManager(
			s2s_model=self.s2s_model,
			num_slots=self.batch_size,
			max_len=self.max_len,
		)
		t_ctx = time.time()

		with torch.no_grad(), torch.inference_mode():
			self._prefill_system_prompt(stream_id, system_prompt)
		t_prefill = time.time()

		self._stream_has_prompt = bool(system_prompt)
		logging.debug(f"prefill_for_new_stream: context_manager={1000*(t_ctx-t0):.1f}ms, "
			  f"_prefill_system_prompt={1000*(t_prefill-t_ctx):.1f}ms, "
			  f"total={1000*(t_prefill-t0):.1f}ms, has_prompt={self._stream_has_prompt}")
		return self._stream_has_prompt

	_WARMUP_FALLBACK_PROMPT = "Mock system prompt for warmup."

	def warmup(self, system_prompt: str | None = None) -> None:
		"""Run a throwaway prefill cycle to warm up the inference engine.

		The very first prefill incurs one-time overhead (e.g. CUDA graph
		compilation, memory pool allocation, DynamicCache initialization).
		Calling this once during startup moves that cost out of the
		critical path so the first real client request is fast.

		The method performs a full prefill (TTS speaker embedding + LLM
		system prompt), then aborts the request and resets all pipeline
		state so the next real stream starts cleanly.

		Args:
			system_prompt: Prompt text to use for warmup.  Falls back to
				the YAML-configured ``self.system_prompt``, then to a
				short fallback string so the LLM prefill path is always
				exercised.
		"""
		prompt = system_prompt if system_prompt is not None else self.system_prompt
		if not prompt:
			prompt = self._WARMUP_FALLBACK_PROMPT
			logging.info(f"No system prompt configured — using fallback prompt for warmup: \"{prompt}\"")

		warmup_stream_id = -1

		logging.info("Running pipeline warmup prefill...")
		t0 = time.time()

		self.prefill_for_new_stream(warmup_stream_id, prompt)

		# Tear down the warmup request so the engine is clean for real traffic
		self._abort_stream_request(warmup_stream_id)
		self.context_manager.reset()
		self._stream_has_prompt = False

		logging.info(f"Pipeline warmup complete in {time.time() - t0:.3f}s")

	def generate_step(self, frames: List[Frame]):
		"""Main streaming API similar to *transcribe_step* in recognizers.

		If the batch contains a single zero-length first frame with a system
		prompt in ``options``, this is treated as a **prefill-only** request:
		the context manager and system prompt are initialized but no audio
		inference runs.  This is the unified protocol used by both the CLI
		(``run()``) and the Triton backend.
		"""
		# Detect prefill-only frame: is_first + zero-length audio
		if (len(frames) == 1
				and frames[0].is_first
				and frames[0].samples.numel() == 0):
			opts = frames[0].options
			prompt = None
			if opts is not None and hasattr(opts, "system_prompt"):
				prompt = opts.system_prompt
			self.prefill_for_new_stream(frames[0].stream_id, prompt)
			return

		buffers, left_paddings = self.bufferer.update(frames)
		ready_feats = [True] * len(frames)

		with torch.no_grad(), torch.inference_mode():
			self.inner_generate_step(frames, buffers, left_paddings, ready_feats)
		
	# ------------------------------------------------------------------
	# Finalization helpers
	# ------------------------------------------------------------------
	def _finalize_and_save_finished_streams(
		self,
		frames: List[Frame],
		audio_filepaths: List[str],
		saved_paths_by_stream: dict[int, str],
	) -> None:
		"""Finalize any streams that ended in this batch and save their audio."""
		for frame in frames:
			if frame.is_last:
				stream_id = frame.stream_id
				state = self.get_or_create_state(stream_id)

				# Flush remaining buffered samples and assemble waveform
				if hasattr(state, "finalize"):
					state.finalize()
				# Concatenate emitted chunks and squeeze (B=1,C=1) to mono waveform
				generated_audio = torch.cat(state.speech_frames, dim=-1)
				# Ensure 1D mono waveform and float32 dtype for soundfile
				if generated_audio.dim() == 3 and generated_audio.size(0) == 1 and generated_audio.size(1) == 1:
					generated_audio = generated_audio.squeeze(0).squeeze(0)
				elif generated_audio.dim() == 2 and generated_audio.size(0) == 1:
					generated_audio = generated_audio.squeeze(0)
				generated_audio = generated_audio.to(torch.float32)

				# Build output paths in subdirectories under output_dir
				in_path = audio_filepaths[stream_id]
				base = os.path.splitext(os.path.basename(in_path))[0]

				wav_dir = os.path.join(self.output_dir, "wav")
				stereo_dir = os.path.join(self.output_dir, "stereo")
				txt_dir = os.path.join(self.output_dir, "txt")
				os.makedirs(wav_dir, exist_ok=True)
				os.makedirs(stereo_dir, exist_ok=True)
				os.makedirs(txt_dir, exist_ok=True)

				out_path = os.path.join(wav_dir, f"{base}.wav")

				# Write audio to disk
				if generated_audio.numel() > 0:
					sf.write(out_path, generated_audio.detach().cpu().numpy(), self.output_sample_rate)

				# Also save a stereo file with input (ch0) and output (ch1)
				# Load input with librosa (handles mono conversion and resampling)
				input_np, _ = librosa.load(in_path, sr=self.output_sample_rate, mono=True)
				input_audio = torch.from_numpy(input_np).to(torch.float32)
				gen_cpu = generated_audio.detach().cpu().to(input_audio.dtype)

				# Prepend silence to output channel to account for
				# the one-chunk processing delay: the server can't
				# produce output until it has received a full input chunk.
				delay_samples = int(self.chunk_size_in_secs * self.output_sample_rate)
				silence = torch.zeros(delay_samples, dtype=gen_cpu.dtype)
				gen_cpu = torch.cat([silence, gen_cpu], dim=-1)

				gen_len = int(gen_cpu.shape[-1])
				in_len = int(input_audio.shape[-1])
				max_len = max(gen_len, in_len)
				if in_len < max_len:
					input_audio = torch.cat([input_audio, torch.zeros(max_len - in_len, dtype=input_audio.dtype)], dim=-1)
				if gen_len < max_len:
					gen_cpu = torch.cat([gen_cpu, torch.zeros(max_len - gen_len, dtype=gen_cpu.dtype)], dim=-1)
				stereo = torch.stack([input_audio, gen_cpu], dim=0).transpose(0, 1)
				stereo_path = os.path.join(stereo_dir, f"{base}_input_output.wav")
				sf.write(stereo_path, stereo.detach().cpu().numpy(), self.output_sample_rate)

				# Save accumulated text
				text_out = state.get_output_text() if hasattr(state, "get_output_text") else ""
				if isinstance(text_out, str):
					try:
						with open(os.path.join(txt_dir, f"{base}.txt"), "w", encoding="utf-8") as f:
							f.write(text_out)
					except Exception:
						pass

				# Save accumulated ASR text
				asr_text_out = state.get_output_asr_text() if hasattr(state, "get_output_asr_text") else ""
				if isinstance(asr_text_out, str) and asr_text_out:
					try:
						with open(os.path.join(txt_dir, f"{base}_asr.txt"), "w", encoding="utf-8") as f:
							f.write(asr_text_out)
					except Exception:
						pass

				saved_paths_by_stream[stream_id] = out_path

				# Keep state until outputs are assembled; will be cleared on close_session


	# ------------------------------------------------------------------
	# Session helpers (extend S2SPipelineInterface)
	# ------------------------------------------------------------------

	def reset_session(self) -> None:
		"""Reset feature buffer and ContextManager together."""
		for stream_id in list(self.context_manager.streamidx2slotidx.keys()):
			self._abort_stream_request(stream_id)
		self.bufferer.reset()
		self.context_manager.reset()

		super().reset_session() # clears state pool

	# ------------------------------------------------------------------
	# Orchestrator – mirrors recognizers' *run* method
	# ------------------------------------------------------------------
	def run(
		self,
		audio_filepaths: List[str],
		options: List[S2SRequestOptions] | None = None,
		progress_bar: Optional[ProgressBar] = None,
	) -> PipelineOutput:
		"""Stream all *audio_filepaths* through the pipeline and save outputs.

		Saves one generated ``.wav`` per input under ``self.output_dir`` and
		returns their paths in ``PipelineOutput.texts``.
		"""
		if progress_bar and not isinstance(progress_bar, ProgressBar):
			raise ValueError("progress_bar must be an instance of ProgressBar.")

		if options is None:
			options = [S2SRequestOptions(system_prompt=self.system_prompt) for _ in audio_filepaths]

		streamer = ContinuousBatchedFrameStreamer(
			n_frames_per_stream=1,
			frame_size_in_secs=self.chunk_size_in_secs,
			sample_rate=self.input_sample_rate,
			batch_size=self.batch_size,
			pad_last_frame=True,
		)
		
		streamer.set_audio_filepaths(audio_filepaths, options)
		streamer.set_progress_bar(progress_bar)

		# Ensure output directory exists
		os.makedirs(self.output_dir, exist_ok=True)

		# Track saved paths by stream id to preserve input order
		saved_paths_by_stream: dict[int, str] = {}
		chunk_samples = int(self.chunk_size_in_secs * self.input_sample_rate)

		self.open_session()
		for frames in streamer:
			# Unified prefill protocol: if the first frame of a new stream
			# carries a system prompt, emit a zero-length prefill frame first.
			if (len(frames) == 1
					and frames[0].is_first
					and frames[0].options is not None
					and hasattr(frames[0].options, "system_prompt")
					and frames[0].options.system_prompt):
				prefill_frame = Frame(
					samples=torch.empty(0),
					stream_id=frames[0].stream_id,
					is_first=True,
					is_last=False,
					options=frames[0].options,
				)
				self.generate_step([prefill_frame])

			# If padding is configured, intercept last frames so the
			# bufferer/context stay alive for the silence-padding phase.
			# Padding is generated immediately (same iteration) to avoid
			# the next stream's setup destroying this stream's context.
			pad_targets: dict[int, float] = {}
			if self.pad_audio_to_sec or self.pad_silence_ratio or self.pad_audio_by_sec:
				processed_frames = []
				for frame in frames:
					if frame.is_last:
						elapsed = streamer.elapsed_durations[frame.stream_id]
						remaining = self._padding_remaining_secs(elapsed)
						if remaining > 0:
							processed_frames.append(Frame(
								samples=frame.samples,
								stream_id=frame.stream_id,
								is_first=frame.is_first,
								is_last=False,
								length=frame.length,
								options=frame.options,
							))
							pad_targets[frame.stream_id] = remaining
							continue
					processed_frames.append(frame)
				frames = processed_frames

			self.generate_step(frames)
			self._finalize_and_save_finished_streams(frames, audio_filepaths, saved_paths_by_stream)

			# Generate silence padding before the next iteration adds a new stream
			for stream_id, remaining_secs in pad_targets.items():
				num_pad_frames = max(1, round(remaining_secs / self.chunk_size_in_secs))
				for i in range(num_pad_frames):
					is_last = (i == num_pad_frames - 1)
					silence_frame = Frame(
						samples=torch.zeros(chunk_samples),
						stream_id=stream_id,
						is_first=False,
						is_last=is_last,
						length=chunk_samples,
					)
					self.generate_step([silence_frame])
					if is_last:
						self._finalize_and_save_finished_streams(
							[silence_frame], audio_filepaths, saved_paths_by_stream
						)
		# Build outputs before closing the session
		texts = []
		words = []
		asr_texts = []
		texts_with_timestamps = []
		asr_texts_with_timestamps = []
		raw_texts = []
		raw_asr_texts = []

		tokenizer = self.s2s_model.tokenizer
		pad_id = self.s2s_model.model.stt_model.text_pad_id

		for idx in range(len(audio_filepaths)):
			state = self.get_or_create_state(idx)
			text_value = state.get_output_text() if hasattr(state, "get_output_text") else ""
			if not text_value:
				text_value = saved_paths_by_stream.get(idx, "")
			texts.append(text_value)
			per_stream_words = state.get_output_words() if hasattr(state, "get_output_words") else []
			words.append(per_stream_words)
			asr_text_value = state.get_output_asr_text() if hasattr(state, "get_output_asr_text") else ""
			asr_texts.append(asr_text_value)

			token_data = state.get_token_tensors()
			if token_data is not None:
				gen_text, gen_asr_text, total_frames, gen_function_text = token_data
				lengths = torch.tensor([total_frames], dtype=torch.long)
				texts_with_timestamps.append(
					tokens_to_str(gen_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=True)[0]
				)
				asr_texts_with_timestamps.append(
					tokens_to_str(gen_asr_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=True)[0]
				)
				raw_texts.append(
					tokens_to_str_raw(gen_text, lengths, tokenizer=tokenizer, pad_id=pad_id)[0]
				)
				raw_asr_texts.append(
					tokens_to_str_raw(gen_asr_text, lengths, tokenizer=tokenizer, pad_id=pad_id)[0]
				)
				if gen_function_text is not None:
					fc_text = tokens_to_str(gen_function_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=False)[0]
					fc_text_raw = tokens_to_str_raw(gen_function_text, lengths, tokenizer=tokenizer, pad_id=pad_id)[0]
					logging.info(f"Function calling channel: {fc_text}")
			else:
				texts_with_timestamps.append("")
				asr_texts_with_timestamps.append("")
				raw_texts.append("")
				raw_asr_texts.append("")

		self.close_session()

		return PipelineOutput(
			texts=texts,
			words=words,
			asr_texts=asr_texts,
			texts_with_timestamps=texts_with_timestamps,
			asr_texts_with_timestamps=asr_texts_with_timestamps,
			raw_texts=raw_texts,
			raw_asr_texts=raw_asr_texts,
		)

	def _prefill_system_prompt(self, stream_id: int, system_prompt: str | None = None) -> Optional[torch.Tensor]:
		"""Prefill the system prompt for a new stream.
		
		This prepares the system prompt embeddings and processes them through
		the LLM to update the KV cache before audio streaming begins.
		Also prefills the TTS model with speaker embeddings when using vLLM EarTTS.
		
		Args:
			stream_id: The stream identifier.
			system_prompt: The system prompt text for this stream. If *None*,
				TTS prefill still runs (for vLLM EarTTS) but no LLM prompt
				is injected.

		Note on TTS prefill codes:
			The TTS prefill generates output codes, but these should NOT be used
			to initialize context.code for inference. The batch approach uses
			first_tts_code_input (INPUT codes from speaker reference) instead.
			Using prefill OUTPUT codes causes audio quality issues (mumbling).
		
		Returns:
			Optional[torch.Tensor]: The TTS prefill output codes if vLLM EarTTS prefill
			happened, None otherwise. These are returned for logging/debugging but
			should NOT be used to update context.code.
		"""
		request_id = self._request_id_for_stream(stream_id)
		engine_type = getattr(self.s2s_model, "engine_type", "native")
		tts_output_code = None
		
		# Prefill TTS with speaker embedding when using vLLM EarTTS
		# This initializes the vLLM TTS engine with the speaker context via prompt_token_ids
		use_vllm_eartts = "vllm_eartts" in engine_type.lower()
		if use_vllm_eartts:
			tts_init_inputs = getattr(self.s2s_model, "tts_init_inputs", None)
			tts_prompt_token_ids = getattr(self.s2s_model, "tts_prompt_token_ids", None)
			if tts_init_inputs is not None and tts_prompt_token_ids is not None:
				logging.info(f"Prefilling TTS speaker embedding for stream {stream_id}...")
				start_tts_prefill = time.time()
				with torch.no_grad():
					# Clone tts_init_inputs to avoid any tensor sharing issues
					import copy
					tts_inputs_copy = copy.deepcopy(tts_init_inputs)
					tts_result = self.s2s_model.model.tts_model.tts_model(
						tts_inputs_copy,
						request_id=request_id,
						prompt_token_ids=tts_prompt_token_ids
					)
					# Capture the generated codes to sync context with vLLM state
					if hasattr(tts_result, 'codes') and tts_result.codes is not None:
						tts_output_code = tts_result.codes.detach().clone()
						logging.debug(f"TTS prefill generated codes shape: {tts_output_code.shape}")
				logging.info(f"TTS speaker embedding prefilled in {time.time() - start_tts_prefill:.3f}s")
			else:
				logging.warning("TTS init inputs not available, skipping TTS prefill")
		
		if not system_prompt:
			return tts_output_code
		
		logging.info(f"Prefilling system prompt for stream {stream_id}...")
		start_get_prompt_embeddings = time.time()
		prompt_embedded, prompt_len = self.s2s_model._prepare_system_prompt_embeddings(system_prompt)
		logging.debug(f"Time taken to get prompt embeddings: {time.time() - start_get_prompt_embeddings:.3f}s")
		
		if prompt_embedded is None:
			logging.warning("System prompt embedding returned None, skipping prefill")
			return tts_output_code
		
		# Check if using vLLM for LLM (matches vllm_llm, vllm_llm_vllm_eartts, etc.)
		use_vllm_llm = "vllm_llm" in engine_type.lower()
		
		if use_vllm_llm:
			# For vLLM LLM: prefill all prompt embeddings in one shot
			# (decode_steps=0 triggers a single bulk prefill in the vLLM engine)
			logging.info(f"Prefilling {prompt_len} prompt embeddings for vLLM LLM...")
			start_prefill = time.time()
			with torch.no_grad():
				_ = self.s2s_model.model_llm_interface(
					prompt_embedded,
					request_id=request_id,
					decode_steps=0,
					prompt_token_ids=None,
				)
			logging.info(f"System prompt prefilled ({prompt_len} tokens) in {time.time() - start_prefill:.3f}s")
		
		else:
			context, _ = self.context_manager.get_context([stream_id])
			if context.dynamic_cache is not None:
				# Native cache mode: process prompt through LLM to update KV cache
				with torch.no_grad():
					llm_cache = context.dynamic_cache
					ans = self.s2s_model.model_llm_interface(
						prompt_embedded,
						cache=llm_cache,
						generated_tokens=None,
						current_step=0
					)
					context.dynamic_cache = ans.get("cache", llm_cache)
				logging.info(f"System prompt processed, cache updated ({prompt_len} tokens)")
			else:
				# No-cache mode (e.g. Nemotron): add prompt embeddings to history
				for t in range(prompt_len):
					context.input_embeds_history.append(prompt_embedded[:, t:t+1, :])
				logging.info(f"Added {prompt_len} prompt embeddings to input_embeds_history")
		
		return tts_output_code

	def _padding_remaining_secs(self, elapsed_secs: float) -> float:
		"""Return how many seconds of silence padding are still needed."""
		if self.pad_audio_to_sec is not None:
			return max(0.0, self.pad_audio_to_sec - elapsed_secs)
		if self.pad_silence_ratio is not None:
			return elapsed_secs * self.pad_silence_ratio
		if self.pad_audio_by_sec is not None:
			return self.pad_audio_by_sec
		return 0.0

	def _request_id_for_stream(self, stream_id: int) -> str:
		return str(stream_id)

	def _abort_stream_request(self, stream_id: int) -> None:
		request_id = self._request_id_for_stream(stream_id)
		abort_fn = getattr(self.s2s_model, "abort_request", None)
		if callable(abort_fn):
			try:
				abort_fn(request_id)
			except Exception as exc:
				logging.warning(f"Failed to abort request {request_id} for stream {stream_id}: {exc}")
