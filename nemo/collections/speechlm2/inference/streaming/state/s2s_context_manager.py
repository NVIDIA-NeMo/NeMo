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

from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from transformers import DynamicCache

from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import PerceptionCacheState


@dataclass
class StreamingRealtimeContext:
	frame_idx: int
	gen_text: torch.Tensor
	gen_asr_text: torch.Tensor
	gen_function_text: Optional[torch.Tensor]
	audio_toks_buffer: Optional[torch.Tensor]
	input_embeds_history: List[torch.Tensor]
	dynamic_cache: Optional[DynamicCache]
	past_key_values: Any
	code: Optional[torch.Tensor]
	subword_mask: Optional[torch.Tensor]
	perception_cache: Optional["PerceptionCacheState"] = None
	codec_cache: Optional[CausalConv1dCache] = None


class S2SContextManager:

	def __init__(
		self,
		s2s_model,
		num_slots: int,
		max_len: int,
	):
		self.s2s_model = s2s_model
		self.num_slots = num_slots
		
		# Detect Nemotron models and disable DynamicCache
		# (they require NemotronHHybridDynamicCache which isn't supported yet)
		self.cache_disabled = False
		stt_model = getattr(self.s2s_model.model, "stt_model", None)
		if stt_model is not None:
			pretrained_llm = stt_model.cfg.get("pretrained_llm", "")
			if "Nemotron" in pretrained_llm:
				logging.warning(
					f"Detected Nemotron model ({pretrained_llm}). "
					"Disabling DynamicCache (Nemotron requires NemotronHHybridDynamicCache which is not yet supported)."
				)
				self.cache_disabled = True
		self.max_len = max_len
		self.device = getattr(self.s2s_model, "device", torch.device("cpu"))
		self.dtype = getattr(self.s2s_model, "dtype", torch.float32)
		self.text_pad_id = getattr(getattr(self.s2s_model, "model", None), "text_pad_id", 0)
		self.codec_token_history_size = int(getattr(self.s2s_model, "codec_token_history_size", 0))
		self.decode_audio = bool(getattr(self.s2s_model, "decode_audio", False))
		self.use_perception_cache = bool(getattr(self.s2s_model, "use_perception_cache", False))
		self.use_codec_cache = bool(getattr(self.s2s_model, "use_codec_cache", True))

		self.reset()

	def reset(self) -> None:
		"""Reset all bookkeeping for a new streaming session."""
		self.streamidx2slotidx: Dict[int, int] = {}
		self.slotidx2streamidx: Dict[int, int] = {}
		self.free_slots = Queue(self.num_slots)
		for i in range(self.num_slots):
			self.free_slots.put(i)
		self.slot_contexts: List[Optional[StreamingRealtimeContext]] = [None] * self.num_slots

	def _create_context(self) -> StreamingRealtimeContext:
		"""Allocate a fresh context backed by the realtime inference model."""
		gen_text = torch.full(
			(1, self.max_len),
			fill_value=self.text_pad_id,
			device=self.device,
			dtype=torch.long,
		)

		gen_asr_text = torch.full(
			(1, self.max_len),
			fill_value=self.text_pad_id,
			device=self.device,
			dtype=torch.long,
		)

		stt_model = getattr(self.s2s_model.model, "stt_model", None)
		has_function_head = stt_model is not None and getattr(stt_model, "function_head", None) is not None
		gen_function_text = None
		if has_function_head:
			gen_function_text = torch.full(
				(1, self.max_len),
				fill_value=self.text_pad_id,
				device=self.device,
				dtype=torch.long,
			)

		dynamic_cache = None if self.cache_disabled else DynamicCache()
		audio_toks_buffer: Optional[torch.Tensor] = None
		past_key_values: Any = None
		code: Optional[torch.Tensor] = None
		subword_mask: Optional[torch.Tensor] = None
		perception_cache = None
		codec_cache = None

		if self.decode_audio and hasattr(getattr(self.s2s_model, "model", None), "tts_model"):
			tts_model = self.s2s_model.model.tts_model
			if self.use_codec_cache:
				# Incremental decode path: CausalConv1dCache maintains all codec
				# context internally, so no audio_toks_buffer is needed and
				# codec_token_history_size is irrelevant.
				codec_cache = CausalConv1dCache()
			elif self.codec_token_history_size > 0:
				# Sliding-window fallback: allocate silence buffer of
				# codec_token_history_size tokens that is re-decoded every step.
				silence_tokens_base = tts_model.codec_silence_tokens.detach().clone()
				silence_tokens = silence_tokens_base.view(1, 1, -1).expand(
					-1, self.codec_token_history_size, -1
				).contiguous()  # contiguous() ensures it's a real copy, not a view
				audio_toks_buffer = silence_tokens.to(self.device).clone()
			subword_mask = torch.ones((1, self.max_len), device=self.device, dtype=torch.bool)

			if getattr(self.s2s_model, "first_tts_past_key_values_input", None) is not None:
				past_key_values = self.s2s_model._clone_cache(self.s2s_model.first_tts_past_key_values_input)
			if getattr(self.s2s_model, "first_tts_code_input", None) is not None:
				code = self.s2s_model.first_tts_code_input.detach().clone()

		# Initialize perception cache if enabled
		if self.use_perception_cache:
			mgr = getattr(self.s2s_model, "perception_cache_mgr", None)
			if mgr is not None:
				perception_cache = mgr.get_initial_state(batch_size=1)

		return StreamingRealtimeContext(
			frame_idx=0,
			gen_text=gen_text,
			gen_asr_text=gen_asr_text,
			gen_function_text=gen_function_text,
			audio_toks_buffer=audio_toks_buffer,
			input_embeds_history=[],
			dynamic_cache=dynamic_cache,
			past_key_values=past_key_values,
			code=code,
			subword_mask=subword_mask,
			perception_cache=perception_cache,
			codec_cache=codec_cache,
		)

	def _ensure_slot(self, stream_id: int) -> int:
		if stream_id not in self.streamidx2slotidx:
			if self.free_slots.empty():
				# Emergency cleanup: force-release all slots for a fresh start
				# This handles cases where previous streams didn't end properly
				# (e.g., exceptions, client disconnects, missing is_last=True)
				logging.warning(f"No free slots available - forcing cleanup of all {self.num_slots} slots")
				orphaned_streams = list(self.slotidx2streamidx.values())
				if orphaned_streams:
					logging.warning(f"Orphaned streams being cleaned up: {orphaned_streams}")
				for slot_idx in range(self.num_slots):
					self.reset_slot(slot_idx)
			slot_idx = self.free_slots.get()
			# Ensure the slot is completely clean before assigning to new stream
			if self.slot_contexts[slot_idx] is not None:
				logging.warning(f"Slot {slot_idx} was not properly cleaned. Forcing cleanup.")
				self.slot_contexts[slot_idx] = None
			self.streamidx2slotidx[stream_id] = slot_idx
			self.slotidx2streamidx[slot_idx] = stream_id
		return self.streamidx2slotidx[stream_id]

	def reset_slot(self, slot_idx: int) -> None:
		"""Release a slot back to the pool."""
		if slot_idx < 0 or slot_idx >= self.num_slots:
			return
		# Set to None to break reference and allow garbage collection
		self.slot_contexts[slot_idx] = None
		stream_id = self.slotidx2streamidx.get(slot_idx)
		if stream_id is not None:
			del self.slotidx2streamidx[slot_idx]
			del self.streamidx2slotidx[stream_id]
		self.free_slots.put(slot_idx)

	def update_context(
		self,
		stream_ids: List[int],
		step_result: Dict[str, Any],
		num_frames: int,
	) -> None:
		"""Persist model outputs back into the cached context."""
		if len(stream_ids) == 0:
			return
		if len(stream_ids) != 1:
			raise NotImplementedError("update_context currently supports batch_size == 1")

		stream_id = stream_ids[0]
		slot_idx = self.streamidx2slotidx.get(stream_id)
		if slot_idx is None:
			raise RuntimeError(f"Stream {stream_id} is not registered in the context manager")

		context = self.slot_contexts[slot_idx]
		if context is None:
			context = self._create_context()
			self.slot_contexts[slot_idx] = context

		start_idx = context.frame_idx
		end_idx = start_idx + num_frames
		if end_idx > context.gen_text.shape[1]:
			raise RuntimeError(
				"Context maximum length exceeded. Consider increasing `streaming.max_len` in the configuration."
			)

		predicted_tokens = step_result.get("predicted_text_tokens")
		if predicted_tokens is not None:
			if predicted_tokens.dim() == 1:
				token_slice = predicted_tokens.unsqueeze(0)
			else:
				token_slice = predicted_tokens[0:1]
			context.gen_text[:, start_idx:end_idx] = token_slice.to(context.gen_text.device)

		asr_predicted_tokens = step_result.get("asr_predicted_text_tokens")
		if asr_predicted_tokens is not None:
			if asr_predicted_tokens.dim() == 1:
				asr_token_slice = asr_predicted_tokens.unsqueeze(0)
			else:
				asr_token_slice = asr_predicted_tokens[0:1]
			context.gen_asr_text[:, start_idx:end_idx] = asr_token_slice.to(context.gen_asr_text.device)

		func_predicted_tokens = step_result.get("function_predicted_text_tokens")
		if func_predicted_tokens is not None and context.gen_function_text is not None:
			if func_predicted_tokens.dim() == 1:
				func_token_slice = func_predicted_tokens.unsqueeze(0)
			else:
				func_token_slice = func_predicted_tokens[0:1]
			context.gen_function_text[:, start_idx:end_idx] = func_token_slice.to(context.gen_function_text.device)

		context.frame_idx = end_idx

		if step_result.get("dynamic_cache") is not None:
			context.dynamic_cache = step_result["dynamic_cache"]
		if "audio_toks_buffer" in step_result:
			context.audio_toks_buffer = step_result["audio_toks_buffer"]
		if "input_embeds_history" in step_result:
			context.input_embeds_history = step_result["input_embeds_history"]
		if "past_key_values" in step_result:
			context.past_key_values = step_result["past_key_values"]
		if "code" in step_result:
			context.code = step_result["code"]
		if context.subword_mask is not None:
			context.subword_mask[:, start_idx:end_idx] = True
		if "perception_cache" in step_result and step_result["perception_cache"] is not None:
			context.perception_cache = step_result["perception_cache"]
		if "codec_cache" in step_result and step_result["codec_cache"] is not None:
			context.codec_cache = step_result["codec_cache"]

	def reset_slots(self, stream_ids: List[int], eos_flags: List[bool]) -> None:
		"""Release contexts for streams that signalled end-of-stream."""
		if len(stream_ids) != len(eos_flags):
			raise ValueError("stream_ids and eos_flags must have the same length")
		for stream_id, eos_flag in zip(stream_ids, eos_flags):
			if eos_flag and stream_id in self.streamidx2slotidx:
				self.reset_slot(self.streamidx2slotidx[stream_id])

	def get_context(self, stream_ids: List[int]) -> Tuple[StreamingRealtimeContext, Dict[int, int]]:
		"""Return the cached context associated with the provided stream ids."""
		if len(stream_ids) == 0:
			return self._create_context(), {}
		if len(stream_ids) != 1:
			raise NotImplementedError("get_context currently supports batch_size == 1")

		stream_id = stream_ids[0]
		slot_idx = self._ensure_slot(stream_id)

		if self.slot_contexts[slot_idx] is None:
			self.slot_contexts[slot_idx] = self._create_context()

		return self.slot_contexts[slot_idx], {slot_idx: 0}
