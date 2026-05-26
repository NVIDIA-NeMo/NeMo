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

import json
import os
import queue as _queue_module
import random
import threading
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

import concurrent.futures
import re


def _parse_tool_name(call_text: str) -> str | None:
	"""Extract tool name from '<TOOLCALL>[{"name": "tool_name", ...}]</TOOLCALL>'."""
	m = re.search(r'"name"\s*:\s*"([^"]+)"', call_text)
	return m.group(1) if m else None


def _response_is_401(tool_response: str | None) -> bool:
	"""Return True if the tool response signals a 401 / auth error."""
	if not tool_response:
		return False
	try:
		data = json.loads(tool_response)
		err = str(data.get("error", ""))
		return "401" in err or "unauthorized" in err.lower() or "authentication" in err.lower()
	except Exception:
		return "401" in tool_response


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

		# UI ASR source: 'rnnt' (default) or 'asr_head'
		self.ui_asr_source = getattr(s2s_cfg, "ui_asr_source", "rnnt")

		# Built-in tool executor for two-phase async FC (guarded by config flag)
		self.tool_registry: dict[str, callable] = {}
		if getattr(s2s_cfg, "enable_builtin_tools", False):
			self._register_builtin_tools()

		# ------------------------------------------------------------------
		# Input audio padding (silence appended after real audio)
		# ------------------------------------------------------------------
		self.pad_audio_to_sec: float | None = cfg.get("pad_audio_to_sec", None)
		self.pad_silence_ratio: float | None = cfg.get("pad_silence_ratio", None)
		self.pad_audio_by_sec: float | None = cfg.get("pad_audio_by_sec", None)
		if sum(x is not None for x in [self.pad_audio_to_sec, self.pad_silence_ratio, self.pad_audio_by_sec]) > 1:
			raise ValueError("Set at most one of: pad_audio_to_sec, pad_silence_ratio, pad_audio_by_sec")

		# Trailing PAD count scales with phrase length so longer ACK / on-hold
		# messages get proportionally more codec-drain time. Short phrases keep
		# the floor. Shared across ACK, FC reminder, per-tool on-hold, and generic
		# on-hold loops. Rate ~0.5 pad/char (1 pad per 2 chars); floor 17 matches
		# the previous fixed value so short phrases see no change.
		_FC_TRAILING_PAD_MIN = 17
		_FC_TRAILING_PAD_PER_CHAR = 0.5

		# ------------------------------------------------------------------
		# FC async acknowledgement message.
		# If set, the TTS generates this phrase during FC instead of silence,
		# warming up the codec cache with real speech for smooth audio after FC.
		# One message is randomly selected per FC call from the list below.
		# ------------------------------------------------------------------
		# Short, polite acknowledgement phrases — fill the TTS audio during the
		# LLM's TOOLCALL emission (SOTC → EOTC). Designed to be warm and positive
		# while staying noticeably shorter than per-tool on-hold variants
		# (~20 vs ~50 chars). Lead-ins (Happy / My pleasure / Great / Lovely /
		# Glad / Wonderful / Pleased) are intentionally disjoint from on-hold
		# lead-ins (Let me / I'll / Looking / Getting / Checking / Pulling),
		# so the two phases sound like distinct conversational moves rather
		# than the same sentence repeated.
		_FC_ACK_MESSAGES = [
			"Happy to help with that.",
			"My pleasure to help.",
			"Great question, looking now.",
			"Lovely, on it now.",
			"Glad you asked.",
			"Happy to take a look.",
			"Wonderful, looking now.",
			"Happy to look into that.",
			"Glad to help with that.",
			"Happy to look that up.",
			"Pleased to help out.",
			"Great, happy to help.",
			"Lovely question.",
			"Glad to check.",
			"Happy to check that.",
			"Wonderful question.",
			"Pleased to look into that.",
			"Happy to assist.",
			"Great, on it now.",
			"Lovely, looking now.",
		]
		self._fc_ack_tokens: list | None = None
		self._fc_ack_token_list: list | None = None
		fc_ack_text = getattr(s2s_cfg, "fc_random_ack_enabled", None)
		if fc_ack_text:
			# Wrap each ack message with BOS/EOS tokens identical to what the LLM emits
			# for natural turns. Without these, TTS thinks the utterance never ended →
			# autoregressive state stays "mid-speech" → contaminates the verbal response
			# that follows the FC cycle. Insert 17 PAD frames before EOS so the codec
			# has time to finish rendering last content tokens before silence reset.
			_stt = self.s2s_model.model.stt_model
			_bos_id = getattr(_stt, "text_bos_id", None)
			_eos_id = getattr(_stt, "text_eos_id", None)
			_pad_id = getattr(_stt, "text_pad_id", None)
			try:
				_bos_text = self.s2s_model.tokenizer.ids_to_text([_bos_id]) if _bos_id is not None else None
				_eos_text = self.s2s_model.tokenizer.ids_to_text([_eos_id]) if _eos_id is not None else None
			except Exception:
				_bos_text = _eos_text = "(decode failed)"
			logging.info(
				"FC acknowledgement: text_bos_id=%s (text=%r), text_eos_id=%s (text=%r)",
				_bos_id, _bos_text, _eos_id, _eos_text,
			)
			self._fc_ack_token_list = []
			for msg in _FC_ACK_MESSAGES:
				_ack_ids = list(self.s2s_model.tokenizer.text_to_ids(msg))
				_trailing_pad_count = max(_FC_TRAILING_PAD_MIN, math.ceil(_FC_TRAILING_PAD_PER_CHAR * len(msg)))
				if _bos_id is not None:
					_ack_ids = [_bos_id] + _ack_ids
				if _pad_id is not None and _trailing_pad_count > 0:
					_ack_ids = _ack_ids + [_pad_id] * _trailing_pad_count
				if _eos_id is not None:
					_ack_ids = _ack_ids + [_eos_id]
				self._fc_ack_token_list.append(_ack_ids)
			self._fc_ack_tokens = self._fc_ack_token_list[0]
			try:
				_sample_text = self.s2s_model.tokenizer.ids_to_text(self._fc_ack_token_list[0])
			except Exception:
				_sample_text = "(decode failed)"
			logging.info(
				"FC acknowledgement: random selection enabled, %d messages loaded "
				"(each wrapped with <s>/</s>, %d trailing PADs); sample[0]=%d tokens, decoded=%r",
				len(self._fc_ack_token_list), _trailing_pad_count,
				len(self._fc_ack_token_list[0]), _sample_text,
			)

		# ------------------------------------------------------------------
		# FC tool-call reminder messages: played after the initial ACK audio
		# ends but while the tool API is still executing (e.g. slow weather
		# API). Enabled via s2s_cfg.tool_reminder_enabled (env S2S_TOOL_REMINDER_ENABLED).
		# One message is randomly chosen per FC call and synthesized via
		# _run_tts_reminder() in the background thread between Phase 1 and
		# execute_tool_fn(), filling the silence gap.
		# ------------------------------------------------------------------
		_FC_REMINDER_MESSAGES = [
			"Still working on that.",
			"Just a moment longer.",
			"Almost there, one second.",
			"Bear with me just a bit.",
			"Still looking that up for you.",
			"Give me just another second.",
			"Working on it, almost done.",
			"Still checking, won't be long.",
			"Hang tight, almost there.",
			"Nearly done, one more moment.",
		]
		self._fc_reminder_token_list: list | None = None
		fc_reminder_enabled = getattr(s2s_cfg, "tool_reminder_enabled", None)
		if fc_reminder_enabled:
			_stt_r = self.s2s_model.model.stt_model
			_bos_id_r = getattr(_stt_r, "text_bos_id", None)
			_eos_id_r = getattr(_stt_r, "text_eos_id", None)
			_pad_id_r = getattr(_stt_r, "text_pad_id", None)
			self._fc_reminder_token_list = []
			for msg in _FC_REMINDER_MESSAGES:
				_rem_ids = list(self.s2s_model.tokenizer.text_to_ids(msg))
				_trailing_pad_r = max(_FC_TRAILING_PAD_MIN, math.ceil(_FC_TRAILING_PAD_PER_CHAR * len(msg)))
				if _bos_id_r is not None:
					_rem_ids = [_bos_id_r] + _rem_ids
				if _pad_id_r is not None and _trailing_pad_r > 0:
					_rem_ids = _rem_ids + [_pad_id_r] * _trailing_pad_r
				if _eos_id_r is not None:
					_rem_ids = _rem_ids + [_eos_id_r]
				self._fc_reminder_token_list.append(_rem_ids)
			try:
				_sample_r = self.s2s_model.tokenizer.ids_to_text(self._fc_reminder_token_list[0])
			except Exception:
				_sample_r = "(decode failed)"
			logging.info(
				"FC reminder: enabled, %d messages loaded; sample[0]=%r",
				len(self._fc_reminder_token_list), _sample_r,
			)

		# ------------------------------------------------------------------
		# Per-tool on-hold messages (JSON: {tool_name: phrase}).
		# Played at EOTC as Layer 1 reminder while the external API executes.
		# ------------------------------------------------------------------
		self._fc_on_hold_token_map: dict | None = None
		_on_hold_path = getattr(s2s_cfg, "fc_on_hold_messages_path", None) or ""
		if _on_hold_path and os.path.isfile(_on_hold_path):
			with open(_on_hold_path) as _f:
				_on_hold_data = json.load(_f)
			self._fc_on_hold_token_map = {}
			_stt_oh = self.s2s_model.model.stt_model
			_bos_oh = getattr(_stt_oh, "text_bos_id", None)
			_eos_oh = getattr(_stt_oh, "text_eos_id", None)
			_pad_oh = getattr(_stt_oh, "text_pad_id", None)
			for _tool_key, _oh_value in _on_hold_data.items():
				# Accept either a single string (back-compat) or a list of phrases.
				# Stored uniformly as list-of-token-lists so the dispatcher can random.choice.
				_msgs = [_oh_value] if isinstance(_oh_value, str) else list(_oh_value)
				_tokenized_list = []
				for _oh_msg in _msgs:
					_ids = list(self.s2s_model.tokenizer.text_to_ids(_oh_msg))
					_trailing_pad_count = max(_FC_TRAILING_PAD_MIN, math.ceil(_FC_TRAILING_PAD_PER_CHAR * len(_oh_msg)))
					if _bos_oh is not None:
						_ids = [_bos_oh] + _ids
					if _pad_oh is not None:
						_ids += [_pad_oh] * _trailing_pad_count
					if _eos_oh is not None:
						_ids += [_eos_oh]
					_tokenized_list.append(_ids)
				self._fc_on_hold_token_map[_tool_key] = _tokenized_list
			logging.info("FC on-hold: %d per-tool messages loaded from %s", len(self._fc_on_hold_token_map), _on_hold_path)

		# ------------------------------------------------------------------
		# Generic on-hold messages (JSON: list of phrases).
		# Up to 2 played (with 1s gap) if API is still blocked after Layer 1.
		# ------------------------------------------------------------------
		self._fc_generic_on_hold_token_list: list | None = None
		_generic_path = getattr(s2s_cfg, "fc_generic_on_hold_messages_path", None) or ""
		if _generic_path and os.path.isfile(_generic_path):
			with open(_generic_path) as _f:
				_generic_data = json.load(_f)
			self._fc_generic_on_hold_token_list = []
			_stt_g = self.s2s_model.model.stt_model
			_bos_g = getattr(_stt_g, "text_bos_id", None)
			_eos_g = getattr(_stt_g, "text_eos_id", None)
			_pad_g = getattr(_stt_g, "text_pad_id", None)
			for _g_msg in _generic_data:
				_ids = list(self.s2s_model.tokenizer.text_to_ids(_g_msg))
				_trailing_pad_count = max(_FC_TRAILING_PAD_MIN, math.ceil(_FC_TRAILING_PAD_PER_CHAR * len(_g_msg)))
				if _bos_g is not None:
					_ids = [_bos_g] + _ids
				if _pad_g is not None:
					_ids += [_pad_g] * _trailing_pad_count
				if _eos_g is not None:
					_ids += [_eos_g]
				self._fc_generic_on_hold_token_list.append(_ids)
			logging.info("FC generic on-hold: %d messages loaded from %s", len(self._fc_generic_on_hold_token_list), _generic_path)

		# Pre-tokenize fixed phrases used for timeout and 401 errors.
		_stt_sp = self.s2s_model.model.stt_model
		_bos_sp = getattr(_stt_sp, "text_bos_id", None)
		_eos_sp = getattr(_stt_sp, "text_eos_id", None)
		_pad_sp = getattr(_stt_sp, "text_pad_id", None)

		def _tok_special(msg: str) -> list:
			_ids = list(self.s2s_model.tokenizer.text_to_ids(msg))
			if _bos_sp is not None:
				_ids = [_bos_sp] + _ids
			if _pad_sp is not None:
				_ids += [_pad_sp] * 17
			if _eos_sp is not None:
				_ids += [_eos_sp]
			return _ids

		self._fc_timeout_fallback_tokens: list = _tok_special(
			"Please let me know how can I help?"
		)
		self._fc_401_error_tokens: list = _tok_special(
			"I apologize, I am having difficulty in fulfilling your request as of now, please try again later."
		)
		self._fc_401_redirect_tokens: list = _tok_special(
			"How can I help you?"
		)
		self._fc_tool_timeout_sec: float = float(getattr(s2s_cfg, "fc_tool_timeout_sec", 15.0))

		# ------------------------------------------------------------------
		# Non-blocking FC async state (per stream).
		# When FC async is active, execute() returns immediately (silence)
		# while the LLM loop runs in a background thread.  Subsequent audio
		# frames are pushed to a live queue so the perception encoder and
		# RNNT keep running every 80 ms during the function call.
		# ------------------------------------------------------------------
		self._fc_async_bg: dict = {}  # stream_id → bg-state dict

		super().__init__()

	# ------------------------------------------------------------------
	# Built-in tool executor (used in async FC when enable_builtin_tools=True)
	# ------------------------------------------------------------------
	def _register_builtin_tools(self):
		"""Register built-in tool functions for async FC two-phase mode."""
		self.tool_registry["calculate_bmi"] = self._tool_calculate_bmi
		self.tool_registry["get_weather"] = self._tool_get_weather
		self.tool_registry["check_gpu_usage"] = self._tool_check_gpu_usage
		self.tool_registry["get_stock_price"] = self._tool_get_stock_price
		self.tool_registry["get_top_paper"] = self._tool_get_top_paper
		self.tool_registry["get_top_news"] = self._tool_get_top_news
		self.tool_registry["generate_random_number"] = self._tool_generate_random_number
		self.tool_registry["find_nearby_restaurants"] = self._tool_find_nearby_restaurants
		logging.info("Registered %d built-in tool(s): %s", len(self.tool_registry), list(self.tool_registry.keys()))

	@staticmethod
	def _tool_calculate_bmi(args: dict) -> str:
		weight = float(args["weight"])
		height = float(args["height"])
		if height <= 0:
			return json.dumps({"error": "Height must be positive"})
		bmi = weight / (height ** 2)
		if bmi < 18.5:
			category = "underweight"
		elif bmi < 25:
			category = "normal weight"
		elif bmi < 30:
			category = "overweight"
		else:
			category = "obese"
		return json.dumps({"bmi": round(bmi, 1), "category": category})

	@staticmethod
	def _tool_get_weather(args: dict) -> str:
		import urllib.request
		import urllib.parse
		CITY_COORDS = {
			# US
			"san francisco": (37.77, -122.42),
			"santa clara": (37.35, -121.95),
			"new york": (40.71, -74.01),
			"los angeles": (34.05, -118.24),
			"chicago": (41.88, -87.63),
			"seattle": (47.61, -122.33),
			"austin": (30.27, -97.74),
			"denver": (39.74, -104.99),
			"miami": (25.76, -80.19),
			"boston": (42.36, -71.06),
			"washington": (38.91, -77.04),
			"washington dc": (38.91, -77.04),
			"dallas": (32.78, -96.80),
			"houston": (29.76, -95.37),
			"phoenix": (33.45, -112.07),
			"atlanta": (33.75, -84.39),
			"philadelphia": (39.95, -75.17),
			"san diego": (32.72, -117.16),
			"las vegas": (36.17, -115.14),
			"portland": (45.52, -122.68),
			"minneapolis": (44.98, -93.27),
			"detroit": (42.33, -83.05),
			"san jose": (37.34, -121.89),
			# California — Bay Area / Silicon Valley
			"oakland": (37.80, -122.27),
			"berkeley": (37.87, -122.27),
			"palo alto": (37.44, -122.14),
			"mountain view": (37.39, -122.08),
			"sunnyvale": (37.37, -122.04),
			"cupertino": (37.32, -122.03),
			"fremont": (37.55, -121.99),
			"hayward": (37.67, -122.08),
			"concord": (37.98, -122.03),
			"san mateo": (37.56, -122.32),
			"redwood city": (37.49, -122.24),
			"menlo park": (37.45, -122.18),
			# California — Southern / Central
			"long beach": (33.77, -118.19),
			"anaheim": (33.84, -117.91),
			"santa monica": (34.02, -118.49),
			"pasadena": (34.15, -118.14),
			"burbank": (34.18, -118.31),
			"glendale": (34.14, -118.25),
			"hollywood": (34.10, -118.33),
			"beverly hills": (34.07, -118.40),
			"irvine": (33.68, -117.83),
			"riverside": (33.95, -117.40),
			"san bernardino": (34.11, -117.29),
			"oxnard": (34.20, -119.18),
			"ventura": (34.27, -119.23),
			"santa barbara": (34.42, -119.70),
			"bakersfield": (35.37, -119.02),
			"fresno": (36.74, -119.78),
			"stockton": (37.96, -121.29),
			"modesto": (37.64, -120.99),
			"monterey": (36.60, -121.89),
			"santa cruz": (36.97, -122.03),
			"napa": (38.30, -122.29),
			"santa rosa": (38.44, -122.71),
			"lake tahoe": (39.10, -120.04),
			"honolulu": (21.31, -157.86),
			"anchorage": (61.22, -149.90),
			"sacramento": (38.58, -121.49),
			"salt lake city": (40.76, -111.89),
			"nashville": (36.16, -86.78),
			"new orleans": (29.95, -90.07),
			"charlotte": (35.23, -80.84),
			"pittsburgh": (40.44, -79.99),
			"saint louis": (38.63, -90.20),
			"st louis": (38.63, -90.20),
			"orlando": (28.54, -81.38),
			"tampa": (27.95, -82.46),
			"baltimore": (39.29, -76.61),
			"indianapolis": (39.77, -86.16),
			"kansas city": (39.10, -94.58),
			"toronto": (43.65, -79.38),
			"montreal": (45.50, -73.57),
			"vancouver": (49.28, -123.12),
			"mexico city": (19.43, -99.13),
			# Europe
			"london": (51.51, -0.13),
			"paris": (48.86, 2.35),
			"berlin": (52.52, 13.40),
			"madrid": (40.42, -3.70),
			"rome": (41.90, 12.50),
			"amsterdam": (52.37, 4.90),
			"barcelona": (41.39, 2.17),
			"vienna": (48.21, 16.37),
			"moscow": (55.76, 37.62),
			"istanbul": (41.01, 28.98),
			"stockholm": (59.33, 18.07),
			"dublin": (53.35, -6.26),
			"zurich": (47.38, 8.54),
			"brussels": (50.85, 4.35),
			"munich": (48.14, 11.58),
			"lisbon": (38.72, -9.14),
			"copenhagen": (55.68, 12.57),
			"warsaw": (52.23, 21.01),
			"prague": (50.08, 14.44),
			"athens": (37.98, 23.73),
			"frankfurt": (50.11, 8.68),
			"hamburg": (53.55, 9.99),
			"milan": (45.46, 9.19),
			"geneva": (46.20, 6.14),
			"helsinki": (60.17, 24.94),
			"oslo": (59.91, 10.75),
			"budapest": (47.50, 19.04),
			"edinburgh": (55.95, -3.19),
			"manchester": (53.48, -2.24),
			"birmingham": (52.49, -1.90),
			"marseille": (43.30, 5.37),
			"lyon": (45.76, 4.84),
			"naples": (40.85, 14.27),
			"florence": (43.77, 11.26),
			"venice": (45.44, 12.32),
			"reykjavik": (64.15, -21.94),
			# Asia / ME / Oceania
			"tokyo": (35.68, 139.69),
			"beijing": (39.90, 116.41),
			"shanghai": (31.23, 121.47),
			"hong kong": (22.32, 114.17),
			"singapore": (1.35, 103.82),
			"seoul": (37.57, 126.98),
			"bangkok": (13.76, 100.50),
			"mumbai": (19.08, 72.88),
			"delhi": (28.61, 77.21),
			"new delhi": (28.61, 77.21),
			"dubai": (25.20, 55.27),
			"jakarta": (-6.21, 106.85),
			"manila": (14.60, 120.98),
			"taipei": (25.03, 121.57),
			"kuala lumpur": (3.14, 101.69),
			"ho chi minh city": (10.82, 106.63),
			"saigon": (10.82, 106.63),
			"hanoi": (21.03, 105.85),
			"osaka": (34.69, 135.50),
			"kyoto": (35.01, 135.77),
			"shenzhen": (22.54, 114.06),
			"guangzhou": (23.13, 113.26),
			"riyadh": (24.71, 46.68),
			"tel aviv": (32.08, 34.78),
			"karachi": (24.86, 67.01),
			"lahore": (31.55, 74.34),
			"chennai": (13.08, 80.27),
			"bangalore": (12.97, 77.59),
			"bengaluru": (12.97, 77.59),
			"kolkata": (22.57, 88.36),
			"hyderabad": (17.39, 78.49),
			"ahmedabad": (23.02, 72.57),
			"pune": (18.52, 73.86),
			"jaipur": (26.92, 75.79),
			"lucknow": (26.85, 80.95),
			"surat": (21.17, 72.83),
			"kanpur": (26.45, 80.33),
			"nagpur": (21.15, 79.09),
			"indore": (22.72, 75.86),
			"bhopal": (23.26, 77.41),
			"patna": (25.59, 85.14),
			"coimbatore": (11.02, 76.96),
			"agra": (27.18, 78.01),
			"visakhapatnam": (17.69, 83.22),
			"vizag": (17.69, 83.22),
			"kochi": (9.93, 76.27),
			"goa": (15.30, 74.12),
			"thiruvananthapuram": (8.52, 76.94),
			"trivandrum": (8.52, 76.94),
			"chandigarh": (30.73, 76.78),
			"varanasi": (25.32, 82.99),
			"doha": (25.29, 51.53),
			"abu dhabi": (24.45, 54.38),
			"kuwait city": (29.38, 47.99),
			"sapporo": (43.07, 141.35),
			"yokohama": (35.44, 139.64),
			"busan": (35.18, 129.08),
			"chengdu": (30.57, 104.07),
			"hangzhou": (30.27, 120.16),
			"wuhan": (30.59, 114.31),
			"xi'an": (34.27, 108.95),
			"sydney": (-33.87, 151.21),
			"melbourne": (-37.81, 144.96),
			"auckland": (-36.85, 174.76),
		}
		WMO_CODES = {
			0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
			45: "foggy", 48: "depositing rime fog",
			51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
			61: "slight rain", 63: "moderate rain", 65: "heavy rain",
			71: "slight snow", 73: "moderate snow", 75: "heavy snow",
			80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
			95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
		}
		city_raw = args.get("city", "").strip()
		city = city_raw.lower()
		# Try exact match first, then fuzzy: strip state/country suffixes and partial match
		coords = CITY_COORDS.get(city)
		if coords is None:
			# Strip common suffixes like ", CA, USA" or ", California"
			city_base = city.split(",")[0].strip()
			coords = CITY_COORDS.get(city_base)
		if coords is None:
			# Partial match: check if any known city is a substring or vice versa
			for known_city, known_coords in CITY_COORDS.items():
				if known_city in city or city_base in known_city:
					coords = known_coords
					break
		if coords is None:
			return json.dumps({"error": f"Unknown city '{city_raw}'. Try a major city in America, Europe, or Asia."})
		lat, lon = coords
		url = (
			f"https://api.open-meteo.com/v1/forecast?"
			f"latitude={lat}&longitude={lon}"
			f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
			f"&temperature_unit=fahrenheit"
		)
		try:
			with urllib.request.urlopen(url, timeout=5) as resp:
				data = json.loads(resp.read().decode())
			current = data["current"]
			weather_desc = WMO_CODES.get(current["weather_code"], "unknown")
			return json.dumps({
				"city": args.get("city", city),
				"temperature_f": current["temperature_2m"],
				"humidity_percent": current["relative_humidity_2m"],
				"wind_speed_kmh": current["wind_speed_10m"],
				"conditions": weather_desc,
			})
		except Exception as e:
			return json.dumps({"error": f"Weather API failed: {e}"})

	@staticmethod
	def _tool_check_gpu_usage(args: dict) -> str:
		import subprocess
		try:
			result = subprocess.run(
				["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
				 "--format=csv,noheader,nounits"],
				capture_output=True, text=True, timeout=5,
			)
			if result.returncode != 0:
				return json.dumps({"error": result.stderr.strip()})
			gpus = []
			for line in result.stdout.strip().split("\n"):
				parts = [p.strip() for p in line.split(",")]
				if len(parts) >= 6:
					gpus.append({
						"gpu_index": int(parts[0]),
						"name": parts[1],
						"utilization_percent": int(parts[2]),
						"memory_used_mb": int(parts[3]),
						"memory_total_mb": int(parts[4]),
						"temperature_c": int(parts[5]),
					})
			return json.dumps({"gpus": gpus})
		except Exception as e:
			return json.dumps({"error": f"nvidia-smi failed: {e}"})

	@staticmethod
	def _tool_get_stock_price(args: dict) -> str:
		import urllib.request
		symbol = args.get("symbol", "").strip().upper()
		if not symbol:
			return json.dumps({"error": "No stock symbol provided"})
		url = (
			f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
			f"?range=1d&interval=1d"
		)
		req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
		try:
			with urllib.request.urlopen(req, timeout=5) as resp:
				data = json.loads(resp.read().decode())
			result = data["chart"]["result"]
			if not result:
				return json.dumps({"error": f"No data found for symbol '{symbol}'"})
			meta = result[0]["meta"]
			return json.dumps({
				"symbol": meta.get("symbol", symbol),
				"currency": meta.get("currency", "USD"),
				"current_price": meta.get("regularMarketPrice"),
				"previous_close": meta.get("chartPreviousClose"),
				"day_high": meta.get("regularMarketDayHigh"),
				"day_low": meta.get("regularMarketDayLow"),
				"market_state": meta.get("marketState", "unknown"),
			})
		except Exception as e:
			return json.dumps({"error": f"Stock API failed: {e}"})

	@staticmethod
	def _tool_get_top_paper(args: dict) -> str:
		import urllib.request
		import urllib.parse
		from datetime import date as _date, timedelta
		date_str = args.get("date", "").strip()
		if not date_str:
			date_str = _date.today().strftime("%Y-%m-%d")
		url = f"https://huggingface.co/api/daily_papers?date={urllib.parse.quote(date_str)}&limit=1"
		req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
		try:
			with urllib.request.urlopen(req, timeout=10) as resp:
				data = json.loads(resp.read().decode())
			if not data:
				yesterday = (_date.fromisoformat(date_str) - timedelta(days=1)).strftime("%Y-%m-%d")
				url2 = f"https://huggingface.co/api/daily_papers?date={yesterday}&limit=1"
				req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
				with urllib.request.urlopen(req2, timeout=10) as resp2:
					data = json.loads(resp2.read().decode())
				if not data:
					return json.dumps({"error": f"No papers found for {date_str} or {yesterday}"})
				date_str = yesterday
			paper = data[0]
			p = paper.get("paper", {})
			return json.dumps({
				"date": date_str,
				"title": p.get("title", "Unknown"),
				"authors": ", ".join(a.get("name", "") for a in (p.get("authors", []) or [])[:3]),
				"summary": (p.get("summary", "") or "")[:300],
				"upvotes": paper.get("numUpvotes", 0),
			})
		except Exception as e:
			return json.dumps({"error": f"HuggingFace papers API failed: {e}"})

	@staticmethod
	def _tool_get_top_news(args: dict) -> str:
		import urllib.request
		import xml.etree.ElementTree as ET
		topic = args.get("topic", "").strip().lower()
		topic_map = {
			"business": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
			"technology": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
			"science": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB",
			"health": "CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ",
			"sports": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB",
			"entertainment": "CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB",
		}
		n = min(int(args.get("n", 1)), 1)
		if topic and topic in topic_map:
			url = f"https://news.google.com/rss/topics/{topic_map[topic]}?hl=en-US&gl=US&ceid=US:en"
		else:
			url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
		req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
		try:
			with urllib.request.urlopen(req, timeout=10) as resp:
				tree = ET.parse(resp)
			items = tree.findall(".//item")[:n]
			if not items:
				return json.dumps({"error": "No news articles found"})
			articles = []
			for item in items:
				title = item.find("title")
				pub = item.find("pubDate")
				source = item.find("source")
				articles.append({
					"title": title.text if title is not None else "",
					"source": source.text if source is not None else "",
					"published": pub.text if pub is not None else "",
				})
			return json.dumps({"topic": topic or "top stories", "articles": articles})
		except Exception as e:
			return json.dumps({"error": f"Google News fetch failed: {e}"})

	@staticmethod
	def _tool_generate_random_number(args: dict) -> str:
		import random
		min_val = int(args.get("min", 1))
		max_val = int(args.get("max", 100))
		if min_val > max_val:
			return json.dumps({"error": "min must be less than or equal to max"})
		result = random.randint(min_val, max_val)
		return json.dumps({"result": result, "min": min_val, "max": max_val})

	@staticmethod
	def _tool_find_nearby_restaurants(args: dict) -> str:
		"""Find nearby restaurants using OpenStreetMap Nominatim + Overpass API (no API key)."""
		import urllib.request, urllib.parse
		city = args.get("city", "").strip()
		cuisine = args.get("cuisine", "").strip()
		limit = min(int(args.get("limit", 1)), 1)
		if not city:
			return json.dumps({"error": "city is required"})
		# Step 1: geocode city → lat/lng via Nominatim
		nominatim_url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
			"q": city, "format": "json", "limit": 1,
		})
		req = urllib.request.Request(nominatim_url, headers={"User-Agent": "NvidiaVoiceChat/1.0"})
		try:
			with urllib.request.urlopen(req, timeout=8) as resp:
				geo = json.loads(resp.read().decode())
			if not geo:
				return json.dumps({"error": f"Could not geocode city: {city}"})
			lat, lng = float(geo[0]["lat"]), float(geo[0]["lon"])
		except Exception as e:
			return json.dumps({"error": f"Geocoding failed: {e}"})
		# Step 2: query Overpass for restaurants within 3km
		cuisine_filter = f'["cuisine"~"{cuisine}",i]' if cuisine else ""
		overpass_query = f"""
[out:json][timeout:10];
(
  node["amenity"="restaurant"]{cuisine_filter}(around:3000,{lat},{lng});
  way["amenity"="restaurant"]{cuisine_filter}(around:3000,{lat},{lng});
);
out center {limit};
""".strip()
		overpass_url = "https://overpass-api.de/api/interpreter"
		req2 = urllib.request.Request(
			overpass_url,
			data=overpass_query.encode(),
			headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "NvidiaVoiceChat/1.0"},
		)
		try:
			with urllib.request.urlopen(req2, timeout=15) as resp:
				data = json.loads(resp.read().decode())
			restaurants = []
			for el in data.get("elements", []):
				tags = el.get("tags", {})
				name = tags.get("name", "").strip()
				if not name:
					continue
				if el["type"] == "node":
					rlat, rlng = el.get("lat"), el.get("lon")
				else:
					center = el.get("center", {})
					rlat, rlng = center.get("lat"), center.get("lon")
				addr_parts = [tags.get("addr:housenumber", ""), tags.get("addr:street", ""),
							  tags.get("addr:city", "")]
				address = " ".join(p for p in addr_parts if p).strip() or None
				restaurants.append({
					"name": name,
					"address": address,
					"cuisine": tags.get("cuisine"),
					"lat": rlat,
					"lng": rlng,
				})
			return json.dumps({"restaurants": restaurants, "city": city})
		except Exception as e:
			return json.dumps({"error": f"Overpass query failed: {e}"})

	def _execute_tool_call(self, call_text: str) -> str | None:
		"""Parse a tool call string and execute the function if registered.

		Returns the tool response string, or None if execution fails.
		"""
		if not self.tool_registry:
			return None

		# Strip <TOOLCALL>...</TOOLCALL> wrapper if present
		raw = call_text.strip()
		if raw.startswith("<TOOLCALL>"):
			raw = raw[len("<TOOLCALL>"):]
		if raw.endswith("</TOOLCALL>"):
			raw = raw[:-len("</TOOLCALL>")]
		raw = raw.strip()

		try:
			parsed = json.loads(raw)
		except json.JSONDecodeError:
			logging.warning("[Tool Executor] Could not parse call as JSON: %s", raw[:200])
			return None

		# Model may emit a list of calls or a single dict
		if isinstance(parsed, list):
			call = parsed[0] if parsed else {}
		elif isinstance(parsed, dict):
			call = parsed
		else:
			logging.warning("[Tool Executor] Unexpected call format: %s", type(parsed))
			return None

		if isinstance(call, dict):
			func_name = call.get("name")
			arguments = call.get("arguments", call.get("parameters", {}))
		else:
			logging.warning("[Tool Executor] Unexpected call item format: %s", type(call))
			return None

		# Normalize hyphens → underscores: model sometimes emits "get-weather" instead of "get_weather"
		func_name_normalized = func_name.replace("-", "_")
		handler = self.tool_registry.get(func_name_normalized)
		if handler is None:
			logging.warning("[Tool Executor] Unknown function '%s'. Registered: %s", func_name, list(self.tool_registry.keys()))
			return None

		try:
			result = handler(arguments)
			logging.info("[Tool Executor] %s(%s) -> %s", func_name, arguments, result)
			return result
		except Exception as e:
			logging.error("[Tool Executor] %s raised %s: %s", func_name, type(e).__name__, e)
			return json.dumps({"error": str(e)})

	# ------------------------------------------------------------------
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

		# Initialize FC state on first audio frame if not already done via prefill
		if bos_flags[0] and context.fc_state is None:
			has_fc = self.s2s_model.model.stt_model.function_head is not None
			if has_fc:
				context.fc_state = {
					"active": False,
					"call_tokens": [],
					"completed_calls": [],
					"forced_function_tokens": [],
					"injecting_response": False,
				}
			opts = frames[0].options
			if opts is not None and hasattr(opts, "tool_response_text") and opts.tool_response_text:
				raw = opts.tool_response_text
				try:
					parsed = json.loads(raw)
					if isinstance(parsed, list):
						context.tool_response_queue = list(parsed)
						context.tool_response_text = context.tool_response_queue.pop(0) if context.tool_response_queue else None
					else:
						context.tool_response_text = raw
						context.tool_response_queue = []
				except (json.JSONDecodeError, TypeError):
					context.tool_response_text = raw
					context.tool_response_queue = []

		# Initialize per-stream FC timing tracker on BOS
		if bos_flags[0]:
			opts = frames[0].options
			audio_dur = None
			user_segs = None
			if opts is not None:
				if hasattr(opts, "audio_duration_sec"):
					audio_dur = opts.audio_duration_sec
				if hasattr(opts, "user_segments"):
					user_segs = opts.user_segments
			context.fc_timing = {
				"stream_start_wall": time.time(),
				"frame_step_times": [],
				"agent_response_start_wall": None,
				"agent_response_start_frame": None,
				"events": [],
				"audio_duration_sec": audio_dur,
				"user_segments": user_segs,
			}

		# Pre-load full audio for real-time async FC (wall-clock gated).
		# Only done once at BOS when fc_async is enabled and a filepath is available.
		if bos_flags[0] and self.s2s_model._fc_async_enabled:
			opts = frames[0].options
			afp = getattr(opts, "audio_filepath", None) if opts is not None else None
			if afp and os.path.isfile(afp):
				import numpy as np
				audio_np, _ = librosa.load(afp, sr=self.input_sample_rate)
				pad_to = self.pad_audio_to_sec
				pad_ratio = self.pad_silence_ratio
				pad_by = self.pad_audio_by_sec
				orig_samples = len(audio_np)
				orig_dur = orig_samples / self.input_sample_rate
				if pad_to is not None and pad_to > orig_dur:
					audio_np = np.pad(audio_np, (0, int(pad_to * self.input_sample_rate) - orig_samples))
				elif pad_ratio is not None:
					audio_np = np.pad(audio_np, (0, int(orig_dur * pad_ratio * self.input_sample_rate)))
				elif pad_by is not None:
					audio_np = np.pad(audio_np, (0, int(pad_by * self.input_sample_rate)))
				frame_size_samples = int(0.08 * self.input_sample_rate)
				total_samples = len(audio_np)
				total_frames_rt = int(math.ceil(total_samples / frame_size_samples))
				padded_samples = total_frames_rt * frame_size_samples
				if padded_samples > total_samples:
					audio_np = np.pad(audio_np, (0, padded_samples - total_samples))
				context.rt_audio_signal = torch.tensor(
					audio_np, dtype=self.s2s_model.dtype, device=self.s2s_model.device,
				).unsqueeze(0)
				context.rt_audio_total_frames = total_frames_rt
				buf_size_samples = int(self.buffer_size_in_secs * self.input_sample_rate)
				context.rt_audio_buffer_size_samples = buf_size_samples
				logging.info(
					"[FC Async RT] Pre-loaded audio for stream %d: %d frames, "
					"%.2fs (buffer=%d samples)",
					stream_ids[0], total_frames_rt, total_samples / self.input_sample_rate,
					buf_size_samples,
				)

		# ---------------------------------------------------------------
		# Non-blocking FC async: handle frames arriving while the LLM
		# is running FC async in a background thread.
		# ---------------------------------------------------------------
		stream_id = stream_ids[0]
		if stream_id in self._fc_async_bg:
			bg = self._fc_async_bg[stream_id]

			# Extract and push the newest 80ms audio frame to the live queue
			# so the perception worker and RNNT can process it.
			raw_buf = buffers[0]
			if raw_buf is not None and raw_buf.numel() > 0:
				if raw_buf.dim() == 1:
					raw_buf = raw_buf.unsqueeze(0)
				frame_samples = int(0.08 * self.input_sample_rate)
				# Take the rightmost 80ms (newest audio) — the buffer is a
				# rolling window so the last frame_samples samples are newest.
				live_frame = raw_buf[:, -frame_samples:].clone().cpu()
				bg["live_audio_queue"].put(live_frame)

			if not bg["thread"].is_alive():
				# Background thread finished — check abort/quit status first.
				_res_early = bg.get("result", {})
				_was_aborted_early = _res_early.get("aborted", False) if _res_early else False
				_quit_to_normal_early = _res_early.get("quit_to_normal", False) if _res_early else False
				_any_interrupt_early = _was_aborted_early or _quit_to_normal_early

				# Flush any remaining ack audio from the queue, but only if FC
				# completed normally (not interrupted).
				tts_q = bg.get("tts_audio_output_queue")
				if tts_q is not None and not tts_q.empty() and not _any_interrupt_early:
					import torch as _torch
					flush_chunks = []
					while not tts_q.empty():
						try:
							flush_chunks.append(tts_q.get_nowait())
						except _queue_module.Empty:
							break
					if flush_chunks:
						flush_cat = _torch.cat(flush_chunks, dim=-1).to(
							self.s2s_model.device, dtype=_torch.float32
						)
						if flush_cat.dim() == 2:
							flush_cat = flush_cat.unsqueeze(0)
						state = self.get_or_create_state(stream_id)
						state.update_state(flush_cat)
						tts_rate = getattr(self.s2s_model, "target_sample_rate", self.output_sample_rate)
						logging.info(
							"[FC Async NB] Flushed %.2fs remaining TTS audio from queue on thread done",
							flush_cat.shape[-1] / tts_rate,
						)

				# Background thread finished — check if it exited via interrupt.
				res = bg["result"]
				was_aborted = _was_aborted_early
				quit_to_normal = _quit_to_normal_early
				natural_interrupt = res.get("natural_interrupt", False) if res else False

				if not was_aborted and not quit_to_normal and not natural_interrupt and res and "async_steps" in res:
					# Normal completion: apply FC context updates so the main thread
					# can resume generating the verbal response.
					async_steps = res["async_steps"]
					updated_cache = res["updated_cache"]
					tts_state_out = res["tts_state_out"]
					rnnt_hyps_out = res["rnnt_hyps_out"]

					if updated_cache is not None:
						context.dynamic_cache = updated_cache
					if rnnt_hyps_out is not None:
						context.rnnt_partial_hypotheses = rnnt_hyps_out
					if tts_state_out is not None:
						context.code = tts_state_out["code"]
						context.past_key_values = tts_state_out["past_key_values"]
						context.codec_cache = tts_state_out.get("codec_cache", context.codec_cache)
					context.frame_idx += async_steps
					logging.info(
						"[FC Async NB] Context advanced by %d async steps, TTS codec state restored",
						async_steps,
					)

				# Drain any remaining RNNT text from the queue and apply final hypothesis.
				rnnt_tq = bg.get("rnnt_text_queue")
				if rnnt_tq is not None and not rnnt_tq.empty():
					latest_rnnt_text = None
					while not rnnt_tq.empty():
						try:
							latest_rnnt_text = rnnt_tq.get_nowait()
						except _queue_module.Empty:
							break
					if latest_rnnt_text is not None:
						state = self.get_or_create_state(stream_id)
						state.output_asr_text_str = latest_rnnt_text

				del self._fc_async_bg[stream_id]

				if quit_to_normal:
					# User interrupted during FC async — quit_async_event fired.
					# The async loop stopped gracefully WITHOUT killing vLLM sessions.
					# vLLM KV cache (LLM + TTS) remains intact under the same request_id.
					# We just advance frame_idx, reset perception_cache, clear fc_state,
					# insert EOS, and fall through to infer_one_step — vLLM continues
					# from exactly where it left off.
					if res and "async_steps" in res:
						_async_steps_qt = res["async_steps"]
						_rnnt_hyps_qt = res.get("rnnt_hyps_out")
						if _rnnt_hyps_qt is not None:
							context.rnnt_partial_hypotheses = _rnnt_hyps_qt
						context.frame_idx += _async_steps_qt
						# Reset perception_cache: async thread ran its own perception
						# pipeline so sizes are out of sync. Forces full-buffer rebuild.
						if context.perception_cache is not None:
							context.perception_cache.cache_last_channel = None
							context.perception_cache.cache_last_time = None
							context.perception_cache.cache_last_channel_len = None
						# Insert EOS in agent text + PAD in function channel at quit frame.
						# EOS: agent stopped. PAD: function channel silent (no FC).
						# Together they signal both channels are quiet, helping the model
						# break out of FC continuation after resume.
						_eos_id = getattr(
							self.s2s_model.model.stt_model, "text_eos_id", None
						)
						_pad_id = getattr(
							self.s2s_model.model.stt_model, "text_pad_id", None
						)
						if context.frame_idx > 0:
							if _eos_id is not None:
								context.gen_text[0, context.frame_idx - 1] = _eos_id
							if _pad_id is not None and context.gen_function_text is not None:
								context.gen_function_text[0, context.frame_idx - 1] = _pad_id
						logging.info(
							"[FC Async] Quit-to-normal — applied context (%d async steps), "
							"EOS+PAD inserted at frame %d for stream %d. vLLM sessions kept alive.",
							_async_steps_qt, context.frame_idx - 1, stream_id,
						)
					# Reset fc_state: shared by reference, may be mid-call.
					if context.fc_state is not None:
						context.fc_state.update({
							"active": False,
							"call_tokens": [],
							"completed_calls": [],
							"forced_function_tokens": [],
							"injecting_response": False,
						})
					context.tool_response_text = None
					reset_fn = getattr(self.s2s_model, "_reset_rnnt_turn_taking_state", None)
					if callable(reset_fn):
						reset_fn()
					# Fall through to infer_one_step — fc_state is clean, EOS signals
					# agent stopped, vLLM keeps full conversation context, turn-taking
					# handles BOS when user finishes speaking (EOU) → agent answers.
				elif was_aborted:
					# Legacy abort path (abort_event fired) — kept for non-vLLM mode.
					if res and "async_steps" in res:
						_async_steps_ab = res["async_steps"]
						_upd_cache_ab = res.get("updated_cache")
						_tts_out_ab = res.get("tts_state_out")
						_rnnt_hyps_ab = res.get("rnnt_hyps_out")
						if _upd_cache_ab is not None:
							context.dynamic_cache = _upd_cache_ab
						if _tts_out_ab is not None:
							context.code = _tts_out_ab["code"]
							context.past_key_values = None
							context.codec_cache = _tts_out_ab.get("codec_cache", context.codec_cache)
						if _rnnt_hyps_ab is not None:
							context.rnnt_partial_hypotheses = _rnnt_hyps_ab
						context.frame_idx += _async_steps_ab
						if context.perception_cache is not None:
							context.perception_cache.cache_last_channel = None
							context.perception_cache.cache_last_time = None
							context.perception_cache.cache_last_channel_len = None
						_eos_id = getattr(self.s2s_model.model.stt_model, "text_eos_id", None)
						if _eos_id is not None and context.frame_idx > 0:
							context.gen_text[0, context.frame_idx - 1] = _eos_id
						logging.info(
							"[FC Async] Aborted — applied context (%d async steps), "
							"EOS inserted at frame %d for stream %d",
							_async_steps_ab, context.frame_idx - 1, stream_id,
						)
					if context.fc_state is not None:
						context.fc_state.update({
							"active": False,
							"call_tokens": [],
							"completed_calls": [],
							"forced_function_tokens": [],
							"injecting_response": False,
						})
					context.tool_response_text = None
					self._abort_stream_request(stream_id)
					reset_fn = getattr(self.s2s_model, "_reset_rnnt_turn_taking_state", None)
					if callable(reset_fn):
						reset_fn()
				elif natural_interrupt:
					# The model's agent text head produced a non-PAD token while
					# receiving real user audio — it naturally decided to stop FC
					# and respond to the user's interrupt question.
					#
					# Safe to apply updated_cache here because function/agent text
					# channels are separate: partial FC tokens in the function channel
					# do NOT confuse agent text generation.
					#
					# Reset perception_cache: background thread ran its own perception
					# pipeline without updating context.perception_cache, so sizes
					# are out of sync. Resetting forces one full-buffer perception
					# rebuild on the next infer_one_step (no history lost — 5.6s
					# audio buffer has everything).
					if res and "async_steps" in res:
						_async_steps_ni = res["async_steps"]
						_upd_cache_ni = res.get("updated_cache")
						_tts_out_ni = res.get("tts_state_out")
						_rnnt_hyps_ni = res.get("rnnt_hyps_out")
						if _upd_cache_ni is not None:
							context.dynamic_cache = _upd_cache_ni
						if _tts_out_ni is not None:
							context.code = _tts_out_ni["code"]
							context.past_key_values = _tts_out_ni["past_key_values"]
							context.codec_cache = _tts_out_ni.get("codec_cache", context.codec_cache)
						if _rnnt_hyps_ni is not None:
							context.rnnt_partial_hypotheses = _rnnt_hyps_ni
						context.frame_idx += _async_steps_ni
						if context.perception_cache is not None:
							context.perception_cache.cache_last_channel = None
							context.perception_cache.cache_last_time = None
							context.perception_cache.cache_last_channel_len = None
						logging.info(
							"[FC Async] Natural interrupt — applied context (%d async steps), "
							"perception_cache reset for stream %d",
							_async_steps_ni, stream_id,
						)
					# Reset fc_state: shared by reference, may be mid-call.
					if context.fc_state is not None:
						context.fc_state.update({
							"active": False,
							"call_tokens": [],
							"completed_calls": [],
							"forced_function_tokens": [],
							"injecting_response": False,
						})
					context.tool_response_text = None
					reset_fn = getattr(self.s2s_model, "_reset_rnnt_turn_taking_state", None)
					if callable(reset_fn):
						reset_fn()
					# Fall through to infer_one_step — gen_text already has the
					# first response token written by the background thread.
					# vLLM session kept alive (no _abort_stream_request) so the model
					# resumes at full speed without re-prefill cost.
				else:
					# FC state machine complete — fc_state is clean, tool_response_text cleared.
					# Main thread falls through to normal infer_one_step for verbal response.
					_trq = getattr(context, "tool_response_queue", [])
					if _trq:
						context.tool_response_text = _trq.pop(0)
						logging.info("[FC Async NB] Advanced to next tool response (%d remaining)", len(_trq))
					else:
						context.tool_response_text = None
					logging.info("[FC Async NB] Background thread done, resuming normal flow for stream %d", stream_id)
					# Fall through to normal processing for this frame.

			else:
				# Thread still running — drain at most one frame's worth of TTS
				# audio (80ms) so the client gets real-time paced acknowledgement
				# speech instead of silence.  The background thread may produce
				# audio much faster than real-time (LLM speed); rate-limiting the
				# drain prevents the client from receiving a burst of audio.
				#
				# Once quit_async_event is set (user BOU detected), stop sending TTS
				# audio immediately — drain and DISCARD the queue so the client
				# receives silence rather than a mid-word acknowledgement fragment.

				# ── BOU detection: log user barge-in onset during FC async ──
				# We do NOT fire quit_async_event on BOU — the tool-call thread keeps
				# running so the result remains available after the user finishes speaking.
				# Agent EOS is injected by _maybe_apply_rnnt_turn_taking() in the wrapper.
				# rnnt_partial_hypotheses is a state dict; use nonblank_total as proxy for user speech
				_hyp = context.rnnt_partial_hypotheses
				_curr_hyp_len = (
					int(_hyp['nonblank_total'][0].item())
					if isinstance(_hyp, dict) and 'nonblank_total' in _hyp else 0
				)
				if not bg.get("bou_detected", False):
					if _curr_hyp_len > bg.get("last_rnnt_text_len", _curr_hyp_len):
						bg["bou_detected"] = True
						logging.info(
							"[FC Async] BOU onset (user speaking) — tool-call thread continues for stream %d",
							stream_id,
						)

				# ── Agent EOU: kill tool-call thread when agent EOS fires ──
				# The wrapper sets _agent_eos_just_fired=True when it inserts agent EOS
				# (BOU barge-in detected). At that point the tool result is no longer
				# needed for the current turn — fire quit_async_event to stop the thread.
				_quit_ev_eos = bg.get("quit_async_event")
				if (_quit_ev_eos is not None and not _quit_ev_eos.is_set()
						and getattr(self.s2s_model, "_agent_eos_just_fired", False)):
					self.s2s_model._agent_eos_just_fired = False  # consume the flag
					_quit_ev_eos.set()
					logging.info(
						"[FC Async] Agent EOS fired — killing tool-call thread for stream %d",
						stream_id,
					)
				# ────────────────────────────────────────────────────────────────────

				tts_q = bg.get("tts_audio_output_queue")
				_abort_ev = bg.get("abort_event")
				_quit_ev = bg.get("quit_async_event")
				_abort_fired = (
					(_abort_ev is not None and _abort_ev.is_set()) or
					(_quit_ev is not None and _quit_ev.is_set())
				)
				if tts_q is not None and not tts_q.empty():
					if _abort_fired:
						# Drain and discard — client gets silence while waiting
						# for the quit/abort to complete.
						while not tts_q.empty():
							try:
								tts_q.get_nowait()
							except _queue_module.Empty:
								break
						logging.debug(
							"[FC Async] Interrupt fired — discarding TTS queue for stream %d",
							stream_id,
						)
					else:
						tts_rate = getattr(self.s2s_model, "target_sample_rate", self.output_sample_rate)
						max_samples = int(tts_rate * self.chunk_size_in_secs)
						chunks = []
						drained = 0
						while drained < max_samples and not tts_q.empty():
							try:
								chunk = tts_q.get_nowait()
								chunks.append(chunk)
								drained += chunk.shape[-1]
							except _queue_module.Empty:
								break
						if chunks:
							import torch as _torch
							tts_cat = _torch.cat(chunks, dim=-1).to(
								self.s2s_model.device, dtype=_torch.float32
							)
							if tts_cat.dim() == 2:
								tts_cat = tts_cat.unsqueeze(0)
							state = self.get_or_create_state(stream_id)
							state.update_state(tts_cat)
				# Drain RNNT text updates from background thread — keep latest only.
				rnnt_tq = bg.get("rnnt_text_queue")
				if rnnt_tq is not None and not rnnt_tq.empty():
					latest_rnnt_text = None
					while not rnnt_tq.empty():
						try:
							latest_rnnt_text = rnnt_tq.get_nowait()
						except _queue_module.Empty:
							break
					if latest_rnnt_text is not None:
						state = self.get_or_create_state(stream_id)
						state.output_asr_text_str = latest_rnnt_text
				return

		# ---------------------------------------------------------------
		# Normal per-frame processing
		# ---------------------------------------------------------------
		audio_buffer = buffers[0]
		if audio_buffer.dim() == 1:
			audio_buffer = audio_buffer.unsqueeze(0)
		audio_buffer = audio_buffer.to(self.s2s_model.device, dtype=torch.float32)

		# Trim the buffer to exclude left padding (zeros at the beginning before buffer is filled)
		left_pad = left_paddings[0]
		if left_pad > 0:
			audio_buffer = audio_buffer[:, left_pad:]

		step_wall_start = time.time()
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
			rnnt_partial_hypotheses=context.rnnt_partial_hypotheses,
			fc_state=context.fc_state,
			tool_response_text=context.tool_response_text,
		)
		step_wall_end = time.time()

		# Record per-step timing
		if context.fc_timing is not None:
			context.fc_timing["frame_step_times"].append({
				"frame_idx": context.frame_idx,
				"wall_start": step_wall_start,
				"wall_end": step_wall_end,
				"duration_ms": (step_wall_end - step_wall_start) * 1000,
			})
			# Detect agent response start: first non-pad text token produced
			if context.fc_timing["agent_response_start_wall"] is None:
				text_strs = result.get("predicted_text_strs", [])
				if text_strs and text_strs[0]:
					pad_id = self.s2s_model.model.stt_model.text_pad_id
					bos_id = getattr(self.s2s_model.model.stt_model, "text_bos_id", None)
					tok = result.get("predicted_text_tokens")
					has_real_token = False
					if tok is not None and tok.numel() > 0:
						flat = tok.reshape(-1)
						for v in flat.tolist():
							if v != pad_id and v != bos_id:
								has_real_token = True
								break
					if has_real_token:
						context.fc_timing["agent_response_start_wall"] = step_wall_end
						context.fc_timing["agent_response_start_frame"] = context.frame_idx
						context.fc_timing["events"].append({
							"event": "agent_response_start",
							"wall_time": step_wall_end,
							"frame_idx": context.frame_idx,
							"simulated_time_sec": context.frame_idx * 0.08,
						})

		# Persist FC state back to context
		if "fc_state" in result and result["fc_state"] is not None:
			context.fc_state = result["fc_state"]

		# FC interruption PAD override: after quit-to-normal, override both
		# Persist updated cache & clean finished streams
		self.context_manager.update_context(stream_ids, result, self.num_frames_per_chunk)

		# Append this chunk's TTS audio BEFORE the FC silence insertion
		# so the SOTC chunk's speech is ordered correctly in the output.
		# UI ASR source is configurable: 'rnnt' or 'asr_head' (see s2s.ui_asr_source in YAML)
		# When ui_asr_source=="rnnt", never fall back to ASR head (may be absent/untrained).
		use_rnnt = (self.ui_asr_source == "rnnt")
		self.log_output(
			frames, result["decoded_audio_new"], ready_feats, result["predicted_text_strs"],
			None if use_rnnt else result.get("asr_predicted_text_strs"),
		)
		fc_text_strs = result.get("function_predicted_text_strs")
		if fc_text_strs:
			for idx, frame in enumerate(frames):
				if ready_feats[idx] and idx < len(fc_text_strs) and fc_text_strs[idx]:
					fc_state_obj = self.get_or_create_state(frame.stream_id)
					fc_state_obj.output_function_text_str += fc_text_strs[idx]
		# rnnt_partial_hypotheses is now a step-state dict — no text extraction available
		if False:
			pass

		# FC Sync: if EOTC was detected in non-async mode, execute tool and queue response
		if (context.fc_state is not None
				and not self.s2s_model._fc_async_enabled
				and self.tool_registry
				and context.tool_response_text is None):
			completed = context.fc_state.get("completed_calls", [])
			handled = context.fc_state.get("_sync_handled_calls", 0)
			if len(completed) > handled:
				call_text = completed[-1]
				context.fc_state["_sync_handled_calls"] = len(completed)
				logging.info("[FC Sync] EOTC detected, executing tool for: %s", call_text[:300])

				tool_response = self._execute_tool_call(call_text)
				if tool_response:
					wrapped_response = f"<TOOL_RESPONSE>[{tool_response}]</TOOL_RESPONSE>"
					fc_state_obj = self.get_or_create_state(stream_ids[0])
					if self.s2s_model._fc_sotc_id and self.s2s_model._fc_eotc_id:
						sotc_text = self.s2s_model.tokenizer.ids_to_text([self.s2s_model._fc_sotc_id])
						eotc_text = self.s2s_model.tokenizer.ids_to_text([self.s2s_model._fc_eotc_id])
						fc_state_obj.output_function_text_str += f"{sotc_text}{call_text}{eotc_text}\n"
					fc_state_obj.output_function_text_str += f"[INJECTED_RESPONSE] {wrapped_response}\n"
					if self.s2s_model._fc_convert_num_to_text:
						converted = self.s2s_model._convert_tool_response_nums_to_text(wrapped_response)
						fc_state_obj.output_function_text_str += f"[CONVERTED_RESPONSE] {converted}\n"
					response_tokens = self.s2s_model._build_fc_response_tokens(wrapped_response)
					context.fc_state["forced_function_tokens"] = response_tokens
					context.fc_state["injecting_response"] = True
					logging.info(
						"[FC Sync] Queued %d response tokens for injection: %s",
						len(response_tokens), wrapped_response[:200],
					)
				else:
					logging.info("[FC Sync] No built-in handler for call: %s", call_text[:200])

		# FC Async: if SOTC was detected, run the fast async loop
		_enter_async = False
		if (context.fc_state is not None
				and context.fc_state.pop("trigger_async", False)
				and self.s2s_model._fc_async_enabled):
			_enter_async = True

		if _enter_async:
			sotc_frame = context.frame_idx - 1
			fc_wall_start = time.time()

			# Record SOTC (tool call start)
			if context.fc_timing is not None:
				context.fc_timing["events"].append({
					"event": "tool_call_start (SOTC)",
					"wall_time": fc_wall_start,
					"frame_idx": sotc_frame,
					"simulated_time_sec": sotc_frame * 0.08,
				})

			use_cache = context.dynamic_cache is not None
			tool_resp = context.tool_response_text

			# Build realtime audio context if the full audio was pre-loaded (offline sim).
			rt_audio_ctx = None
			if context.rt_audio_signal is not None:
				bufferer = self.bufferer.bufferers.get(stream_id)
				if bufferer is not None:
					buf_tensor = bufferer.sample_buffer.to(
						self.s2s_model.device, dtype=self.s2s_model.dtype,
					).unsqueeze(0)
					buf_fill = context.rt_audio_buffer_size_samples - bufferer.left_padding
				else:
					buf_tensor = torch.zeros(
						1, context.rt_audio_buffer_size_samples,
						device=self.s2s_model.device, dtype=self.s2s_model.dtype,
					)
					buf_fill = 0
				rt_audio_ctx = {
					"audio_signal_tensor": context.rt_audio_signal,
					"next_audio_frame": context.frame_idx,
					"total_audio_frames": context.rt_audio_total_frames,
					"audio_buffer": buf_tensor.clone(),
					"buffer_fill_level": buf_fill,
					"buffer_size_samples": context.rt_audio_buffer_size_samples,
					"wall_start": fc_wall_start,
					"frames_consumed": 0,
				}

			# TTS state snapshot for async warmup
			_tts_state_for_async = None
			if self.s2s_model.decode_audio and context.code is not None:
				_tts_state_for_async = {
					"code": context.code,
					"past_key_values": context.past_key_values,
					"subword_mask": context.subword_mask,
					"codec_cache": context.codec_cache,
				}

			# ------------------------------------------------------------------
			# Non-blocking mode (Triton server): no pre-loaded audio available.
			# Spawn a background thread to run the FC async LLM loop while
			# execute() calls keep delivering live audio frames.  Each frame
			# is pushed to live_audio_queue so the RNNT can track user speech.
			# ------------------------------------------------------------------
			if context.rt_audio_signal is None:
				logging.info(
					"[FC Async NB] Triggered at frame %d — spawning background thread",
					sotc_frame,
				)

				# Capture current audio buffer state for the live perception worker
				_live_audio_queue = _queue_module.Queue()
				_tts_audio_output_queue = _queue_module.Queue()
				_rnnt_text_queue = _queue_module.Queue()
				_abort_event = threading.Event()
				_quit_async_event = threading.Event()
				_buf_size_samples = int(self.buffer_size_in_secs * self.input_sample_rate)
				_live_buf = None
				_live_buf_fill = 0
				bufferer = self.bufferer.bufferers.get(stream_id)
				if bufferer is not None:
					_live_buf = bufferer.sample_buffer.clone().unsqueeze(0).cpu()
					_live_buf_fill = max(0, _buf_size_samples - bufferer.left_padding)

				# Snapshot all context state the thread needs
				_fc_state_snap = context.fc_state
				_gen_text_snap = context.gen_text.clone()
				_gen_asr_text_snap = context.gen_asr_text.clone()
				_gen_func_text_snap = context.gen_function_text.clone()
				_dyn_cache_snap = context.dynamic_cache
				_input_embs_snap = context.input_embeds_history if not use_cache else []
				_rnnt_hyps_snap = context.rnnt_partial_hypotheses
				_tool_resp_snap = tool_resp
				_tts_snap = _tts_state_for_async
				_req_id_snap = request_id
				_sotc_frame_snap = sotc_frame

				# Tool executor: called in two-phase mode after the LLM pauses at EOTC
				_execute_tool_fn_snap = self._execute_tool_call
				_fc_convert_snap = self.s2s_model._fc_convert_num_to_text
				_sotc_id_snap = self.s2s_model._fc_sotc_id
				_eotc_id_snap = self.s2s_model._fc_eotc_id
				_stream_id_snap = stream_id
				_s2s_model_snap = self.s2s_model
				_get_state_fn = self.get_or_create_state

				_bg_result = {}  # populated in-place by the thread

				def _fc_async_thread_fn(
					bg_result,
					fc_state, gen_text, gen_asr_text, gen_function_text,
					dynamic_cache, input_embeds_history, rnnt_hyps,
					tool_resp, rt_audio_ctx, request_id, tts_state,
					sotc_frame, live_audio_queue, live_buf, live_buf_fill,
					buf_size_samples, stream_id, s2s_model, execute_tool_fn,
					fc_convert, sotc_id, eotc_id, get_state_fn,
					tts_audio_output_queue, acknowledgement_tokens, rnnt_text_queue,
					abort_event, quit_async_event, reminder_tokens,
					on_hold_token_map, generic_on_hold_token_list,
					timeout_fallback_tokens, error_401_tokens, error_401_redirect_tokens,
					tool_timeout_sec,
				):
					try:
						import torch as _torch
						with _torch.inference_mode():
							# LLM runs at full speed: SOTC → tool call → EOTC → (if tool_resp provided) inject response + EOTR
							async_steps, updated_cache, tts_out, tts_chunks, rnnt_hyps_out = \
								s2s_model._run_fc_async_steps(
									fc_state=fc_state,
									gen_text=gen_text,
									gen_asr_text=gen_asr_text,
									gen_function_text=gen_function_text,
									current_frame_idx=sotc_frame,
									dynamic_cache=dynamic_cache,
									input_embeds_history=input_embeds_history,
									tool_response_text=tool_resp,
									realtime_audio=rt_audio_ctx,
									request_id=request_id,
									tts_state=tts_state,
									live_audio_queue=live_audio_queue,
									live_audio_buffer=live_buf,
									live_buffer_fill=live_buf_fill,
									live_buffer_size_samples=buf_size_samples,
									rnnt_partial_hypotheses=rnnt_hyps,
									tts_audio_output_queue=tts_audio_output_queue,
									acknowledgement_tokens=acknowledgement_tokens,
									rnnt_text_queue=rnnt_text_queue,
									abort_event=abort_event,
									quit_async_event=quit_async_event,
									)

							# Two-phase async FC: the first LLM run above generated the
							# tool call (SOTC→EOTC) and paused (awaiting_response=True).
							# Now execute the API call, then run the LLM a second time to
							# inject <TOOL_RESPONSE>...</TOOL_RESPONSE> as forced tokens
							# and let it predict <EOTR>.  After <EOTR> fc_state is fully
							# clean; the verbal response is generated in normal
							# frame-by-frame mode once the main thread resumes.
							# TTS during forced-token injection produces silence (agent
							# text = PAD), so we do NOT pass tts_audio_output_queue here.
							if fc_state.pop("awaiting_response", False):
								call_text = fc_state.get("last_call_text", "")
								phase2_start_t = fc_state.pop("phase1_end_t", sotc_frame + async_steps)
								fc_state_obj = get_state_fn(stream_id)
								if call_text and sotc_id and eotc_id:
									sotc_text = s2s_model.tokenizer.ids_to_text([sotc_id])
									eotc_text = s2s_model.tokenizer.ids_to_text([eotc_id])
									fc_state_obj.output_function_text_str += f"{sotc_text}{call_text}{eotc_text}\n"
								# ----------------------------------------------------------
								# On-hold audio + timeout state machine
								# Layer 1: per-tool reminder (JSON lookup)
								# Layer 2: up to 2 generic messages (1s gap) if API slow
								# Layer 3: timeout fallback if still blocked after Layer 2
								# Error:   401 branch plays apology + redirect
								# ----------------------------------------------------------
								_tts_for_reminder = tts_out if tts_out is not None else tts_state
								_aborted = abort_event is not None and abort_event.is_set()

								# Resolve per-tool reminder tokens. Each entry in on_hold_token_map is
								# now a list of tokenized variants; pick one at random per call.
								_tool_name = _parse_tool_name(call_text)
								_reminder_tok = None
								if on_hold_token_map:
									_options = (on_hold_token_map.get(_tool_name)
											or on_hold_token_map.get("default"))
									if _options:
										_reminder_tok = random.choice(_options)
								if _reminder_tok is None:
									_reminder_tok = reminder_tokens  # backward compat

								# Output sample rate for computing audio playback duration
								_OUTPUT_SAMPLE_RATE = getattr(self, "output_sample_rate", 22050)

								def _play(tok):
									"""Generate TTS audio and block until it has had time to play."""
									if tok and _tts_for_reminder is not None and not (abort_event is not None and abort_event.is_set()):
										_t0 = time.monotonic()
										_n_samples = s2s_model._run_tts_reminder(
											reminder_tokens=tok,
											tts_state=_tts_for_reminder,
											tts_audio_output_queue=tts_audio_output_queue,
											abort_event=abort_event,
											request_id=request_id,
										)
										# TTS generation is faster than real-time playback.
										# Sleep for the remaining playback duration so the caller
										# can treat _play() as blocking (audio finishes before returning).
										_gen_time = time.monotonic() - _t0
										_play_dur = _n_samples / _OUTPUT_SAMPLE_RATE if _n_samples > 0 else 0.0
										_remaining = _play_dur - _gen_time
										while _remaining > 0:
											if abort_event is not None and abort_event.is_set():
												break
											time.sleep(min(0.05, _remaining))
											_remaining -= 0.05

								tool_response = None
								if _tts_for_reminder is not None and not _aborted:
									# Run tool in background so audio plays concurrently
									with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _exec:
										_fut = _exec.submit(execute_tool_fn, call_text)

										# Layer 1 — per-tool reminder
										if _reminder_tok:
											logging.info("[FC Reminder] Layer 1: per-tool='%s' (%d tokens)", _tool_name, len(_reminder_tok))
											_play(_reminder_tok)

										# Layer 2 — generic messages (up to 2, with 1s gap between)
										_generic_pool = (
											random.sample(generic_on_hold_token_list, min(2, len(generic_on_hold_token_list)))
											if generic_on_hold_token_list else []
										)
										_played_generic = 0
										for _gen_tok in _generic_pool:
											if _fut.done() or (abort_event is not None and abort_event.is_set()):
												break
											time.sleep(1.0)
											if _fut.done():
												break
											logging.info("[FC Reminder] Layer 2: generic message %d/2", _played_generic + 1)
											_play(_gen_tok)
											_played_generic += 1

										# Layer 3 — tool still running after all messages: wait silently
										if not _fut.done():
											logging.warning("[FC Reminder] Layer 3: tool still running after on-hold messages — waiting silently")

										# Final wait — up to tool_timeout_sec after all messages
										try:
											tool_response = _fut.result(timeout=tool_timeout_sec)
										except concurrent.futures.TimeoutError:
											logging.warning("[FC Reminder] Tool timed out after %ss — using canned error response", tool_timeout_sec)
											tool_response = json.dumps({
												"error": "timeout",
												"message": "I'm sorry, that's taking longer than expected.",
											})
								else:
									tool_response = execute_tool_fn(call_text)

								# 401 / auth error branch
								if _response_is_401(tool_response):
									logging.warning("[FC Reminder] 401 error in tool response — playing apology")
									_play(error_401_tokens)
									time.sleep(0.5)
									_play(error_401_redirect_tokens)
									tool_response = json.dumps({
										"error": "401",
										"message": "Authentication failed. Please try again later.",
									})
								if tool_response:
									wrapped_response = f"<TOOL_RESPONSE>[{tool_response}]</TOOL_RESPONSE>"
									fc_state_obj.output_function_text_str += f"[INJECTED_RESPONSE] {wrapped_response}\n"
									if fc_convert:
										converted = s2s_model._convert_tool_response_nums_to_text(wrapped_response)
										fc_state_obj.output_function_text_str += f"[CONVERTED_RESPONSE] {converted}\n"
										wrapped_response = converted
									_tts_p2 = tts_out if tts_out is not None else tts_state
									async_steps_p2, updated_cache, tts_out, tts_chunks_p2, rnnt_hyps_out = \
										s2s_model._run_fc_async_steps(
											fc_state=fc_state,
											gen_text=gen_text,
											gen_asr_text=gen_asr_text,
											gen_function_text=gen_function_text,
											current_frame_idx=phase2_start_t - 1,
											dynamic_cache=updated_cache,
											input_embeds_history=input_embeds_history if updated_cache is None else [],
											tool_response_text=wrapped_response,
											realtime_audio=rt_audio_ctx,
											request_id=request_id,
											tts_state=_tts_p2,
											live_audio_queue=live_audio_queue,
											live_buffer_size_samples=buf_size_samples,
											rnnt_partial_hypotheses=rnnt_hyps_out,
											rnnt_text_queue=rnnt_text_queue,
											abort_event=abort_event,
											quit_async_event=quit_async_event,
											)
									tts_chunks = tts_chunks + tts_chunks_p2
									async_steps += async_steps_p2

						bg_result["async_steps"] = async_steps
						bg_result["updated_cache"] = updated_cache
						bg_result["tts_state_out"] = tts_out
						bg_result["tts_audio_chunks"] = tts_chunks
						bg_result["rnnt_hyps_out"] = rnnt_hyps_out
						bg_result["aborted"] = abort_event is not None and abort_event.is_set()
						bg_result["quit_to_normal"] = quit_async_event is not None and quit_async_event.is_set()
						bg_result["natural_interrupt"] = fc_state.get("natural_interrupt", False)
					except Exception as _exc:
						logging.error("[FC Async NB] Background thread failed: %s: %s",
									  type(_exc).__name__, _exc, exc_info=True)
						bg_result["async_steps"] = 0
						bg_result["updated_cache"] = dynamic_cache
						bg_result["tts_state_out"] = tts_state
						bg_result["tts_audio_chunks"] = []
						bg_result["rnnt_hyps_out"] = rnnt_hyps

				_bg_thread = threading.Thread(
					target=_fc_async_thread_fn,
					kwargs=dict(
						bg_result=_bg_result,
						fc_state=_fc_state_snap,
						gen_text=_gen_text_snap,
						gen_asr_text=_gen_asr_text_snap,
						gen_function_text=_gen_func_text_snap,
						dynamic_cache=_dyn_cache_snap,
						input_embeds_history=_input_embs_snap,
						rnnt_hyps=_rnnt_hyps_snap,
						tool_resp=_tool_resp_snap,
						rt_audio_ctx=rt_audio_ctx,
						request_id=_req_id_snap,
						tts_state=_tts_snap,
						sotc_frame=_sotc_frame_snap,
						live_audio_queue=_live_audio_queue,
						live_buf=_live_buf,
						live_buf_fill=_live_buf_fill,
						buf_size_samples=_buf_size_samples,
						stream_id=_stream_id_snap,
						s2s_model=_s2s_model_snap,
						execute_tool_fn=_execute_tool_fn_snap,
						fc_convert=_fc_convert_snap,
						sotc_id=_sotc_id_snap,
						eotc_id=_eotc_id_snap,
						get_state_fn=_get_state_fn,
						tts_audio_output_queue=_tts_audio_output_queue,
						acknowledgement_tokens=(
							random.choice(self._fc_ack_token_list) if self._fc_ack_token_list else None
						),
						reminder_tokens=(
							random.choice(self._fc_reminder_token_list) if self._fc_reminder_token_list else None
						),
						on_hold_token_map=self._fc_on_hold_token_map,
						generic_on_hold_token_list=self._fc_generic_on_hold_token_list,
						timeout_fallback_tokens=self._fc_timeout_fallback_tokens,
						error_401_tokens=self._fc_401_error_tokens,
						error_401_redirect_tokens=self._fc_401_redirect_tokens,
						tool_timeout_sec=self._fc_tool_timeout_sec,
						rnnt_text_queue=_rnnt_text_queue,
						abort_event=_abort_event,
						quit_async_event=_quit_async_event,
					),
					daemon=True,
				)
				_bg_thread.start()

				# Seed last_rnnt_text_len with nonblank_total at FC start (dict-based state)
				_initial_rnnt_len = 0
				if isinstance(_rnnt_hyps_snap, dict) and 'nonblank_total' in _rnnt_hyps_snap:
					_initial_rnnt_len = int(_rnnt_hyps_snap['nonblank_total'][0].item())

				self._fc_async_bg[stream_id] = {
					"thread": _bg_thread,
					"live_audio_queue": _live_audio_queue,
					"tts_audio_output_queue": _tts_audio_output_queue,
					"rnnt_text_queue": _rnnt_text_queue,
					"result": _bg_result,
					"fc_wall_start": fc_wall_start,
					"abort_event": _abort_event,
					"quit_async_event": _quit_async_event,
					# Speech-interrupt tracking (BOU + elapsed frames).
					# bou_detected: True once RNNT hypothesis first exceeds its
					#   initial length, i.e. user started speaking during FC async.
					# bou_frames_elapsed: 80 ms frames counted since BOU fired.
					#   When >= rnnt_fc_interrupt_frames (default 3 × 80 ms = 240 ms)
					#   quit_async_event is fired to stop the async loop gracefully.
					# last_rnnt_text_len: initial hypothesis length at FC start,
					#   used as the BOU detection baseline (not a high-water mark).
					"bou_detected": False,
					"bou_frames_elapsed": 0,
					"last_rnnt_text_len": _initial_rnnt_len,
				}
				logging.info(
					"[FC Async NB] Background thread spawned for stream %d at frame %d",
					stream_id, sotc_frame,
				)
				# Return immediately — the next execute() call will detect active
				# FC async and push audio to _live_audio_queue.
				return

			# ------------------------------------------------------------------
			# Blocking mode (offline simulation): pre-loaded audio available.
			# Keep existing wall-clock gated behaviour.
			# ------------------------------------------------------------------
			logging.info(
				"[FC Async] Triggered at frame %d — entering async loop (audio paused)",
				sotc_frame,
			)
			async_steps, updated_cache, _tts_state_out, _tts_audio_chunks, _rnnt_hyps_out = self.s2s_model._run_fc_async_steps(
				fc_state=context.fc_state,
				gen_text=context.gen_text,
				gen_asr_text=context.gen_asr_text,
				gen_function_text=context.gen_function_text,
				current_frame_idx=sotc_frame,
				dynamic_cache=context.dynamic_cache,
				input_embeds_history=context.input_embeds_history if not use_cache else [],
				tool_response_text=tool_resp,
				realtime_audio=rt_audio_ctx,
				request_id=request_id,
				tts_state=_tts_state_for_async,
				rnnt_partial_hypotheses=context.rnnt_partial_hypotheses,
			)
			eotc_wall = time.time()
			if updated_cache is not None:
				context.dynamic_cache = updated_cache
			if _rnnt_hyps_out is not None:
				context.rnnt_partial_hypotheses = _rnnt_hyps_out
			if _tts_state_out is not None:
				context.code = _tts_state_out["code"]
				context.past_key_values = _tts_state_out["past_key_values"]
				context.codec_cache = _tts_state_out.get("codec_cache", context.codec_cache)
			context.frame_idx += async_steps

			# Record EOTC (tool call end) and async step speed
			if context.fc_timing is not None:
				async_duration = eotc_wall - fc_wall_start
				event_name = "tool_call_end (EOTC)"
				if tool_resp:
					event_name = "tool_call_end (EOTC) + response injected"
				context.fc_timing["events"].append({
					"event": event_name,
					"wall_time": eotc_wall,
					"frame_idx": sotc_frame + async_steps,
					"simulated_time_sec": (sotc_frame + async_steps) * 0.08,
					"async_steps": async_steps,
					"async_wall_sec": async_duration,
					"async_tokens_per_sec": async_steps / async_duration if async_duration > 0 else 0,
				})

			# Record API call timing if simulated latency was used
			if context.fc_timing is not None and tool_resp:
				api_start = context.fc_state.get("api_call_start_wall")
				api_end = context.fc_state.get("api_call_end_wall")
				api_frame = context.fc_state.get("api_call_start_frame")
				if api_start is not None and api_end is not None:
					context.fc_timing["events"].append({
						"event": "api_call_start",
						"wall_time": api_start,
						"frame_idx": api_frame,
						"simulated_time_sec": api_frame * 0.08 if api_frame else None,
					})
					context.fc_timing["events"].append({
						"event": "api_call_end",
						"wall_time": api_end,
						"frame_idx": api_frame,
						"simulated_time_sec": api_frame * 0.08 if api_frame else None,
						"api_latency_ms": (api_end - api_start) * 1000,
					})

				tr_start_wall = context.fc_state.get("tool_response_inject_start_wall")
				tr_end_wall = context.fc_state.get("tool_response_inject_end_wall")
				tr_start_frame = context.fc_state.get("tool_response_inject_start_frame")
				tr_end_frame = context.fc_state.get("tool_response_inject_end_frame")
				tr_num_tokens = context.fc_state.get("tool_response_num_tokens", 0)
				if tr_start_wall is not None:
					context.fc_timing["events"].append({
						"event": "tool_response_start",
						"wall_time": tr_start_wall,
						"frame_idx": tr_start_frame,
						"simulated_time_sec": tr_start_frame * 0.08 if tr_start_frame else None,
					})
				if tr_end_wall is not None:
					tr_duration = tr_end_wall - tr_start_wall if tr_start_wall else 0
					context.fc_timing["events"].append({
						"event": "tool_response_end",
						"wall_time": tr_end_wall,
						"frame_idx": tr_end_frame,
						"simulated_time_sec": tr_end_frame * 0.08 if tr_end_frame else None,
						"response_tokens": tr_num_tokens,
						"response_wall_sec": tr_duration,
						"response_tokens_per_sec": tr_num_tokens / tr_duration if tr_duration > 0 else 0,
					})

			# Phase 2 (two-phase): if no tool_response_text was available upfront,
			# the async loop exited at EOTC. Execute tool then re-enter async.
			if context.fc_state.pop("awaiting_response", False):
				call_text = context.fc_state.get("last_call_text", "")
				phase2_start_t = context.fc_state.pop("phase1_end_t", sotc_frame + async_steps)

				# Save function channel call text to state for logging/demo recording
				fc_state_obj = self.get_or_create_state(stream_ids[0])
				if call_text:
					sotc_text = self.s2s_model.tokenizer.ids_to_text([self.s2s_model._fc_sotc_id])
					eotc_text = self.s2s_model.tokenizer.ids_to_text([self.s2s_model._fc_eotc_id])
					fc_state_obj.output_function_text_str += f"{sotc_text}{call_text}{eotc_text}\n"
					logging.info("[FC Async] Phase 2: raw call_text = %s", call_text[:500])

				tool_response = self._execute_tool_call(call_text)

				if tool_response:
					wrapped_response = f"<TOOL_RESPONSE>[{tool_response}]</TOOL_RESPONSE>"
					fc_state_obj.output_function_text_str += f"[INJECTED_RESPONSE] {wrapped_response}\n"
					if self.s2s_model._fc_convert_num_to_text:
						converted = self.s2s_model._convert_tool_response_nums_to_text(wrapped_response)
						fc_state_obj.output_function_text_str += f"[CONVERTED_RESPONSE] {converted}\n"
					logging.info(
						"[FC Async] Phase 2: tool returned %d chars, wrapped as: %s",
						len(tool_response), wrapped_response[:300],
					)
					_tts_state_p2 = None
					if self.s2s_model.decode_audio and context.code is not None:
						_tts_state_p2 = {
							"code": context.code,
							"past_key_values": context.past_key_values,
							"subword_mask": context.subword_mask,
							"codec_cache": context.codec_cache,
						}
					async_steps_p2, updated_cache, _tts_state_out_p2, _tts_audio_chunks_p2, _rnnt_hyps_p2 = self.s2s_model._run_fc_async_steps(
						fc_state=context.fc_state,
						gen_text=context.gen_text,
						gen_asr_text=context.gen_asr_text,
						gen_function_text=context.gen_function_text,
						current_frame_idx=phase2_start_t - 1,
						dynamic_cache=context.dynamic_cache,
						input_embeds_history=context.input_embeds_history if context.dynamic_cache is None else [],
						tool_response_text=wrapped_response,
						request_id=request_id,
						tts_state=_tts_state_p2,
						rnnt_partial_hypotheses=context.rnnt_partial_hypotheses,
					)
					if updated_cache is not None:
						context.dynamic_cache = updated_cache
					if _rnnt_hyps_p2 is not None:
						context.rnnt_partial_hypotheses = _rnnt_hyps_p2
					if _tts_state_out_p2 is not None:
						context.code = _tts_state_out_p2["code"]
						context.past_key_values = _tts_state_out_p2["past_key_values"]
						context.codec_cache = _tts_state_out_p2.get("codec_cache", context.codec_cache)
					if _tts_audio_chunks_p2:
						_tts_audio_chunks.extend(_tts_audio_chunks_p2)
					context.frame_idx += async_steps_p2
					logging.info(
						"[FC Async] Phase 2 completed: %d steps (total async: %d)",
						async_steps_p2, async_steps + async_steps_p2,
					)
				else:
					logging.info(
						"[FC Async] Phase 2: no built-in tool handler for call (call: %s)",
						call_text,
					)

			fc_wall_elapsed = time.time() - fc_wall_start

			# Insert audio into the output stream to reflect FC wait.
			state = self.get_or_create_state(stream_ids[0])
			tts_rate = getattr(self.s2s_model, "target_sample_rate", self.output_sample_rate)
			fc_wall_sec = fc_wall_elapsed
			if _tts_audio_chunks:
				tts_audio_cat = torch.cat(_tts_audio_chunks, dim=-1)
				tts_audio_device = tts_audio_cat.to(self.s2s_model.device)
				if tts_audio_device.dim() == 2:
					tts_audio_device = tts_audio_device.unsqueeze(0)
				tts_samples = tts_audio_device.shape[-1]
				target_samples = int(fc_wall_sec * tts_rate)
				if tts_samples < target_samples:
					padding = torch.zeros(
						tts_audio_device.shape[0], tts_audio_device.shape[1],
						target_samples - tts_samples,
						device=self.s2s_model.device, dtype=torch.float32,
					)
					tts_audio_device = torch.cat([tts_audio_device, padding], dim=-1)
				elif tts_samples > target_samples:
					tts_audio_device = tts_audio_device[:, :, :target_samples]
				state.update_state(tts_audio_device)
				logging.info(
					"[FC Async] Inserted %.2fs TTS-warmed audio (%d chunks, %d samples) "
					"into output (wall-clock=%.2fs, simulated=%.2fs)",
					tts_samples / tts_rate, len(_tts_audio_chunks), tts_samples,
					fc_wall_sec, async_steps * 0.08,
				)
			else:
				silence_samples = int(fc_wall_sec * tts_rate)
				if silence_samples > 0:
					silence_chunk = torch.zeros(
						1, 1, silence_samples,
						device=self.s2s_model.device, dtype=torch.float32,
					)
					state.update_state(silence_chunk)
					logging.info(
						"[FC Async] Inserted %.2fs silence (%d samples) into audio output "
						"(wall-clock time; simulated=%.2fs)",
						fc_wall_sec, silence_samples, async_steps * 0.08,
					)

			# Record total FC wall time
			if context.fc_timing is not None:
				context.fc_timing["events"].append({
					"event": "fc_async_complete",
					"wall_time": time.time(),
					"total_fc_wall_sec": fc_wall_elapsed,
				})

			# Account for real audio frames consumed during async.
			rt_consumed = rt_audio_ctx["frames_consumed"] if rt_audio_ctx is not None else 0
			if rt_consumed > 0:
				bufferer = self.bufferer.bufferers.get(stream_id)
				if bufferer is not None:
					buf_tensor = rt_audio_ctx["audio_buffer"]
					bufferer.sample_buffer = buf_tensor.squeeze(0).to(bufferer.sample_buffer.device, dtype=bufferer.sample_buffer.dtype)
					bufferer.left_padding = max(0, bufferer.buffer_size - rt_audio_ctx["buffer_fill_level"])
				logging.info(
					"[FC Async] Consumed %d real audio frames during async",
					rt_consumed,
				)

			# Advance to the next tool response in the queue for subsequent calls
			_tool_resp_queue = getattr(context, "tool_response_queue", [])
			if _tool_resp_queue:
				context.tool_response_text = _tool_resp_queue.pop(0)
				logging.info("[FC Async] Advanced to next tool response (%d remaining)", len(_tool_resp_queue))
			else:
				context.tool_response_text = None

			logging.info(
				"[FC Async] Done: %d async steps in %.3fs (rt_audio=%d frames). "
				"Audio resumes at frame %d",
				async_steps, fc_wall_elapsed, rt_consumed, context.frame_idx,
			)

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
					# Carry over FC timing data to state for saving later
					if ctx.fc_timing is not None:
						ctx.fc_timing["stream_end_wall"] = time.time()
						state.fc_timing = ctx.fc_timing
					# rnnt_partial_hypotheses is now a step-state dict — no text extraction
					pass

		self.context_manager.reset_slots(stream_ids, eos_flags)
		
		# Explicitly clean up bufferer and state for finished streams
		for stream_id, eos_flag in zip(stream_ids, eos_flags):
			if eos_flag:
				logging.debug(f"Ending stream {stream_id} - cleaning up bufferer and context")
				self.bufferer.rm_bufferer(stream_id)
				self._abort_stream_request(stream_id)
					# Note: We keep the state in _state_pool until finalization to save audio
				# It will be cleaned up in close_session()

	def prefill_for_new_stream(
		self,
		stream_id: int,
		system_prompt: str | None = None,
		tool_response_text: str | None = None,
	) -> bool:
		"""Prepare the pipeline for a new stream by resetting context and prefilling the system prompt.

		This is the public API for prefill-only calls (e.g. from the Triton backend)
		that need to initialize TTS speaker embeddings and/or inject a system prompt
		into the LLM KV cache *without* processing any audio.

		Args:
			stream_id: Unique identifier for the new stream.
			system_prompt: System prompt text. If *None*, falls back to
				the YAML-configured ``self.system_prompt``.
			tool_response_text: Pre-provided tool response for function-calling
				benchmarking.  Stored on the context so that the FC state
				machine can inject it when the model emits EOTC.

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

		# Store tool_response_text and initialize FC state on the context
		context, _ = self.context_manager.get_context([stream_id])
		if tool_response_text:
			try:
				parsed = json.loads(tool_response_text)
				if isinstance(parsed, list):
					context.tool_response_queue = list(parsed)
					context.tool_response_text = context.tool_response_queue.pop(0) if context.tool_response_queue else None
				else:
					context.tool_response_text = tool_response_text
					context.tool_response_queue = []
			except (json.JSONDecodeError, TypeError):
				context.tool_response_text = tool_response_text
				context.tool_response_queue = []
		has_fc = self.s2s_model.model.stt_model.function_head is not None
		if has_fc:
			context.fc_state = {
				"active": False,
				"call_tokens": [],
				"completed_calls": [],
				"forced_function_tokens": [],
				"injecting_response": False,
			}

		# Reset RNNT turn-taking state for the new stream
		reset_fn = getattr(self.s2s_model, "_reset_rnnt_turn_taking_state", None)
		if callable(reset_fn):
			reset_fn()

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
			tool_resp = None
			if opts is not None:
				if hasattr(opts, "system_prompt"):
					prompt = opts.system_prompt
				if hasattr(opts, "tool_response_text"):
					tool_resp = opts.tool_response_text

			self.prefill_for_new_stream(frames[0].stream_id, prompt, tool_response_text=tool_resp)
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

				# Insert silence into the user audio channel at each FC point
				# so both channels stay aligned in the stereo file.
				# Collect paired (SOTC, fc_async_complete) events in order,
				# then insert silence for each, adjusting positions for
				# previously inserted gaps.
				fc_timing = getattr(state, "fc_timing", None)
				if fc_timing is not None:
					events = fc_timing.get("events", [])
					sotc_events = [e for e in events if e["event"] == "tool_call_start (SOTC)"]
					complete_events = [e for e in events if e["event"] == "fc_async_complete" and "total_fc_wall_sec" in e]
					cumulative_inserted = 0
					for sotc_ev, compl_ev in zip(sotc_events, complete_events):
						sotc_frame = sotc_ev["frame_idx"]
						fc_wall_sec = compl_ev["total_fc_wall_sec"]
						insert_pos = int(sotc_frame * 0.08 * self.output_sample_rate) + cumulative_inserted
						insert_samples = int(fc_wall_sec * self.output_sample_rate)
						if insert_pos <= input_audio.shape[-1] and insert_samples > 0:
							fc_silence = torch.zeros(insert_samples, dtype=input_audio.dtype)
							input_audio = torch.cat([
								input_audio[:insert_pos],
								fc_silence,
								input_audio[insert_pos:],
							], dim=-1)
							cumulative_inserted += insert_samples
							logging.info(
								"[Stereo] Inserted %.2fs silence into user channel at %.2fs "
								"(frame %d) to align with FC gap #%d",
								fc_wall_sec, sotc_frame * 0.08, sotc_frame,
								sotc_events.index(sotc_ev) + 1,
							)

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

				# Save RNNT ASR text if available
				rnnt_text_out = getattr(state, "final_rnnt_text", None)
				if isinstance(rnnt_text_out, str) and rnnt_text_out:
					try:
						with open(os.path.join(txt_dir, f"{base}_rnnt.txt"), "w", encoding="utf-8") as f:
							f.write(rnnt_text_out)
					except Exception:
						pass

				# Save function calling channel text if available
				token_data = state.get_token_tensors()
				if token_data is not None:
					_, _, total_frames, gen_function_text = token_data
					if gen_function_text is not None:
						tokenizer = self.s2s_model.tokenizer
						pad_id = self.s2s_model.model.stt_model.text_pad_id
						lengths = torch.tensor([total_frames], dtype=torch.long)
						fc_text_out = tokens_to_str(
							gen_function_text, lengths, tokenizer=tokenizer, pad_id=pad_id, eval_text_turn_taking=False
						)[0]
						if fc_text_out:
							try:
								with open(os.path.join(txt_dir, f"{base}_fc.txt"), "w", encoding="utf-8") as f:
									f.write(fc_text_out)
							except Exception:
								pass

				# Save FC timing report if available
				fc_timing = getattr(state, "fc_timing", None)
				if fc_timing is not None:
					try:
						timing_path = os.path.join(txt_dir, f"{base}_timing.txt")
						with open(timing_path, "w", encoding="utf-8") as tf:
							t0 = fc_timing.get("stream_start_wall", 0)
							t_end = fc_timing.get("stream_end_wall", t0)
							tf.write(f"=== FC Timing Report: {base} ===\n")
							tf.write(f"Total stream wall time: {t_end - t0:.3f}s\n\n")

							# Normal (lockstep) step speed stats
							steps = fc_timing.get("frame_step_times", [])
							if steps:
								durations = [s["duration_ms"] for s in steps]
								avg_ms = sum(durations) / len(durations)
								tf.write(f"--- Lockstep Step Timing (perception + LLM + TTS per chunk) ---\n")
								tf.write(f"  Total steps: {len(steps)}\n")
								tf.write(f"  Speech frame duration: 80ms (per frame), chunk = {self.num_frames_per_chunk} frames = {self.num_frames_per_chunk * 80}ms\n")
								tf.write(f"  Avg step wall time: {avg_ms:.2f}ms\n")
								tf.write(f"  Min step wall time: {min(durations):.2f}ms\n")
								tf.write(f"  Max step wall time: {max(durations):.2f}ms\n")
								tf.write(f"  Realtime ratio: {avg_ms / (self.num_frames_per_chunk * 80):.3f}x (< 1.0 = faster than realtime)\n\n")

							# Key events timeline
							events = fc_timing.get("events", [])
							if events:
								tf.write(f"--- Event Timeline ---\n")
								for ev in events:
									evt_name = ev["event"]
									wall_rel = ev.get("wall_time", 0) - t0
									sim_t = ev.get("simulated_time_sec")
									frame = ev.get("frame_idx")
									line = f"  [{wall_rel:8.3f}s wall]"
									if sim_t is not None:
										line += f"  [{sim_t:7.2f}s sim]"
									if frame is not None:
										line += f"  [frame {frame:4d}]"
									line += f"  {evt_name}"

									if "async_steps" in ev:
										line += f" — {ev['async_steps']} tokens in {ev['async_wall_sec']:.3f}s"
										line += f" ({ev['async_tokens_per_sec']:.1f} tok/s)"
									if "api_latency_ms" in ev:
										line += f" — API latency: {ev['api_latency_ms']:.0f}ms"
									if "response_tokens" in ev:
										line += f" — {ev['response_tokens']} tokens in {ev['response_wall_sec']:.3f}s"
										line += f" ({ev['response_tokens_per_sec']:.1f} tok/s)"
									if "total_fc_wall_sec" in ev:
										line += f" — total FC: {ev['total_fc_wall_sec']:.3f}s"
									tf.write(line + "\n")
								tf.write("\n")

							# Summary: async text token speed vs speech frame rate
							eotc_events = [e for e in events if "async_tokens_per_sec" in e]
							if eotc_events:
								tf.write(f"--- Async FC Speed Summary ---\n")
								tf.write(f"  Speech frame rate: {1/0.08:.1f} tok/s (one frame every 80ms)\n")
								for i, ev in enumerate(eotc_events):
									speedup = ev["async_tokens_per_sec"] / (1/0.08)
									tf.write(f"  FC call #{i}: {ev['async_tokens_per_sec']:.1f} tok/s")
									tf.write(f" ({speedup:.1f}x faster than speech)\n")
								tf.write("\n")

							# Channel timeline visualization from raw token tensors
							if token_data is not None:
								gen_t, gen_asr, n_frames, gen_fc = token_data
								tokenizer = self.s2s_model.tokenizer
								pad_id = self.s2s_model.model.stt_model.text_pad_id
								bos_id = getattr(self.s2s_model.model.stt_model, "text_bos_id", None)
								eos_id = getattr(self.s2s_model.model.stt_model, "text_eos_id", None)
								sotc_id = getattr(self.s2s_model, "_fc_sotc_id", None)
								eotc_id = getattr(self.s2s_model, "_fc_eotc_id", None)

								skip_ids = {pad_id}
								if bos_id is not None:
									skip_ids.add(bos_id)
								if eos_id is not None:
									skip_ids.add(eos_id)

								def _get_spans(tensor_1d, skip):
									"""Return list of (start_frame, end_frame, tokens_text) for non-skip runs."""
									spans = []
									ids = tensor_1d.tolist()
									i = 0
									while i < len(ids):
										if ids[i] not in skip:
											start = i
											run_ids = []
											while i < len(ids) and ids[i] not in skip:
												run_ids.append(ids[i])
												i += 1
											text = tokenizer.ids_to_text(run_ids) if run_ids else ""
											spans.append((start, i - 1, text.strip()))
										else:
											i += 1
									return spans

								agent_spans = _get_spans(gen_t[0, :n_frames], skip_ids)
								asr_spans = _get_spans(gen_asr[0, :n_frames], skip_ids)
								fc_spans = _get_spans(gen_fc[0, :n_frames], skip_ids) if gen_fc is not None else []

								# Build FC mapping: (sotc_frame, async_steps, wall_sec)
								# for converting simulated frame → actual audio time.
								sotc_evts = [e for e in events if e["event"] == "tool_call_start (SOTC)"]
								eotc_evts = [e for e in events if "async_steps" in e]
								compl_evts = [e for e in events if e["event"] == "fc_async_complete" and "total_fc_wall_sec" in e]
								fc_map = []
								for s_ev, e_ev, c_ev in zip(sotc_evts, eotc_evts, compl_evts):
									fc_map.append((s_ev.get("frame_idx", 0), e_ev["async_steps"], c_ev["total_fc_wall_sec"]))

								def _sim_to_audio(frame):
									"""Convert simulated frame index to actual audio time (seconds)."""
									cum_offset = 0.0
									for sotc_f, asteps, wsec in fc_map:
										if frame <= sotc_f:
											break
										if frame < sotc_f + asteps:
											frac = (frame - sotc_f) / asteps
											return sotc_f * 0.08 + cum_offset + frac * wsec
										cum_offset += wsec - asteps * 0.08
									return frame * 0.08 + cum_offset

								# Compute total audio duration
								audio_total_sec = _sim_to_audio(n_frames)

								tf.write(f"--- Channel Timeline (actual audio time) ---\n")
								tf.write(f"  Total frames: {n_frames} (audio duration: {audio_total_sec:.2f}s)\n\n")

								# Agent text channel
								tf.write(f"  [Agent Text]  ({len(agent_spans)} active spans)\n")
								for s_start, s_end, text in agent_spans:
									a_start = _sim_to_audio(s_start)
									a_end = _sim_to_audio(s_end)
									preview = text[:80] + ("..." if len(text) > 80 else "")
									tf.write(f"    {a_start:7.2f}s – {a_end:7.2f}s  (frames {s_start}-{s_end})  \"{preview}\"\n")
								tf.write("\n")

								# User Speech channel: map user speech segments
								# from original audio time to stereo time,
								# accounting for FC silence gaps.
								raw_user_segs = fc_timing.get("user_segments") or []
								audio_dur_sec = fc_timing.get("audio_duration_sec")

								def _orig_to_stereo(orig_t):
									"""Map a time in the original audio to stereo time.
									FC silence is inserted at sotc_f*0.08 in original,
									shifting everything at or after that point forward."""
									offset = 0.0
									for sotc_f, asteps, wsec in fc_map:
										fc_orig = sotc_f * 0.08
										if orig_t < fc_orig:
											break
										offset += wsec
									return orig_t + offset

								user_speech_segs = []
								for seg in raw_user_segs:
									orig_s = seg.get("start", 0.0)
									orig_e = orig_s + seg.get("duration", 0.0)
									stereo_s = _orig_to_stereo(orig_s)
									stereo_e = _orig_to_stereo(orig_e)
									user_speech_segs.append((stereo_s, stereo_e, orig_s, orig_e, seg.get("text", "")))

								rnnt_text = getattr(state, "final_rnnt_text", "") or ""

								tf.write(f"  [User Speech]  ({len(user_speech_segs)} segments from manifest")
								if fc_map:
									parts = ", ".join(f"+{wsec:.2f}s FC gap at orig {sf*0.08:.2f}s" for sf, _, wsec in fc_map)
									tf.write(f", {parts}")
								tf.write(")\n")
								for stereo_s, stereo_e, orig_s, orig_e, text in user_speech_segs:
									preview = text[:80] + ("..." if len(text) > 80 else "")
									tf.write(f"    {stereo_s:7.2f}s – {stereo_e:7.2f}s  (original {orig_s:.2f}s–{orig_e:.2f}s)  \"{preview}\"\n")
								if rnnt_text:
									tf.write(f"    RNNT transcript: \"{rnnt_text}\"\n")
								tf.write("\n")

								# Function call channel
								tf.write(f"  [Function Call] ({len(fc_spans)} active spans)\n")
								for s_start, s_end, text in fc_spans:
									a_start = _sim_to_audio(s_start)
									a_end = _sim_to_audio(s_end)
									preview = text[:120] + ("..." if len(text) > 120 else "")
									tf.write(f"    {a_start:7.2f}s – {a_end:7.2f}s  (frames {s_start}-{s_end})  \"{preview}\"\n")
								tf.write("\n")

								# ASCII timeline bar chart in audio time
								# Each char = 0.08s of actual audio time
								sec_per_char = 0.08
								bar_len = int(audio_total_sec / sec_per_char) + 1

								def _frame_to_bar_idx(frame):
									return int(_sim_to_audio(frame) / sec_per_char)

								def _make_bar(spans, length):
									bar = ['.'] * length
									for s_start, s_end, _ in spans:
										for idx in range(_frame_to_bar_idx(s_start), min(_frame_to_bar_idx(s_end) + 1, length)):
											bar[idx] = '#'
									return ''.join(bar)

								def _make_user_speech_bar(length, segments):
									bar = ['.'] * length
									for seg in segments:
										seg_s, seg_e = seg[0], seg[1]
										for idx in range(int(seg_s / sec_per_char), min(int(seg_e / sec_per_char) + 1, length)):
											bar[idx] = '#'
									return ''.join(bar)

								def _make_fc_bar(length, fc_tensor, total, sotc_id_val, eotc_id_val):
									bar = ['.'] * length
									if fc_tensor is not None:
										ids = fc_tensor[0, :total].tolist()
										in_response = False
										for f, tok_id in enumerate(ids):
											idx = _frame_to_bar_idx(f)
											if idx < length:
												if tok_id == sotc_id_val:
													bar[idx] = 'S'
													in_response = False
												elif tok_id == eotc_id_val:
													bar[idx] = 'E'
													in_response = True
												elif in_response and tok_id not in skip_ids:
													bar[idx] = 'R'
												elif tok_id not in skip_ids:
													bar[idx] = '#'
									return ''.join(bar)

								agent_bar = _make_bar(agent_spans, bar_len)
								user_bar = _make_user_speech_bar(bar_len, user_speech_segs)
								fc_bar = _make_fc_bar(bar_len, gen_fc, n_frames, sotc_id, eotc_id)

								tf.write(f"  --- ASCII Timeline (each char = {sec_per_char:.2f}s audio time, '.' = silence/pad, '#' = active, S = SOTC, E = EOTC, R = response) ---\n")
								tick_interval = int(5.0 / sec_per_char)
								tick_line = [' '] * bar_len
								label_line = [' '] * bar_len
								for tick_pos in range(0, bar_len, tick_interval):
									tick_line[tick_pos] = '|'
									t_sec = f"{tick_pos * sec_per_char:.0f}s"
									for ci, ch in enumerate(t_sec):
										if tick_pos + ci < bar_len:
											label_line[tick_pos + ci] = ch
								tf.write(f"  Time:         {''.join(label_line)}\n")
								tf.write(f"                {''.join(tick_line)}\n")
								tf.write(f"  Agent Text:   {agent_bar}\n")
								tf.write(f"  User Speech:  {user_bar}\n")
								tf.write(f"  FC Channel:   {fc_bar}\n")
								tf.write("\n")

						logging.info(f"Saved FC timing report: {timing_path}")
					except Exception as e:
						logging.warning(f"Failed to save FC timing: {e}")

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
		rnnt_asr_texts = []
		fc_texts = []
		raw_fc_texts = []

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
			rnnt_asr_texts.append(getattr(state, "final_rnnt_text", None) or "")

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
					fc_texts.append(fc_text)
					raw_fc_texts.append(fc_text_raw)
				else:
					fc_texts.append("")
					raw_fc_texts.append("")
			else:
				texts_with_timestamps.append("")
				asr_texts_with_timestamps.append("")
				raw_texts.append("")
				raw_asr_texts.append("")
				fc_texts.append("")
				raw_fc_texts.append("")

		self.close_session()

		return PipelineOutput(
			texts=texts,
			words=words,
			asr_texts=asr_texts,
			texts_with_timestamps=texts_with_timestamps,
			asr_texts_with_timestamps=asr_texts_with_timestamps,
			raw_texts=raw_texts,
			raw_asr_texts=raw_asr_texts,
			rnnt_asr_texts=rnnt_asr_texts,
			fc_texts=fc_texts,
			raw_fc_texts=raw_fc_texts,
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
