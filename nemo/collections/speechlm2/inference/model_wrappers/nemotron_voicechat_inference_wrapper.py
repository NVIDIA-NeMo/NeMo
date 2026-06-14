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
import yaml
from omegaconf import OmegaConf, DictConfig
import numpy as np
import librosa
import time
import threading
import queue
from transformers import DynamicCache
import json
import re
import os
import sys
import argparse
import math
import torchaudio
import functools
from dataclasses import dataclass
from typing import Optional, Tuple
from nemo.utils import logging
from jiwer import wer

import gc
import types


# Set environment variables (use existing env vars if set, otherwise use defaults)
_default_cache = "/tmp/cache"
os.environ.setdefault("HF_HOME", _default_cache)
os.environ.setdefault("TORCH_HOME", _default_cache)
os.environ.setdefault("NEMO_CACHE_DIR", _default_cache)
os.environ.setdefault("NEMO_NLP_TMP", os.path.join(_default_cache, "nemo_nlp_tmp"))

from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat

from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.fusion import create_fusion_module
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model
from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
    PerceptionCacheState,
    PerceptionCacheManager,
)
from nemo.collections.speechlm2.inference.utils.pipeline_utils import clean_pred_text


def tokens_to_str_raw(tokens: torch.Tensor, lengths: torch.Tensor, tokenizer, pad_id: int) -> list:
    """
    Convert token IDs to text strings, preserving ALL special tokens including <SPECIAL_12> (pad token).

    Unlike tokens_to_str, this function uses ids_to_tokens which preserves special tokens,
    and does NOT filter out any tokens (including pad tokens like <SPECIAL_12>).

    Args:
        tokens: Token IDs tensor (B, T)
        lengths: Length of each sequence (B,)
        tokenizer: Tokenizer for decoding
        pad_id: Pad token ID (not used for filtering in raw mode, kept for API compatibility)

    Returns:
        List of decoded text strings with ALL special tokens preserved (including <SPECIAL_12>)
    """
    ans = []
    for hyp_ids, hyp_len in zip(tokens.cpu(), lengths.cpu()):
        hyp_ids = hyp_ids[:hyp_len]
        # Do NOT filter out any tokens - keep everything including pad tokens (<SPECIAL_12>)
        hyp_ids_list = hyp_ids.tolist()

        # Use ids_to_tokens which preserves special tokens like <SPECIAL_12>
        toks = tokenizer.ids_to_tokens(hyp_ids_list)

        # Only replace 'Ġ' with space for proper word boundaries, keep all special tokens
        toks = [tok.replace('Ġ', ' ') for tok in toks]

        ans.append("".join(toks))
    return ans



# --- Configuration ---
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples

TTS_SAMPLE_RATE = 22050


# Default hyper-parameters that can be overridden via `model_cfg`
DEFAULT_BUFFER_SIZE_FRAMES = 71
DEFAULT_NUM_FRAMES_PER_CHUNK = 1
# Only used when use_codec_cache=False (sliding-window fallback).
# Ignored when the codec streaming cache is enabled.
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 600


class NemotronVoicechatInferenceWrapper:
    """
    Inference wrapper for NemotronVoiceChat models.
    Uses a sliding window buffer and processes audio frame by frame.
    """

    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for realtime streaming inference.

        Args:
            model_cfg (DictConfig): Configuration describing the model paths and runtime parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)


        logging.info(f"pythonpath: {sys.path}")


        logging.info(f"before setting - torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
        logging.info(f"before setting - torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
        logging.info(f"before setting - torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

        self._deterministic = bool(model_cfg.get("deterministic", False))
        if self._deterministic:
            engine_type = model_cfg.get("engine_type", "native")
            if "vllm" in engine_type.lower():
                raise ValueError(
                    "`deterministic` is not compatible with vLLM engines because vLLM uses custom "
                    "CUDA kernels (PagedAttention, FlashAttention) that do not support deterministic mode. "
                    f"Got engine_type='{engine_type}'. Use engine_type='native' for deterministic inference."
                )

            # Required by torch.use_deterministic_algorithms for cuBLAS reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.use_deterministic_algorithms(True, warn_only=False)

            logging.info("Deterministic mode ENABLED")
            logging.info(f"  CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
            logging.info(f"  flash_sdp enabled: {torch.backends.cuda.flash_sdp_enabled()}")
            logging.info(f"  mem_efficient_sdp enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
            logging.info(
                "  NOTE: deterministic mode uses different CUDA kernels (e.g. math SDPA instead of "
                "FlashAttention), so results may differ slightly from non-deterministic mode. "
                "Inference will also be slower."
            )

        logging.info(f"torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")
        logging.info(f"torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
        logging.info(f"torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")

        self.model_cfg = model_cfg

        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("`model_cfg.model_path` must be provided.")

        self.llm_checkpoint_path = model_cfg.get("llm_checkpoint_path")
        if not self.llm_checkpoint_path:
            raise ValueError("`model_cfg.llm_checkpoint_path` must be provided.")

        self.decode_audio = bool(model_cfg.get("decode_audio", True))
        # Number of past codec tokens kept in the sliding-window decode buffer.
        # Only used when use_codec_cache=False (the fallback path). When the
        # codec cache is enabled, context is maintained incrementally inside
        # CausalConv1dCache and this value is ignored.
        self.codec_token_history_size = int(
            model_cfg.get("codec_token_history_size", DEFAULT_CODEC_TOKEN_HISTORY_SIZE)
        )

        self.speaker_reference = model_cfg.get("speaker_reference")
        if self.decode_audio and not self.speaker_reference:
            raise ValueError("`model_cfg.speaker_reference` must be provided when decode_audio is enabled.")

        self.tts_system_prompt = model_cfg.get("tts_system_prompt", None)
        logging.info(f"TTS system prompt: {self.tts_system_prompt}")

        compute_dtype = model_cfg.get("compute_dtype", "bfloat16")
        self.dtype = self._resolve_dtype(compute_dtype)

        self.device = self._resolve_device(
            device=model_cfg.get("device"),
            device_id=model_cfg.get("device_id"),
        )

        logging.info("=" * 70)
        logging.info("INITIALIZING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"Decode audio: {self.decode_audio}")
        logging.info(f"Engine type: {model_cfg.get('engine_type', 'native')}")
        logging.info(f"Sampling - top_p: {model_cfg.get('top_p', 1.0)}, repetition_penalty: {model_cfg.get('repetition_penalty', 1.0)}, temperature: {model_cfg.get('temperature', 1.0)}")
        logging.info("=" * 70)

        # Cached TTS helpers populated during initialization/warmup
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None


        self.model = None
        self.model_llm_interface = None
        self.tokenizer = None

        # Max-response watchdog defaults — set properly after model loads.
        # Must be defined early because _apply_rnnt_turn_taking is called during warmup.
        self._max_agent_response_frames: int = 0
        self._redirect_tokens_queue: list = []
        self._max_response_redirect_tokens: list = []
        # After tool-response injection completes, exempt the next agent BOS from
        # self-play suppression so the post-TC verbal response is not silenced.
        self._post_tc_bos_exempt: bool = False

        # vLLM configuration
        self.engine_type = model_cfg.get("engine_type", "native")
        self.use_vllm_llm = "vllm_llm" in self.engine_type.lower()
        self.use_vllm_eartts = "vllm_eartts" in self.engine_type.lower()
        self.vllm_llm_config = model_cfg.get("vllm_llm_config", None)
        self.vllm_tts_config = model_cfg.get("vllm_tts_config", None)
        self.request_id = "streaming_request_0"  # For vLLM streaming

        # Sampling parameters
        self.top_p = float(model_cfg.get("top_p", 1.0))
        self.repetition_penalty = float(model_cfg.get("repetition_penalty", 1.0))
        self.temperature = float(model_cfg.get("temperature", 1.0))

        # Codec streaming cache: decode only new tokens each step using the
        # codec's CausalConv1dCache, which maintains ConvNeXt and ISTFT state
        # across calls for sample-continuous audio. When enabled, the
        # codec_token_history_size parameter and audio_toks_buffer are unused.
        # When disabled, falls back to the sliding-window decode that re-decodes
        # codec_token_history_size tokens each step and extracts the tail.
        self.use_codec_cache = bool(model_cfg.get("use_codec_cache", True))
        if self.use_codec_cache and self.decode_audio:
            configured_history = model_cfg.get("codec_token_history_size", None)
            if configured_history is not None:
                logging.info(
                    f"use_codec_cache is enabled — codec_token_history_size ({configured_history}) "
                    f"will be ignored (context is maintained incrementally by the codec cache)."
                )

        # RNNT loading strategy:
        # use_separate_rnnt_ckpt=false (default) → prefer RNNT from combined checkpoint
        # use_separate_rnnt_ckpt=true → always load RNNT from the separate .nemo file
        self.use_separate_rnnt_ckpt = bool(model_cfg.get("use_separate_rnnt_ckpt", False))

        # Perception cache configuration
        self.use_perception_cache = bool(model_cfg.get("use_perception_cache", False))
        use_perception_cudagraph = bool(model_cfg.get("use_perception_cudagraph", False))
        if use_perception_cudagraph and not self.use_perception_cache:
            raise ValueError(
                "use_perception_cudagraph requires use_perception_cache to be enabled. "
                "Please also set use_perception_cache=True."
            )
        self.perception_cache_mgr: Optional[PerceptionCacheManager] = None
        self._use_perception_cudagraph = use_perception_cudagraph
        self._pad_and_drop_preencoded = bool(model_cfg.get("pad_and_drop_preencoded", False))

        self._initialize_model()

        logging.info("NemotronVoicechatInferenceWrapper initialized successfully.")

        logging.info(f"{self.model.stt_model.perception.encoder._cfg = }")
        logging.info(f"{self.model.stt_model.perception.encoder.streaming_cfg = }")

    @staticmethod
    def _resolve_dtype(compute_dtype):
        if isinstance(compute_dtype, torch.dtype):
            return compute_dtype
        if compute_dtype is None:
            return torch.bfloat16
        if isinstance(compute_dtype, str):
            key = compute_dtype.lower()
            mapping = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "full": torch.float32,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

    @staticmethod
    def _resolve_device(device=None, device_id=None):
        if isinstance(device, torch.device):
            resolved_device = device
        else:
            if device is None:
                resolved_device = DEFAULT_DEVICE
            else:
                device_str = str(device)
                base = device_str
                if device_id is not None and device_str.startswith("cuda") and ":" not in device_str:
                    base = f"{device_str}:{device_id}"
                resolved_device = torch.device(base)
        return resolved_device

    def _samples_per_audio_output_frame(self):
        rate = getattr(self, "target_sample_rate", None)
        if rate is None:
            cfg_rate = None
            try:
                cfg_rate = self.model_cfg.get("tts_sample_rate", None)
            except Exception:
                cfg_rate = None
            if cfg_rate is None:
                try:
                    cfg_rate = self.model_cfg.get("output_sample_rate", None)
                except Exception:
                    cfg_rate = None
            if cfg_rate is not None:
                rate = float(cfg_rate)
        if rate is None:
            rate = TTS_SAMPLE_RATE
        samples = int(float(rate) * FRAME_SIZE_SEC)
        return samples

    def _load_and_merge_configs(self):
        """Load and merge configurations from both nano and eartts checkpoints."""
        logging.info("Loading and merging configurations...")

        # Load nano's config (for LLM, perception)
        nano_config_file = os.path.join(self.llm_checkpoint_path, "config.json")
        logging.info(f"  Loading nano config: {nano_config_file}")
        with open(nano_config_file, 'r') as f:
            import json
            nano_cfg_dict = json.load(f)
        nano_cfg = DictConfig(nano_cfg_dict)

        # Load eartts's config (for TTS)
        eartts_config_file = os.path.join(self.model_path, "config.json")
        logging.info(f"  Loading eartts config: {eartts_config_file}")
        with open(eartts_config_file, 'r') as f:
            eartts_cfg_dict = json.load(f)
        eartts_cfg = DictConfig(eartts_cfg_dict)

        # Start with nano's config as base
        merged_cfg = nano_cfg

        # Override TTS-related parts with eartts's config
        logging.info("  Merging: Using nano's config for LLM/perception, eartts's for TTS")
        if 'model' in eartts_cfg and 'speech_generation' in eartts_cfg.model:
            merged_cfg.model.speech_generation = eartts_cfg.model.speech_generation
            logging.info("    TTS config from eartts")

        # Set speaker reference
        if 'model' not in merged_cfg:
            merged_cfg.model = {}
        merged_cfg.model.inference_speaker_reference = self.speaker_reference

        # Ensure data section has correct sample rates
        if 'data' not in merged_cfg:
            merged_cfg.data = eartts_cfg.data

        logging.info(f"  Final config:")
        logging.info(f"    - pretrained_llm: {merged_cfg.model.stt.model.pretrained_llm}")
        logging.info(f"    - perception.d_model: {merged_cfg.model.stt.model.perception.modality_adapter.d_model}")
        logging.info(f"    - speech_generation: {'present' if 'speech_generation' in merged_cfg.model else 'missing'}")

        return merged_cfg

    def _initialize_model(self):
        """Initialize the NemotronVoiceChat with hybrid loading."""
        from safetensors.torch import load_file
        from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init

        logging.info("Initializing model with hybrid loading strategy...")

        # Step 1: Load and merge configs
        cfg = self._load_and_merge_configs()

        # Step 2: DO NOT set pretrained_s2s_model - we'll load weights manually
        cfg.model.stt.model.pretrained_s2s_model = None
        cfg.model.speech_generation.model.pretrained_model = None

        # Determine RNNT loading strategy:
        # (a) Check if the combined checkpoint contains RNNT keys and config
        # (b) Decide whether to use combined ckpt or separate .nemo for RNNT
        rnnt_asr_path = self.model_cfg.get("pretrained_rnnt_asr", None)
        rnnt_merge_info = None
        ckpt_has_rnnt = False

        safetensors_path = os.path.join(self.llm_checkpoint_path, "model.safetensors")
        config_json_path = os.path.join(self.llm_checkpoint_path, "config.json")

        if os.path.isfile(config_json_path):
            import json as _json
            with open(config_json_path, "r") as f:
                ckpt_config = _json.load(f)
            rnnt_merge_info = ckpt_config.get("_rnnt_merge_info", None)
            if rnnt_merge_info and rnnt_merge_info.get("decoder_config") and rnnt_merge_info.get("joint_config"):
                ckpt_has_rnnt = True
                logging.info("Combined checkpoint contains RNNT decoder/joint config.")

        use_rnnt_from_combined = ckpt_has_rnnt and not self.use_separate_rnnt_ckpt

        if use_rnnt_from_combined:
            logging.info("RNNT will be loaded from combined checkpoint (embedded weights + saved config).")
            if rnnt_asr_path:
                logging.info("  pretrained_rnnt_asr=%s is set but will be ignored (use_separate_rnnt_ckpt=false).", rnnt_asr_path)
        elif rnnt_asr_path:
            logging.info("RNNT will be loaded from separate .nemo checkpoint: %s", rnnt_asr_path)
            OmegaConf.set_struct(cfg.model.stt.model, False)
            cfg.model.stt.model.pretrained_rnnt_asr = rnnt_asr_path
            OmegaConf.set_struct(cfg.model.stt.model, True)
        else:
            logging.info("No RNNT source configured. RNNT decoding will be disabled.")

        # If loading RNNT from combined ckpt, prevent DuplexSTTModel.__init__
        # from loading the .nemo file (the module structure will be set up after init).
        if use_rnnt_from_combined:
            OmegaConf.set_struct(cfg.model.stt.model, False)
            cfg.model.stt.model.pretrained_rnnt_asr = None
            OmegaConf.set_struct(cfg.model.stt.model, True)

        # Convert to dict for model initialization
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Step 3: Initialize model structure
        logging.info("Initializing model structure...")
        start_DuplexS2S_init = time.time()
        self.model = NemotronVoiceChat(cfg_dict)
        logging.info(f"Time taken to initialize NemotronVoiceChat: {time.time() - start_DuplexS2S_init} seconds")
        logging.info("  Model structure initialized")

        # =====================================================================
        # TTS PAD-SILENCE SUBSTITUTION  (Option B — per-stream state)
        # =====================================================================
        # Bug fixed:
        #   After a function-calling cycle ends, the LLM sometimes fails to
        #   emit BOS on the agent text channel, leading to a long stretch of
        #   PAD frames. During this gap the TTS codec sometimes hallucinates
        #   "garbled syllables" / phantom speech instead of clean silence
        #   (codec_cache contamination from prior ack#2 / Phase 2 activity).
        #
        # Fix:
        #   At three TTS call sites in this wrapper, substitute the existing
        #   `codec_silence_tokens` for the codec tokens TTS produces, whenever:
        #     (a) text input was PAD, AND
        #     (b) the agent is currently IDLE — see condition below.
        #
        # Why condition (b) matters:
        #   PAD on the agent text channel does NOT always mean "silence in
        #   audio". With Branch-B training (this checkpoint), the LLM emits
        #   dense content tokens then a long PAD trail until EOS at the
        #   audio-end frame. The TTS keeps rendering audio during that PAD
        #   trail. Silencing those in-turn PADs would chop the audio. So we
        #   only substitute when the agent is genuinely IDLE — no turn open
        #   (session just started, or last non-PAD token was EOS).
        #
        # How the "agent_idle" flag is tracked:
        #   - Per-stream:   S2SStreamingState.agent_idle (Option B, preferred)
        #   - Per-wrapper:  self._agent_idle (fallback for older code paths)
        #   - Helpers:      _get_agent_idle(stream_id), _set_agent_idle(...)
        #                   _mark_agent_idle(stream_id)
        #   - NOT touched by S2SStreamingState.cleanup_after_response() —
        #     that method runs every Triton step, NOT just at session end.
        #     Flipping the flag there caused in-turn silencing and audio cut-off
        #     (regression discovered & fixed during the multi-stream refactor).
        #
        # Env var:
        #   S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD=true → enables the feature.
        #   When unset/false: substitution never fires, behaves like baseline.
        #
        # Mirrors the existing `inference_force_speech_silence_on_eos` block in
        # `infer_one_step` (same mechanism, different trigger token — but EOS
        # only fires for ONE frame per turn, so it doesn't trigger the long-run
        # corruption risk this PAD version has).
        # =====================================================================
        _pad_silence_env = os.environ.get('S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD', '').lower()
        if _pad_silence_env in ('true', '1', 'yes'):
            try:
                self.model.cfg['inference_force_speech_silence_on_pad'] = True
                logging.info(
                    "[Env] S2S_INFERENCE_FORCE_SPEECH_SILENCE_ON_PAD=true → "
                    "model.cfg.inference_force_speech_silence_on_pad=True"
                )
            except Exception as _e:
                logging.warning(
                    f"[Env] Could not set inference_force_speech_silence_on_pad on model.cfg: {_e}"
                )
        # Disable RNNT self-play suppression — lets LLM-native BOS through so the
        # agent can self-initiate a turn even if the user hasn't spoken.
        self.model_cfg["rnnt_self_play_suppression"] = False
        self._rnnt_max_symbols = int(self.model_cfg.get("rnnt_max_symbols", 10))

        # Initialize "agent idle" state. True at session start (no agent turn yet).
        # Updated in infer_one_step: BOS → False (turn open), EOS → True (agent idle).
        #
        # This is the SINGLE-STREAM fallback. For proper multi-stream operation,
        # the pipeline injects a per-stream lookup callable via:
        #   self._set_streaming_state_getter(self.get_or_create_state)
        # When set, the wrapper reads/writes streaming_state.agent_idle
        # (per-stream) instead of self._agent_idle (per-wrapper).
        self._agent_idle = True
        self._streaming_state_getter = None   # set by pipeline if multi-stream support is needed

        # If using combined checkpoint for RNNT, set up modules from saved config
        if use_rnnt_from_combined:
            from nemo.collections.speechlm2.parts.pretrained import setup_rnnt_from_combined_checkpoint
            tokenizer_dir = os.path.join(self.llm_checkpoint_path, "rnnt_tokenizer")
            if not os.path.isdir(tokenizer_dir):
                tokenizer_dir = None
            setup_rnnt_from_combined_checkpoint(
                self.model.stt_model, rnnt_merge_info, tokenizer_dir=tokenizer_dir,
            )

        # Step 4: Load nano's checkpoint (LLM + perception)
        if self.llm_checkpoint_path is not None:
            logging.info("Loading LLM + perception:")
            logging.info(f"  Path: {self.llm_checkpoint_path}")

            nano_state_dict = load_file(os.path.join(self.llm_checkpoint_path, "model.safetensors"))

            # Filter to non-TTS weights
            tts_keys = ['tts_model.', 'speech_generation.']

            # If using vLLM for LLM, also exclude LLM weights to save memory
            # vLLM will load its own copy of the LLM
            if self.use_vllm_llm:
                llm_keys = ['stt_model.llm.']
                exclude_keys = tts_keys + llm_keys
                logging.info(f"  Using vLLM - excluding LLM weights from nano checkpoint")
            else:
                exclude_keys = tts_keys

            nano_filtered = {k: v for k, v in nano_state_dict.items()
                           if not any(k.startswith(prefix) for prefix in exclude_keys)}

            logging.info(f"  Loading {len(nano_filtered)} parameters (excluded: {exclude_keys})...")

            # Free the full state dict immediately to save CPU memory
            del nano_state_dict
            gc.collect()

            nano_filtered = set_model_dict_for_partial_init(nano_filtered, self.model.state_dict())
            missing, unexpected = self.model.load_state_dict(nano_filtered, strict=False)

            # Free filtered dict
            del nano_filtered
            gc.collect()

            missing_non_excluded = [k for k in missing if not any(k.startswith(prefix) for prefix in exclude_keys)]
            unexpected_non_excluded = [k for k in unexpected if not any(k.startswith(prefix) for prefix in exclude_keys)]

            if missing_non_excluded:
                logging.info(f"  {len(missing_non_excluded)} keys missing (might be OK)")
            if unexpected_non_excluded:
                logging.info(f"  {len(unexpected_non_excluded)} unexpected keys")

        # Step 5: Load eartts's checkpoint (TTS only)
        if self.model_path is not None:
            logging.info("Loading TTS checkpoint:")
            logging.info(f"  Path: {self.model_path}")

            eartts_state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))

            # Filter to only TTS weights
            tts_keys_filter = ['tts_model.']
            eartts_tts_only = {k: v for k, v in eartts_state_dict.items()
                                 if any(k.startswith(prefix) for prefix in tts_keys_filter)}

            logging.info(f"  Loading {len(eartts_tts_only)} TTS parameters...")

            start_tts_load_state_dict = time.time()
            missing, unexpected = self.model.load_state_dict(eartts_tts_only, strict=False)
            logging.info(f"Time taken to load TTS state dict: {time.time() - start_tts_load_state_dict} seconds")

            missing_tts = [k for k in missing if any(k.startswith(prefix) for prefix in tts_keys_filter)]
            unexpected_tts = [k for k in unexpected if any(k.startswith(prefix) for prefix in tts_keys_filter)]

            if missing_tts:
                logging.info(f"  {len(missing_tts)} TTS keys missing")
                for mk in missing_tts:
                    logging.info(f"    missing: {mk}")
            if unexpected_tts:
                logging.info(f"  {len(unexpected_tts)} unexpected TTS keys")

            if self.use_vllm_eartts:
                # gonna convert and load vllm eartts engine
                # Use object.__setattr__ to bypass PyTorch's module registration
                # since VllmEARTTSModel is not a torch.nn.Module
                del self.model.tts_model.tts_model
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                object.__setattr__(
                    self.model.tts_model,
                    'tts_model',
                    create_model(
                        model=self.model_path,
                        engine_type="vllm_eartts",
                        vllm_config=self.vllm_tts_config)
                )
                from nemo.collections.speechlm2.inference.vllm.vllm_patch import patched_infer_codes_one_step
                self.model.tts_model.infer_codes_one_step = types.MethodType(patched_infer_codes_one_step, self.model.tts_model)

            logging.info(f"  eartts checkpoint loaded (TTS only)")

        logging.info("\nHybrid loading completed!")

        # If using vLLM for LLM, delete native LLM BEFORE moving to device to save memory
        if self.use_vllm_llm:
            logging.info("\nDeleting native LLM before GPU transfer (will use vLLM instead)...")
            if hasattr(self.model.stt_model, 'llm') and self.model.stt_model.llm is not None:
                # Delete all submodules of LLM to free memory
                for name, child in list(self.model.stt_model.llm.named_children()):
                    delattr(self.model.stt_model.llm, name)
                del self.model.stt_model.llm
                self.model.stt_model.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("  Native LLM deleted")

        # Setup model
        self.model.to(self.device)
        self.model.eval()

        # Convert only the S2S components to the configured dtype, not the TTS model
        logging.info(f"Converting S2S components to {self.dtype} (keeping TTS in float32)...")
        if self.model.stt_model.llm is not None:
            self.model.stt_model.llm = self.model.stt_model.llm.to(self.dtype)
        self.model.stt_model.lm_head = self.model.stt_model.lm_head.to(self.dtype)
        self.model.stt_model.embed_tokens = self.model.stt_model.embed_tokens.to(self.dtype)
        if getattr(self.model.stt_model, "asr_head", None) is not None:
            self.model.stt_model.asr_head = self.model.stt_model.asr_head.to(self.dtype)
            self.model.stt_model.embed_asr_tokens = self.model.stt_model.embed_asr_tokens.to(self.dtype)
        if self.model.stt_model.function_head is not None:
            self.model.stt_model.function_head = self.model.stt_model.function_head.to(self.dtype)
            logging.info("function_head converted to %s", self.dtype)
        #self.model.stt_model.perception = self.model.stt_model.perception.to(self.dtype)
        if getattr(self.model.stt_model, "rnnt_decoder", None) is not None:
            self.model.stt_model.rnnt_decoder = self.model.stt_model.rnnt_decoder.to(self.dtype)
            logging.info("rnnt_decoder converted to %s", self.dtype)
        if getattr(self.model.stt_model, "rnnt_joint", None) is not None:
            self.model.stt_model.rnnt_joint = self.model.stt_model.rnnt_joint.to(self.dtype)
            logging.info("rnnt_joint converted to %s", self.dtype)
        logging.info("S2S components converted, TTS kept in float32")
        logging.info("new update, perception also is kept in float32")

        # commenting this out to avoid error when try vllm tts
        # and anyway - when sticking to "native", saw no difference in output
        # with and without this call
        #self.model.on_train_epoch_start()
        self.tokenizer = self.model.stt_model.tokenizer

        # Pre-tokenize max-response redirect message now that tokenizer is ready.
        if self._max_agent_response_frames > 0:
            _stt_r = self.model.stt_model
            _bos_r = getattr(_stt_r, "text_bos_id", None)
            _eos_r = getattr(_stt_r, "text_eos_id", None)
            _pad_r = getattr(_stt_r, "text_pad_id", None)
            _ids_r = list(self.tokenizer.text_to_ids("How can I help you?"))
            if _bos_r is not None:
                _ids_r = [_bos_r] + _ids_r
            if _pad_r is not None:
                _ids_r += [_pad_r] * 17
            if _eos_r is not None:
                _ids_r += [_eos_r]
            self._max_response_redirect_tokens = _ids_r
            logging.info(
                "[MaxResponse] Enabled: max %.1fs (%d frames), redirect=%d tokens",
                _max_resp_sec, self._max_agent_response_frames, len(_ids_r),
            )

        # allow overrides/additions from the self.model_cfg of nemotron_voicechat_inference_wrapper,
        # into the model cfg that is read from config.json of the model.
        # Specifically, this is so that we can specify inference_pad_boost, ... etc.
        for key in (
            "inference_pad_boost",
            "inference_bos_boost",
            "inference_eos_boost",
            "inference_user_pad_boost",
            "inference_user_bos_boost",
            "inference_user_eos_boost",
        ):
            val = self.model_cfg.get(key, None)
            if val is not None:
                OmegaConf.update(self.model.stt_model.cfg, key, val)

        # Print inference boost values
        logging.info(f"inference_eos_boost: {self.model.stt_model.cfg.get('inference_eos_boost', None)}")
        logging.info(f"inference_bos_boost: {self.model.stt_model.cfg.get('inference_bos_boost', None)}")
        logging.info(f"inference_pad_boost: {self.model.stt_model.cfg.get('inference_pad_boost', None)}")
        logging.info(f"inference_user_pad_boost: {self.model.stt_model.cfg.get('inference_user_pad_boost', None)}")
        logging.info(f"inference_user_bos_boost: {self.model.stt_model.cfg.get('inference_user_bos_boost', None)}")
        logging.info(f"inference_user_eos_boost: {self.model.stt_model.cfg.get('inference_user_eos_boost', None)}")

        # Create fusion module — mirrors training's create_fusion_module() call
        # in duplex_stt_model.py so that each modality is combined with the same
        # weights / method the LLM was trained with.
        stt_cfg = self.model.stt_model.cfg
        has_function_head = self.model.stt_model.function_head is not None
        self._has_asr_head = getattr(self.model.stt_model, "asr_head", None) is not None
        hidden_dim = self.model.stt_model.embed_tokens.weight.shape[1]
        self.fusion_module = create_fusion_module(
            fuse_method=stt_cfg.get("fuse_method", None),
            hidden_dim=hidden_dim,
            agent_text_weight=stt_cfg.get("duplex_text_channel_weight", 1.0),
            user_audio_weight=stt_cfg.get("duplex_user_channel_weight", 1.0),
            user_text_weight=stt_cfg.get("duplex_asr_text_weight", 1.0) if self._has_asr_head else 0.0,
            function_weight=stt_cfg.get("duplex_function_channel_weight", 1.0) if has_function_head else 0.0,
        )
        self.fusion_module = self.fusion_module.to(device=self.device, dtype=self.dtype)
        self.fusion_module.eval()
        logging.info(
            "Fusion module created: type=%s, agent_text_weight=%.3f, user_audio_weight=%.3f, "
            "user_text_weight=%.3f, function_weight=%.3f, fuse_method=%s",
            type(self.fusion_module).__name__,
            stt_cfg.get("duplex_text_channel_weight", 1.0),
            stt_cfg.get("duplex_user_channel_weight", 1.0),
            stt_cfg.get("duplex_asr_text_weight", 1.0),
            stt_cfg.get("duplex_function_channel_weight", 1.0) if has_function_head else 0.0,
            stt_cfg.get("fuse_method", None),
        )

        # Resolve special token IDs for function calling
        self._fc_sotc_id = None  # Start-Of-Tool-Call  <SPECIAL_20>
        self._fc_eotc_id = None  # End-Of-Tool-Call    <SPECIAL_21>
        self._fc_eotr_id = None  # End-Of-Tool-Response <SPECIAL_22>
        self._fc_toolresp_open_ids = None   # <TOOLRESPONSE>
        self._fc_toolresp_close_ids = None  # </TOOLRESPONSE>
        if has_function_head:
            try:
                self._fc_sotc_id = self.tokenizer.text_to_ids("<SPECIAL_20>")[0]
                self._fc_eotc_id = self.tokenizer.text_to_ids("<SPECIAL_21>")[0]
                self._fc_toolresp_open_ids = self.tokenizer.text_to_ids("<TOOLRESPONSE>")
                self._fc_toolresp_close_ids = self.tokenizer.text_to_ids("</TOOLRESPONSE>")
                try:
                    self._fc_eotr_id = self.tokenizer.text_to_ids("<SPECIAL_22>")[0]
                except (IndexError, Exception):
                    logging.info("FC: <SPECIAL_22> (EOTR) not in tokenizer, will use step limit for post-injection")
                logging.info(
                    "FC token IDs resolved: SOTC=%s, EOTC=%s, EOTR=%s, TOOLRESPONSE_OPEN=%s, TOOLRESPONSE_CLOSE=%s",
                    self._fc_sotc_id, self._fc_eotc_id, self._fc_eotr_id,
                    self._fc_toolresp_open_ids, self._fc_toolresp_close_ids,
                )
            except Exception as e:
                logging.warning(f"Could not resolve FC special token IDs: {e}. FC injection disabled.")

        # Wrap model with appropriate interface (Native or vLLM)
        if self.use_vllm_llm:
            logging.info("\nWrapping model with VllmLLMModel interface...")
            if self.vllm_llm_config is None:
                raise ValueError("vllm_llm_config must be provided when engine_type contains'vllm_llm'")

            # LLM already deleted above, just ensure cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Set logit boosts as env vars BEFORE creating the vLLM engine,
            # so they are inherited by the forked worker process.  The modified
            # nemotron_h.py reads VLLM_ASR_BOOST_<token_id> and
            # VLLM_TEXT_BOOST_<token_id> in __init__.
            stt = self.model.stt_model
            asr_boost_map = {
                "inference_user_pad_boost": stt.text_pad_id,
                "inference_user_bos_boost": stt.user_bos_id,
                "inference_user_eos_boost": stt.text_eos_id,
            }
            for cfg_key, token_id in asr_boost_map.items():
                val = self.model_cfg.get(cfg_key, None)
                if val is not None and float(val) != 0.0:
                    env_key = f"VLLM_ASR_BOOST_{token_id}"
                    os.environ[env_key] = str(float(val))
                    logging.info(f"Set env {env_key}={val} (from {cfg_key})")

            text_boost_map = {
                "inference_pad_boost": stt.text_pad_id,
                "inference_bos_boost": stt.text_bos_id,
                "inference_eos_boost": stt.text_eos_id,
            }
            for cfg_key, token_id in text_boost_map.items():
                val = self.model_cfg.get(cfg_key, None)
                if val is not None and float(val) != 0.0:
                    env_key = f"VLLM_TEXT_BOOST_{token_id}"
                    os.environ[env_key] = str(float(val))
                    logging.info(f"Set env {env_key}={val} (from {cfg_key})")

            self.model_llm_interface = create_model(
                model=self.model_path,
                engine_type="vllm_llm",
                vllm_config=self.vllm_llm_config,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                text_pad_id=stt.text_pad_id,
            )

            logging.info("VllmLLMModel interface created")
            # vLLM LLM init (device_id=1) changes the CUDA default device in the main process;
            # re-pin stt_model to cuda:0 so CUDAGraph capture uses the correct device.
            self.model.stt_model.to(self.device)
            logging.info(f"stt_model re-pinned to {self.device} after vLLM LLM init")
        else:
            logging.info("\nWrapping model with NativeModel interface...")
            self.model_llm_interface = create_model(
                model=self.model,
                engine_type="native",
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
            )
            logging.info("NativeModel interface created")

        # Get TTS info
        if hasattr(self.model, 'tts_model'):
            self.target_fps = self.model.tts_model.target_fps
            self.target_sample_rate = self.model.tts_model.target_sample_rate
            logging.info(f"\nTTS model initialized: target_fps={self.target_fps}, sample_rate={self.target_sample_rate}")
            if self.decode_audio:
                self._prepare_tts_initial_state()
        else:
            logging.warning("Warning: TTS model not found in the model")

        # Create silence embedding cache for async FC mode
        self._silence_embedding_cache = None
        self._last_user_audio_emb = None  # most recent real audio embedding for async fallback
        self._fc_async_enabled = bool(self.model_cfg.get("fc_async_enabled", False))
        self._fc_async_two_phase = bool(self.model_cfg.get("fc_async_two_phase", False))
        self._fc_async_api_latency_sec = float(self.model_cfg.get("fc_async_api_latency_sec", 0.0))
        # When False (default), async FC always feeds silence to the LLM — no real or cached audio.
        # When True, real audio frames are used if available, falling back to last cached audio.
        self._fc_async_use_real_audio = bool(self.model_cfg.get("fc_async_use_real_audio", False))
        self._fc_convert_num_to_text = bool(self.model_cfg.get("fc_convert_num_to_text", False))
        if self._fc_convert_num_to_text:
            logging.info("[FC] Number-to-text conversion enabled for tool responses")
        # SOTC boost: add this value to SOTC logit when text head predicts <s> with P > 0.2
        self._fc_sotc_boost = float(self.model_cfg.get("fc_sotc_boost", 0.0))
        logging.info(
            "[FC Config] fc_async_enabled=%s, fc_async_two_phase=%s, fc_async_use_real_audio=%s, has_function_head=%s, SOTC_id=%s, sotc_boost=%.1f",
            self._fc_async_enabled, self._fc_async_two_phase, self._fc_async_use_real_audio, has_function_head, self._fc_sotc_id, self._fc_sotc_boost,
        )
        if self._fc_async_enabled and has_function_head:
            self._create_silence_embedding_cache()

        # Turn-taking source: "rnnt" (default) or "asr_head"
        self._turn_taking_source = str(self.model_cfg.get("turn_taking_source", "rnnt")).lower()
        # rnnt_eou_frames: number of consecutive 80 ms frames with NO new RNNT
        # tokens (i.e. silence) before EOU is declared.  On EOU the agent BOS
        # token is force-inserted so the agent starts speaking.
        # Naming note: "EOU" is from the USER's perspective (user End-Of-Utterance
        # = user stopped talking).  The resulting action is on the AGENT side
        # (agent BOS = agent begins speaking).  Default 800 ms → 10 frames.
        self._rnnt_eou_frames = int(self.model_cfg.get("rnnt_eou_frames", 10))
        # rnnt_bou_frames: number of consecutive 80ms frames that must contain a
        # new, non-unk RNNT token before BOU is declared.  Mirrors EOU patience:
        # just as EOU waits for N silent frames before deciding the user stopped,
        # BOU waits for N speech frames before deciding the user started.
        # This prevents a single noise burst (one unk token, a breath, a click)
        # from falsely interrupting the agent.
        # Default 3 frames × 80ms = 240ms — close to Riva/NIM's ~200ms
        # sustained-speech threshold for BOU.
        self._rnnt_bou_frames = int(self.model_cfg.get("rnnt_bou_frames", 3))
        # Counter for consecutive frames of real (non-unk) speech; used by BOU.
        self._rnnt_consecutive_speech_frames = 0
        # FC async interrupt: frames elapsed since user BOU before aborting the
        # background tool-call thread.  Independent of rnnt_eou_frames so it can
        # be tuned without affecting normal turn-taking.
        # Default 240 ms → 240 / 80 = 3 frames.
        _fc_interrupt_ms = int(self.model_cfg.get("rnnt_fc_interrupt_ms", 240))
        self._rnnt_fc_interrupt_frames = max(1, _fc_interrupt_ms // 80)
        self._rnnt_last_speech_frame = -1
        self._rnnt_prev_num_tokens = 0
        self._rnnt_bos_cooldown_until = -1
        self._rnnt_eos_cooldown_until = -1
        # Set to True by _maybe_apply_rnnt_turn_taking when agent EOS is inserted (BOU barge-in).
        # Pipeline monitors this to fire quit_async_event (kill tool-call thread on agent EOU).
        self._agent_eos_just_fired = False
        # Max agent response duration: force EOS after N frames of continuous agent speech.
        # Then inject a redirect message ("How can I help you?") via the token queue.
        # 15s / 80ms per frame = 187 frames. Set to 0 to disable.
        _max_resp_sec = float(self.model_cfg.get("max_agent_response_sec", 15.0))
        self._max_agent_response_frames = int(_max_resp_sec / 0.08) if _max_resp_sec > 0 else 0
        # Build word-start token flag list (same pattern as NeMo's greedy_decoder.py).
        # is_start_tokens[token_id] = True if the token begins a new word (▁ prefix).
        # EOU is suppressed when the last RNNT token is a word-continuation subword —
        # the user is likely mid-word and the pause is within a multi-subword sequence.
        self._rnnt_is_start_tokens: list = []
        _rnnt_vocab = getattr(getattr(self.model.stt_model, "rnnt_joint", None), "vocabulary", None)
        if _rnnt_vocab is not None:
            self._rnnt_is_start_tokens = [token.startswith("\u2581") for token in _rnnt_vocab]
        # Find the unk token id so BOU can exclude noise-only frames.
        # Unk tokens (<unk>, ⁇) are emitted when the RNNT sees audio it cannot
        # recognize — typically noise or an out-of-vocabulary sound.  Counting
        # only unk tokens toward BOU would be misleading, so we skip them.
        self._rnnt_unk_id: Optional[int] = None
        if _rnnt_vocab is not None:
            for _idx, _tok in enumerate(_rnnt_vocab):
                if _tok in ("<unk>", "\u2047"):  # \u2047 = ⁇, another common unk glyph
                    self._rnnt_unk_id = _idx
                    break
        # Punctuation logit biasing: after the user speaks some words, blank frames
        # incrementally boost punct token logits inside the label loop so the RNNT
        # outputs natural sentence boundaries (periods, commas, question marks).
        # Applied ONLY in the inner label loop -- is_blank / turn-taking are untouched.
        self._rnnt_punct_ids: list = []
        self._rnnt_punct_ids_set: set = set()
        self._rnnt_punct_bias_increment: float = float(
            self.model_cfg.get("rnnt_punct_bias_increment", 2.5))
        if self.model_cfg.get("rnnt_punct_bias_enabled", False) and _rnnt_vocab is not None:
            for _pt in self.model_cfg.get("rnnt_punct_bias_tokens", [".", ",", "?", "!"]):
                for _variant in [_pt, "▁" + _pt]:   # bare token and word-start form
                    if _variant in _rnnt_vocab:
                        _pid = _rnnt_vocab.index(_variant)
                        if _pid not in self._rnnt_punct_ids_set:
                            self._rnnt_punct_ids.append(_pid)
                            self._rnnt_punct_ids_set.add(_pid)
            logging.info("[RNNT] Punct bias enabled: tokens=%s IDs=%s increment=%.1f",
                         self.model_cfg.get("rnnt_punct_bias_tokens"),
                         self._rnnt_punct_ids, self._rnnt_punct_bias_increment)
        logging.info(
            "[Turn-taking] RNNT is_start_tokens built: %d tokens (EOU word-boundary check %s)",
            len(self._rnnt_is_start_tokens),
            "enabled" if self._rnnt_is_start_tokens else "disabled (vocab unavailable)",
        )
        logging.info(
            "[Turn-taking] source=%s, rnnt_eou_frames=%d, rnnt_bou_frames=%d, "
            "rnnt_fc_interrupt_frames=%d (%d ms), rnnt_unk_id=%s",
            self._turn_taking_source, self._rnnt_eou_frames, self._rnnt_bou_frames,
            self._rnnt_fc_interrupt_frames, _fc_interrupt_ms, self._rnnt_unk_id,
        )

        # Setup perception cache if enabled
        if self.use_perception_cache:
            self.perception_cache_mgr = PerceptionCacheManager(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                use_cudagraph=self._use_perception_cudagraph,
                pad_and_drop_preencoded=self._pad_and_drop_preencoded,
            )
            if not self.perception_cache_mgr.setup():
                self.use_perception_cache = False
                self.perception_cache_mgr = None

    def _get_bos_embedding(self):
        """Get beginning of sequence embedding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _get_asr_bos_embedding(self) -> torch.Tensor:
        """Get ASR BOS embedding for AR decoding."""
        if not self._has_asr_head:
            hidden_dim = self.model.stt_model.embed_tokens.weight.shape[1]
            return torch.zeros(1, hidden_dim, device=self.device, dtype=self.dtype)
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_asr_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _create_silence_embedding_cache(self):
        """
        Encode 1 second of silence through the perception encoder and cache the result.
        During async FC mode, these embeddings are used in place of real user audio,
        allowing text/function tokens to generate at full LLM speed without waiting
        for 80ms audio frames.
        """
        logging.info("[FC Async] Creating silence embedding cache...")
        sample_rate = SAMPLE_RATE
        silence_audio = torch.zeros(1, sample_rate, device=self.device, dtype=torch.float32)
        silence_len = torch.tensor([sample_rate], device=self.device, dtype=torch.long)

        with torch.no_grad():
            silence_encoded, silence_lens, _ = self.model.stt_model.perception(
                input_signal=silence_audio,
                input_signal_length=silence_len,
                return_encoder_emb=True,
            )

        silence_encoded = silence_encoded.to(self.dtype)
        num_frames = silence_encoded.shape[1]
        self._silence_embedding_cache = silence_encoded[0, 0:1, :].clone()  # [1, 1, H] — single frame
        logging.info(
            "[FC Async] Silence embedding cached: shape=%s (1s → %d frames, using frame 0)",
            list(self._silence_embedding_cache.shape), num_frames,
        )

    def _encode_realtime_audio_frame(self, realtime_audio: dict) -> torch.Tensor | None:
        """Try to consume a real audio frame based on wall-clock gating.

        In real-time mode, audio frames become available at 80ms intervals
        (simulated via wall-clock time since the async loop started).  If a
        new frame has "arrived", we encode it through perception and return
        the embedding.  Otherwise return *None* so the caller uses silence.

        Mutates *realtime_audio* in-place (advances pointer, updates buffer).
        """
        elapsed = time.time() - realtime_audio["wall_start"]
        frames_released = int(elapsed / FRAME_SIZE_SEC)
        consumed_so_far = realtime_audio["frames_consumed"]
        next_frame = realtime_audio["next_audio_frame"]
        total_frames = realtime_audio["total_audio_frames"]

        if consumed_so_far >= frames_released or next_frame >= total_frames:
            return None

        slice_start = next_frame * FRAME_SIZE_SAMPLES
        slice_end = slice_start + FRAME_SIZE_SAMPLES
        new_audio = realtime_audio["audio_signal_tensor"][:, slice_start:slice_end]
        if new_audio.shape[1] == 0:
            return None

        buf = realtime_audio["audio_buffer"]
        fill = realtime_audio["buffer_fill_level"]
        buf_size = realtime_audio["buffer_size_samples"]
        buf, fill, current_buf = self._update_audio_buffer(buf, fill, new_audio, buf_size)
        realtime_audio["audio_buffer"] = buf
        realtime_audio["buffer_fill_level"] = fill

        buf_len = torch.tensor([current_buf.shape[1]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            source_encoded, _, _ = self.model.stt_model.perception(
                input_signal=current_buf,
                input_signal_length=buf_len,
                return_encoder_emb=True,
            )
        source_encoded = source_encoded.to(self.dtype)
        emb_idx = max(source_encoded.shape[1] - 2, 0)
        user_audio_emb = source_encoded[:, emb_idx:emb_idx + 1, :]

        realtime_audio["next_audio_frame"] = next_frame + 1
        realtime_audio["frames_consumed"] += 1
        return user_audio_emb

    def _perception_background_worker(
        self,
        realtime_audio: dict,
        emb_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        """Background thread: encode audio frames as they arrive (80ms gate)
        and put embeddings into *emb_queue* without blocking the LLM loop.

        Uses a dedicated CUDA stream so perception kernels can overlap with
        LLM work happening in the vLLM engine process.
        """
        buf = realtime_audio["audio_buffer"].clone()
        fill = realtime_audio["buffer_fill_level"]
        buf_size = realtime_audio["buffer_size_samples"]
        wall_start = realtime_audio["wall_start"]
        next_frame = realtime_audio["next_audio_frame"]
        total_frames = realtime_audio["total_audio_frames"]
        audio_tensor = realtime_audio["audio_signal_tensor"]
        consumed = realtime_audio["frames_consumed"]

        perception_stream = torch.cuda.Stream(device=self.device)

        while not stop_event.is_set():
            elapsed = time.time() - wall_start
            frames_released = int(elapsed / FRAME_SIZE_SEC)

            if consumed >= frames_released or next_frame >= total_frames:
                time.sleep(0.002)
                continue

            slice_start = next_frame * FRAME_SIZE_SAMPLES
            slice_end = slice_start + FRAME_SIZE_SAMPLES
            new_audio = audio_tensor[:, slice_start:slice_end]
            if new_audio.shape[1] == 0:
                next_frame += 1
                consumed += 1
                continue

            buf, fill, current_buf = self._update_audio_buffer(
                buf, fill, new_audio, buf_size
            )
            buf_len = torch.tensor(
                [current_buf.shape[1]], dtype=torch.long, device=self.device
            )

            with torch.cuda.stream(perception_stream):
                with torch.no_grad():
                    source_encoded, _, _ = self.model.stt_model.perception(
                        input_signal=current_buf,
                        input_signal_length=buf_len,
                        return_encoder_emb=True,
                    )
            perception_stream.synchronize()

            source_encoded = source_encoded.to(self.dtype)
            emb_idx = max(source_encoded.shape[1] - 2, 0)
            user_audio_emb = source_encoded[:, emb_idx : emb_idx + 1, :]

            emb_queue.put(user_audio_emb)
            next_frame += 1
            consumed += 1

        realtime_audio["audio_buffer"] = buf
        realtime_audio["buffer_fill_level"] = fill
        realtime_audio["next_audio_frame"] = next_frame
        realtime_audio["frames_consumed"] = consumed

    def _perception_live_worker(
        self,
        live_audio_queue: "queue.Queue",
        audio_buffer: torch.Tensor,
        buffer_fill: int,
        buffer_size_samples: int,
        emb_queue: "queue.Queue",
        stop_event: "threading.Event",
    ) -> None:
        """Background thread: encode live audio frames as they arrive from the
        Triton execute() calls (one 80ms frame per queue entry) and put
        ``(user_audio_emb, asr_emb_frame)`` tuples into *emb_queue*.

        Unlike ``_perception_background_worker``, there is no wall-clock gate —
        frames arrive exactly when Triton delivers them (every ~80ms in production).
        """
        perception_stream = torch.cuda.Stream(device=self.device)
        # Keep the rolling buffer on GPU so _update_audio_buffer doesn't
        # hit a device mismatch when new_audio arrives on self.device.
        buf = audio_buffer.to(self.device, dtype=self.dtype)
        fill = buffer_fill

        while not stop_event.is_set():
            try:
                new_audio = live_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if new_audio is None:
                break  # sentinel

            new_audio = new_audio.to(self.device, dtype=self.dtype)
            buf, fill, current_buf = self._update_audio_buffer(buf, fill, new_audio, buffer_size_samples)
            buf_len = torch.tensor([current_buf.shape[1]], dtype=torch.long, device=self.device)

            with torch.cuda.stream(perception_stream):
                with torch.no_grad():
                    source_encoded, _, asr_emb = self.model.stt_model.perception(
                        input_signal=current_buf,
                        input_signal_length=buf_len,
                        return_encoder_emb=True,
                    )
            perception_stream.synchronize()

            source_encoded = source_encoded.to(self.dtype)
            emb_idx = max(source_encoded.shape[1] - 2, 0)
            user_audio_emb = source_encoded[:, emb_idx : emb_idx + 1, :]
            # asr_emb: raw encoder output for RNNT (same slice)
            if asr_emb is not None:
                asr_emb = asr_emb.to(self.dtype)
                asr_emb_frame = asr_emb[:, emb_idx : emb_idx + 1, :]
            else:
                asr_emb_frame = None

            emb_queue.put((user_audio_emb, asr_emb_frame))

    def _run_fc_async_steps(
        self,
        fc_state: dict,
        gen_text: torch.Tensor,
        gen_asr_text: torch.Tensor,
        gen_function_text: torch.Tensor,
        current_frame_idx: int,
        dynamic_cache,
        input_embeds_history: list,
        tool_response_text: str | None,
        max_async_steps: int = 2000,
        realtime_audio: dict | None = None,
        request_id: str | None = None,
        tts_state: dict | None = None,
        live_audio_queue: "queue.Queue | None" = None,
        live_audio_buffer: torch.Tensor | None = None,
        live_buffer_fill: int = 0,
        live_buffer_size_samples: int = 0,
        rnnt_partial_hypotheses=None,
        tts_audio_output_queue: "queue.Queue | None" = None,
        acknowledgement_tokens: "list | None" = None,
        rnnt_text_queue: "queue.Queue | None" = None,
        abort_event: "threading.Event | None" = None,
        quit_async_event: "threading.Event | None" = None,
        stream_id: "int | None" = None,
    ) -> tuple[int, dict | None]:
        """
        Run LLM steps at full speed during function calling.

        For the user-audio channel, the behaviour depends on *realtime_audio*:

        * **None (default)** – every step uses the cached silence embedding
          (original behaviour, fastest).
        * **dict** – real-time mode.  Audio frames are gated by wall-clock
          time so that a new frame becomes available every 80 ms, exactly
          as it would arrive from a microphone.  When a frame is ready it
          is encoded through the perception encoder and used; otherwise
          the silence embedding is used.  The dict is mutated in-place to
          track consumed frames and buffer state.

        This is called when SOTC is detected in the function channel. It continues
        running until:
          - The full FC cycle completes (call generated → response injected → EOTR predicted)
          - max_async_steps is reached (safety limit)

        Args:
            fc_state: Function calling state dict (mutated in-place)
            gen_text: Agent text token buffer [B, T]
            gen_asr_text: ASR text token buffer [B, T]
            gen_function_text: Function channel token buffer [B, T]
            current_frame_idx: The frame index where SOTC was detected
            dynamic_cache: LLM KV cache (DynamicCache or None)
            input_embeds_history: List of past input embeddings (no-cache mode)
            tool_response_text: Tool response to inject after EOTC
            max_async_steps: Safety limit on number of async steps
            realtime_audio: Optional dict for real-time audio gating.
                Required keys when provided:
                  audio_signal_tensor  – [1, total_samples] full audio tensor
                  next_audio_frame     – next frame index to consume (mutated)
                  total_audio_frames   – total frames available
                  audio_buffer         – [1, buf_size] sliding window (mutated)
                  buffer_fill_level    – int, current fill (mutated)
                  buffer_size_samples  – int, buffer capacity
                  wall_start           – float, time.time() at async entry
                  frames_consumed      – int, counter (mutated)

        Returns:
            (num_async_steps, dynamic_cache, tts_state, tts_audio_chunks, rnnt_partial_hypotheses):
            Number of extra timeline positions consumed, the updated LLM cache,
            the updated TTS state dict (or None), a list of TTS-generated
            silence audio tensors (one per 80ms warmup tick), and the updated
            RNNT partial hypotheses (or None if RNNT was not run).
        """
        if self._silence_embedding_cache is None:
            logging.warning("[FC Async] No silence cache available, falling back to normal mode")
            return 0, dynamic_cache, tts_state, [], rnnt_partial_hypotheses

        pad_id = self.model.stt_model.text_pad_id
        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]
        T_total = gen_text.shape[1]
        silence_emb = self._silence_embedding_cache.expand(batch_size, 1, -1)  # [B, 1, H]
        has_rnnt = getattr(self.model.stt_model, '_rnnt_decoder', None) is not None

        # Determine fallback audio embedding for async FC steps where no real frame arrives.
        # fc_async_use_real_audio=False (default): always silence — avoids frozen-audio artifacts.
        # fc_async_use_real_audio=True: use last cached real audio frame if available.
        if self._fc_async_use_real_audio and self._last_user_audio_emb is not None:
            fallback_audio_emb = self._last_user_audio_emb.expand(batch_size, 1, -1)
            logging.info("[FC Async] Using cached real audio embedding as fallback (instead of silence)")
        else:
            fallback_audio_emb = silence_emb
            if not self._fc_async_use_real_audio:
                logging.info("[FC Async] fc_async_use_real_audio=False — using silence for all FC frames")
            else:
                logging.info("[FC Async] No cached audio embedding, using silence")

        async_steps = 0
        t = current_frame_idx + 1  # start from next position after SOTC
        rt_real_count = 0  # track how many real audio frames were used (for logging)

        # Phase 2 re-entry: if tool_response_text is provided AND we are NOT in
        # an active call (i.e. call was already generated in a previous invocation),
        # pre-queue forced response tokens before the loop.
        # When active=True, the model still needs to generate call tokens first;
        # response will be queued at EOTC detection instead.
        if (tool_response_text
                and not fc_state.get("forced_function_tokens")
                and not fc_state.get("active", False)):
            import time as _time
            response_tokens = self._build_fc_response_tokens(tool_response_text)
            fc_state["forced_function_tokens"] = response_tokens
            fc_state["injecting_response"] = True
            fc_state.setdefault("tool_response_inject_start_wall", _time.time())
            fc_state.setdefault("tool_response_inject_start_frame", current_frame_idx + 1)
            fc_state["tool_response_num_tokens"] = len(response_tokens)
            logging.info(
                "[FC Async] Pre-queued %d response tokens for injection (phase 2 re-entry)",
                len(response_tokens),
            )

        logging.info(
            "[FC Async] Entering async loop at frame %d (SOTC detected at %d, realtime_audio=%s, live_audio=%s)",
            t, current_frame_idx,
            "ON" if realtime_audio is not None else "OFF",
            "ON" if live_audio_queue is not None else "OFF",
        )

        _t_embed_total = 0.0
        _t_perception_total = 0.0
        _t_fusion_total = 0.0
        _t_llm_total = 0.0
        _t_tts_total = 0.0
        _n_perception_calls = 0
        _n_tts_warmup_calls = 0
        _tts_silence_audio_chunks = []
        _n_rnnt_steps = 0
        _ack_idx = 0  # index into acknowledgement_tokens; advances each TTS step

        # Start background perception thread so LLM never blocks on audio encoding.
        # Two modes:
        #  - realtime_audio (offline sim): wall-clock gated, reads from pre-loaded tensor
        #  - live_audio_queue (Triton server): reads frames pushed by execute() calls
        _percep_queue = None
        _percep_stop = None
        _percep_thread = None
        if live_audio_queue is not None:
            # Live mode: frames arrive from Triton execute() calls via live_audio_queue.
            # The emb_queue entries are (user_audio_emb, asr_emb_frame) tuples.
            _percep_queue = queue.Queue()
            _percep_stop = threading.Event()
            _percep_thread = threading.Thread(
                target=self._perception_live_worker,
                args=(
                    live_audio_queue,
                    live_audio_buffer if live_audio_buffer is not None
                        else torch.zeros(1, live_buffer_size_samples, dtype=self.dtype),  # CPU — avoids inference tensor
                    live_buffer_fill,
                    live_buffer_size_samples,
                    _percep_queue,
                    _percep_stop,
                ),
                daemon=True,
            )
            _percep_thread.start()
        elif realtime_audio is not None:
            _percep_queue = queue.Queue()
            _percep_stop = threading.Event()
            _percep_thread = threading.Thread(
                target=self._perception_background_worker,
                args=(realtime_audio, _percep_queue, _percep_stop),
                daemon=True,
            )
            _percep_thread.start()
        # Track whether emb_queue entries are (emb, asr_emb) tuples (live mode) or plain tensors
        _live_mode = live_audio_queue is not None

        while async_steps < max_async_steps and t < T_total:
            # Check for interrupt signal from the main thread (user spoke during FC).
            if abort_event is not None and abort_event.is_set():
                logging.info(
                    "[FC Async] Abort event set — stopping async generation at step %d (frame %d)",
                    async_steps, t,
                )
                # ----- TTS PAD-silence: Caveat 2 hook (implicit turn end) -----
                # A mid-FC abort (typically user barging in) ends the agent's
                # current "turn" (the FC cycle) WITHOUT a natural EOS landing on
                # the agent text channel. Without this hook, `agent_idle`
                # could remain False from the time BOS was emitted, and the
                # subsequent silent frames after the abort would NOT get the
                # PAD-silence substitution — leading back to the garbled-syllables
                # symptom we are trying to prevent.
                #
                # `_mark_agent_idle(stream_id)` is idempotent and locked
                # (S2SStreamingState.mark_agent_idle() acquires the per-instance
                # _agent_idle_lock). Safe to call multiple times.
                self._mark_agent_idle(stream_id)
                break
            if quit_async_event is not None and quit_async_event.is_set():
                logging.info(
                    "[FC Async] Quit event set — stopping async generation at step %d (frame %d)",
                    async_steps, t,
                )
                break

            _step_start = time.time()

            agent_text_emb = self.model.stt_model.embed_tokens(
                gen_text[:, t - 1]
            ).unsqueeze(1).to(dtype=self.dtype)
            if self._has_asr_head:
                user_text_emb = self.model.stt_model.embed_asr_tokens(
                    gen_asr_text[:, t - 1]
                ).unsqueeze(1).to(dtype=self.dtype)
            else:
                user_text_emb = None
            function_emb = self.model.stt_model.embed_tokens(
                gen_function_text[:, t - 1]
            ).unsqueeze(1).to(dtype=self.dtype)
            _t_embed = time.time()
            _t_embed_total += _t_embed - _step_start

            # Non-blocking: pick up the latest perception embedding if available.
            # In live mode, queue entries are (user_audio_emb, asr_emb_frame) tuples.
            # In offline mode, queue entries are plain user_audio_emb tensors.
            user_audio_emb = None
            live_asr_emb_frame = None  # only populated in live mode
            _got_real_audio_frame = False
            if _percep_queue is not None:
                latest = None
                try:
                    while True:
                        latest = _percep_queue.get_nowait()
                except queue.Empty:
                    pass
                if latest is not None:
                    if _live_mode:
                        user_audio_emb, live_asr_emb_frame = latest
                    else:
                        user_audio_emb = latest
                    rt_real_count += 1
                    _n_perception_calls += 1
                    _got_real_audio_frame = True
            if user_audio_emb is None or not self._fc_async_use_real_audio:
                user_audio_emb = fallback_audio_emb
            else:
                fallback_audio_emb = user_audio_emb.detach()
            _t_percep = time.time()
            _t_perception_total += _t_percep - _t_embed

            fused_emb = self.fusion_module(
                agent_text_embeds=agent_text_emb,
                user_audio_embeds=user_audio_emb,
                user_text_embeds=user_text_emb,
                function_embeds=function_emb,
            )
            _t_fuse = time.time()
            _t_fusion_total += _t_fuse - _t_percep

            if self.use_vllm_llm:
                ans = self.model_llm_interface(
                    fused_emb,
                    request_id=request_id or self.request_id,
                    generated_tokens=gen_text,
                    current_step=t,
                )
            elif use_cache:
                ans = self.model_llm_interface(
                    fused_emb,
                    cache=dynamic_cache,
                    generated_tokens=gen_text,
                    current_step=t,
                )
                dynamic_cache = ans["cache"]
            else:
                input_embeds_history.append(fused_emb)
                full_input_embeds = torch.cat(input_embeds_history, dim=1)
                ans = self.model_llm_interface(
                    full_input_embeds,
                    cache=None,
                    generated_tokens=gen_text,
                    current_step=t,
                )
            _t_llm_total += time.time() - _t_fuse

            # TTS KV-cache sync: run one TTS forward pass with PAD on EVERY async
            # LLM step so that past_key_values stays aligned with frame_idx.
            # Without this, the TTS KV cache only advances K times (once per 80ms
            # real-audio tick) while frame_idx jumps N steps (all async LLM steps),
            # leaving a gap of N-K that causes choppy audio after the FC completes.
            # Codec decode (the expensive part) still only runs on 80ms ticks to
            # produce the silence audio sent to the client.
            if tts_state is not None:
                _t_tts_start = time.time()
                try:
                    _tts_code = tts_state["code"]
                    _tts_pkv = tts_state["past_key_values"]
                    _tts_sw_mask = tts_state["subword_mask"]
                    _tts_codec_cache = tts_state["codec_cache"]

                    # Use acknowledgement token if available, otherwise pad.
                    if acknowledgement_tokens and _ack_idx < len(acknowledgement_tokens):
                        _tts_subword = torch.tensor(
                            [[acknowledgement_tokens[_ack_idx]]], device=self.device, dtype=torch.long
                        )
                        _tts_sw_mask_val = torch.ones(1, 1, device=self.device, dtype=torch.bool)
                    else:
                        _tts_subword = torch.full(
                            (1, 1), pad_id, device=self.device, dtype=torch.long
                        )
                        _tts_sw_mask_val = _tts_sw_mask[:, t].unsqueeze(-1) if t < _tts_sw_mask.shape[1] else torch.ones(1, 1, device=self.device, dtype=torch.bool)
                    _ack_idx += 1

                    _tts_inputs = {
                        "current_subword_id": _tts_subword,
                        "prev_subword_id": _tts_subword,
                        "current_subword_mask": _tts_sw_mask_val,
                        "prev_audio_tokens": _tts_code,
                        "past_key_values": _tts_pkv,
                        "guidance_enabled": True,
                        "generation_config": self.generation_config,
                        "ignore_eos_flag_stop": True,
                    }
                    if self.use_vllm_eartts:
                        _tts_inputs["request_id"] = request_id or self.request_id

                    _tts_code_new, _tts_pkv_new = self.model.tts_model.infer_codes_one_step(**_tts_inputs)

                    # ====== TTS PAD-silence substitution — SITE 1 (FC async warmup) ======
                    # This is the TTS warmup that runs once per async-loop step during
                    # the FC cycle. The agent text channel is PAD throughout the FC
                    # cycle (forced PAD on the agent channel — function tokens go to
                    # the function channel instead).
                    #
                    # Substitute only when ALL THREE conditions are true:
                    #   (a) cfg flag inference_force_speech_silence_on_pad is on
                    #   (b) the stream is "agent idle" (no live agent verbal turn)
                    #   (c) we are NOT mid-ack#1 — ack #1 tokens (BOS + content +
                    #       trailing PAD + EOS) must render in full. Silencing the
                    #       trailing PAD frames of ack #1 would chop the ack audio.
                    #
                    # Note on (c) timing: _ack_idx was already incremented above, so:
                    #   _ack_idx <= len(acknowledgement_tokens)  → we just used an ack token
                    #   _ack_idx >  len(acknowledgement_tokens)  → we used PAD fallback (post-ack)
                    # `not acknowledgement_tokens` matches the truthy check at the consumer
                    # (line ~1406) so both `None` and `[]` are handled as "no ack to play".
                    _ack_done = (not acknowledgement_tokens
                                 or _ack_idx > len(acknowledgement_tokens))
                    if (self.model.cfg.get('inference_force_speech_silence_on_pad', None)
                            and self._get_agent_idle(stream_id)
                            and _ack_done):
                        _pad_silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(_tts_code_new.shape)
                        _tts_code_new = torch.where(
                            _tts_subword.unsqueeze(-1) == self.model.tts_model.text_pad_id,
                            _pad_silence_codes,
                            _tts_code_new,
                        )

                    tts_state["code"] = _tts_code_new
                    tts_state["past_key_values"] = _tts_pkv_new
                    _n_tts_warmup_calls += 1

                    # Codec decode on EVERY async step to keep the codec streaming
                    # cache aligned with the TTS code sequence.
                    # Without this, the codec cache only advances K times (80ms ticks)
                    # while TTS generates N codes, leaving the codec N-K steps behind
                    # and causing choppy audio for the first few post-FC sentences.
                    # Audio is only sent to the client on 80ms ticks.
                    if _tts_codec_cache is not None:
                        with fp32_precision(), torch.no_grad():
                            _tts_new_codes = _tts_code_new.unsqueeze(0) if _tts_code_new.dim() == 2 else _tts_code_new
                            if hasattr(self.model.tts_model, '_control_codes'):
                                from nemo.collections.speechlm2.models.duplex_ear_tts import replace_control_speech_codes
                                _tts_new_codes = replace_control_speech_codes(
                                    _tts_new_codes,
                                    self.model.tts_model._control_codes,
                                    getattr(self.model.tts_model, 'codec_silence_tokens', None),
                                )
                            _tts_code_len = torch.tensor([1], dtype=torch.long, device=self.device)
                            _tts_decoded_audio, _ = self.model.tts_model.audio_codec.decode(
                                _tts_new_codes, _tts_code_len, cache=_tts_codec_cache,
                            )
                            _tts_chunk_cpu = _tts_decoded_audio.detach().cpu()
                            _tts_silence_audio_chunks.append(_tts_chunk_cpu)
                            if tts_audio_output_queue is not None:
                                tts_audio_output_queue.put(_tts_chunk_cpu)
                except Exception as _tts_exc:
                    logging.warning("[FC Async] TTS warmup step failed: %s", _tts_exc)
                _t_tts_total += time.time() - _t_tts_start

            if ans.get("is_finished", False):
                logging.warning(
                    "[FC Async] vLLM sequence finished at step %d (frame %d), exiting async loop",
                    async_steps, t,
                )
                break

            predicted_token = ans["predicted_token"]
            asr_predicted_token = ans["asr_predicted_token"]

            gen_text[:, t] = pad_id  # force PAD during FC async — prevents apology-token KV-cache contamination
            gen_asr_text[:, t] = pad_id  # silence → no user speech
            if "function_predicted_token" in ans:
                gen_function_text[:, t] = ans["function_predicted_token"]

            # Run FC state machine on the function channel output
            func_tok_val = gen_function_text[:, t].item() if gen_function_text[:, t].dim() == 0 else gen_function_text[0, t].item()

            # --- Forced injection phase ---
            forced = fc_state.get("forced_function_tokens", [])
            if forced:
                override_id = forced.pop(0)
                gen_function_text[:, t] = override_id
                gen_text[:, t] = pad_id
                fc_state["injecting_response"] = True
            else:
                if fc_state.get("injecting_response", False):
                    fc_state["injecting_response"] = False
                    logging.info(
                        "[FC Async] Forced TOOL_RESPONSE tokens drained at step %d (frame %d) "
                        "— now awaiting model to predict EOTR on function channel",
                        async_steps, t,
                    )

                # --- Natural token monitoring ---
                func_tok_val = gen_function_text[0, t].item()
                if func_tok_val == self._fc_sotc_id:
                    fc_state["active"] = True
                    fc_state["call_tokens"] = []
                    logging.info(f"[FC Async] SOTC at async step {async_steps} (frame {t})")
                elif func_tok_val == self._fc_eotc_id and fc_state.get("active", False):
                    fc_state["active"] = False
                    call_text = self.tokenizer.ids_to_text(fc_state.get("call_tokens", []))
                    fc_state.setdefault("completed_calls", []).append(call_text)
                    fc_state["eotc_async_step"] = async_steps
                    fc_state["eotc_frame"] = t
                    logging.info(f"[FC Async] EOTC at async step {async_steps} (frame {t}). Call: {call_text}")

                    if tool_response_text:
                        import time as _time
                        api_latency = self._fc_async_api_latency_sec
                        if api_latency > 0:
                            fc_state["api_call_start_wall"] = _time.time()
                            fc_state["api_call_start_frame"] = t + 1
                            logging.info(
                                "[FC Async] Simulating API call (%.0fms latency)...",
                                api_latency * 1000,
                            )
                            _time.sleep(api_latency)
                            fc_state["api_call_end_wall"] = _time.time()
                            fc_state["api_call_end_frame"] = t + 1
                            logging.info("[FC Async] API call complete")
                        fc_state["tool_response_inject_start_wall"] = _time.time()
                        fc_state["tool_response_inject_start_frame"] = t + 1
                        response_tokens = self._build_fc_response_tokens(tool_response_text)
                        fc_state["forced_function_tokens"] = response_tokens
                        fc_state["tool_response_num_tokens"] = len(response_tokens)
                        logging.info(f"[FC Async] Queued {len(response_tokens)} response tokens")
                    elif self._fc_async_two_phase:
                        fc_state["awaiting_response"] = True
                        fc_state["last_call_text"] = call_text
                        fc_state["phase1_end_t"] = t + 1
                        logging.info(
                            "[FC Async] Two-phase: Phase 1 complete (call generated). "
                            "Exiting async to await tool execution."
                        )
                        async_steps += 1
                        t += 1
                        break
                    else:
                        logging.info("[FC Async] No tool_response_text, exiting async after EOTC")
                        async_steps += 1
                        t += 1
                        break
                elif self._fc_eotr_id is not None and func_tok_val == self._fc_eotr_id:
                    logging.info(f"[FC Async] EOTR predicted at async step {async_steps} (frame {t})")
                elif fc_state.get("active", False) and func_tok_val != pad_id:
                    fc_state.setdefault("call_tokens", []).append(func_tok_val)

            # RNNT decode on live audio frames (Triton server mode).
            # Turn-taking is intentionally skipped here: the main thread monitors
            # the rnnt_text_queue and sets abort_event when it detects persistent
            # user speech.  Calling _maybe_apply_rnnt_turn_taking inside the FC
            # loop would insert EOS/BOS tokens into gen_text and disrupt the tool
            # call generation even when the user is not speaking.
            if _live_mode and _got_real_audio_frame and has_rnnt and live_asr_emb_frame is not None:
                # live_asr_emb_frame may be [B,1,D] or [B,D] — normalise to [B,D]
                _lf = live_asr_emb_frame.squeeze(1) if live_asr_emb_frame.dim() == 3 else live_asr_emb_frame
                if rnnt_partial_hypotheses is None:
                    rnnt_partial_hypotheses = self._rnnt_init_state(batch_size, self.device)
                rnnt_partial_hypotheses, _rnnt_is_blank = self._rnnt_step(_lf, rnnt_partial_hypotheses)
                if abort_event is None:
                    self._apply_rnnt_turn_taking(t, gen_text, _rnnt_is_blank, rnnt_partial_hypotheses)
                _n_rnnt_steps += 1
                # Stream user transcript during FC: decode y_sequence and send to UI.
                # Turn-taking is off during FC; this is display-only, no inference impact.
                if rnnt_text_queue is not None:
                    _fc_y_seq = rnnt_partial_hypotheses.get('y_sequence', [])
                    if _fc_y_seq:
                        _fc_text = self._rnnt_decode_text(_fc_y_seq)
                        if _fc_text:
                            try:
                                rnnt_text_queue.put_nowait(_fc_text)
                            except Exception:
                                pass

            # Natural interrupt: the model's agent text head predicted a real
            # token (non-PAD/BOS/EOS) while receiving real user audio.
            # This means the model itself decided to stop FC and respond to
            # the user's interrupt question — no heuristic threshold needed.
            # Only fires on real audio frames (not silence/fallback) to avoid
            # false triggers on background noise.
            if _got_real_audio_frame:
                _agent_tok_val = gen_text[0, t].item()
                _bos_id = getattr(self.model.stt_model, "text_bos_id", pad_id)
                _eos_id = getattr(self.model.stt_model, "text_eos_id", pad_id)
                if _agent_tok_val not in (pad_id, _bos_id, _eos_id):
                    logging.info(
                        "[FC Async] Natural interrupt at async step %d (frame %d): "
                        "agent text head predicted '%s' (id=%d) — breaking to answer user",
                        async_steps, t,
                        self.tokenizer.ids_to_text([_agent_tok_val]),
                        _agent_tok_val,
                    )
                    fc_state["natural_interrupt"] = True
                    async_steps += 1
                    t += 1
                    break

            async_steps += 1
            t += 1

            # Exit conditions: FC cycle complete (no more forced tokens, not in active call,
            # and not injecting response). EOTR is appended to forced tokens so the model
            # has the full context when the main loop resumes.
            no_forced = not fc_state.get("forced_function_tokens", [])
            not_active = not fc_state.get("active", False)
            not_injecting = not fc_state.get("injecting_response", False)
            if no_forced and not_active and not_injecting:
                if "tool_response_inject_start_wall" in fc_state:
                    import time as _time
                    fc_state["tool_response_inject_end_wall"] = _time.time()
                    fc_state["tool_response_inject_end_frame"] = t
                logging.info(
                    "[FC Async] FC cycle complete after %d async steps (frame %d)",
                    async_steps, t,
                )
                break

        # Stop background perception thread
        if _percep_thread is not None:
            _percep_stop.set()
            _percep_thread.join(timeout=2.0)
            if _percep_thread.is_alive():
                logging.warning("[FC Async] Perception thread did not stop in time")

        if async_steps >= max_async_steps:
            logging.warning(
                "[FC Async] Hit max_async_steps=%d safety limit at frame %d",
                max_async_steps, t,
            )

        rt_info = ""
        if live_audio_queue is not None:
            rt_info = (
                f", live_audio: {rt_real_count} real frames / "
                f"{async_steps} total steps ({rt_real_count / max(async_steps, 1) * 100:.1f}% real), "
                f"RNNT steps: {_n_rnnt_steps}"
            )
        elif realtime_audio is not None:
            rt_info = (
                f", realtime_audio: {rt_real_count} real frames / "
                f"{async_steps} total steps ({rt_real_count / max(async_steps, 1) * 100:.1f}% real)"
            )
        _t_total = _t_embed_total + _t_perception_total + _t_fusion_total + _t_llm_total + _t_tts_total
        tts_info = ""
        if _n_tts_warmup_calls > 0:
            tts_info = f", TTS warmup: {_n_tts_warmup_calls} calls ({len(_tts_silence_audio_chunks)} audio chunks)"
        logging.info("[FC Async] Exiting async loop: %d steps, resuming at frame %d%s%s", async_steps, t, rt_info, tts_info)
        logging.info(
            "[FC Async] Step timing breakdown (total %.3fs, %d steps):\n"
            "    embed:      %.3fs (%.1f%%, %.1fms/step)\n"
            "    perception: %.3fs (%.1f%%, %.1fms/step, %d encoder calls)\n"
            "    fusion:     %.3fs (%.1f%%, %.1fms/step)\n"
            "    LLM:        %.3fs (%.1f%%, %.1fms/step)\n"
            "    TTS warmup: %.3fs (%.1f%%, %d calls, %.1fms/call)",
            _t_total, async_steps,
            _t_embed_total, 100 * _t_embed_total / max(_t_total, 1e-9), 1000 * _t_embed_total / max(async_steps, 1),
            _t_perception_total, 100 * _t_perception_total / max(_t_total, 1e-9), 1000 * _t_perception_total / max(async_steps, 1), _n_perception_calls,
            _t_fusion_total, 100 * _t_fusion_total / max(_t_total, 1e-9), 1000 * _t_fusion_total / max(async_steps, 1),
            _t_llm_total, 100 * _t_llm_total / max(_t_total, 1e-9), 1000 * _t_llm_total / max(async_steps, 1),
            _t_tts_total, 100 * _t_tts_total / max(_t_total, 1e-9), _n_tts_warmup_calls, 1000 * _t_tts_total / max(_n_tts_warmup_calls, 1),
        )
        return async_steps, dynamic_cache, tts_state, _tts_silence_audio_chunks, rnnt_partial_hypotheses

    def _prepare_system_prompt_embeddings(
        self,
        system_prompt: str,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        Prepare system prompt embeddings consistent with offline_inference.

        In offline_inference, prompt embeddings are structured as:
        - Position 0: prompt_token_emb + bos_emb + asr_bos
        - Position t > 0: prompt_token_emb + pad_emb + pad_asr

        Args:
            system_prompt: The system prompt text

        Returns:
            Tuple of (prompt_embedded [1, prompt_len, H], prompt_length)
            Returns (None, 0) if system_prompt is empty
        """

        if not system_prompt or not system_prompt.strip():
            return None, 0

        logging.info(f"Preparing system prompt: {system_prompt[:100]}...")

        # Step 1: Tokenize the prompt
        # Format: [bos] + text_tokens + [eos] (consistent with collate_system_prompt)
        single_prompt_token_ids = (
            [self.tokenizer.bos_id] +
            self.tokenizer.text_to_ids(system_prompt) +
            [self.tokenizer.eos_id]
        )
        # Prompt repetition (arxiv 2512.14982): repeat the prompt twice during prefill so
        # each token can attend to all others, improving recall of tool definitions across turns.
        repeat_n = self.model_cfg.get("system_prompt_repeat_n", 2)
        prompt_token_ids = single_prompt_token_ids * repeat_n
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, prompt_len]
        prompt_len = prompt_tokens.shape[1]

        logging.info(f"   Prompt length: {prompt_len} tokens (repeated {repeat_n}x, single={len(single_prompt_token_ids)})")
        # Debug: decode and log the full repeated prompt so the format can be verified in server logs
        try:
            decoded_prompt = self.tokenizer.ids_to_text(prompt_token_ids)
            logging.info(f"   [DEBUG] Repeated prompt decoded: {decoded_prompt[:500]}{'...' if len(decoded_prompt) > 500 else ''}")
        except Exception as e:
            logging.warning(f"   [DEBUG] Could not decode repeated prompt: {e}")

        # Step 2: Build per-channel embedding tensors and fuse them,
        # mirroring how the training code combines modalities via fusion_module.
        pad_id = self.model.stt_model.text_pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)

        # User-audio channel: prompt token embeddings replace actual audio
        user_audio_embeds = self.model.stt_model.embed_tokens(prompt_tokens).to(dtype=self.dtype)  # [1, T, H]

        # Agent-text channel: text feedback = embed_tokens(pad_id) at every position
        # (BOS also uses pad_id in this model, so position 0 is identical)
        pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)  # [1, H]
        agent_text_embeds = pad_emb.unsqueeze(0).expand(1, prompt_len, -1)  # [1, T, H]

        # User-text / ASR channel: embed_asr_tokens(pad_id) at every position
        if self._has_asr_head:
            pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)
            user_text_embeds = pad_asr_emb.unsqueeze(0).expand(1, prompt_len, -1)  # [1, T, H]
        else:
            user_text_embeds = None

        # Function channel
        has_fc = self.model.stt_model.function_head is not None
        function_embeds = pad_emb.unsqueeze(0).expand(1, prompt_len, -1) if has_fc else None

        prompt_embedded = self.fusion_module(
            agent_text_embeds=agent_text_embeds,
            user_audio_embeds=user_audio_embeds,
            user_text_embeds=user_text_embeds,
            function_embeds=function_embeds,
        )

        logging.info(f"   System prompt embeddings prepared: shape {prompt_embedded.shape}")

        return prompt_embedded, prompt_len

    def _clone_cache(self, cache):
        """Deep clone cache structures to ensure complete isolation between streams."""
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            return cache.detach().clone()
        if isinstance(cache, (list, tuple)):
            return type(cache)(self._clone_cache(x) for x in cache)
        if isinstance(cache, dict):
            return {k: self._clone_cache(v) for k, v in cache.items()}
        # Handle complex objects (e.g., DynamicCache with __dict__ attributes)
        # Use deepcopy to ensure complete isolation between streams
        if hasattr(cache, '__dict__'):
            import copy
            return copy.deepcopy(cache)
        return cache

    def _prepare_tts_initial_state(self):
        if not self.decode_audio:
            return
        if not hasattr(self.model, 'tts_model'):
            return

        logging.info("Preparing TTS warmup state...")

        with fp32_precision():
            speaker_audio, speaker_sr = torchaudio.load(self.speaker_reference)
            speaker_audio = resample(speaker_audio, speaker_sr, self.model.tts_model.target_sample_rate)

        speaker_audio = speaker_audio.to(self.device)
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=self.device).long()

        #  init tts_model
        self.model.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            system_prompt=self.tts_system_prompt,
        )
        init_inputs = self.model.tts_model.get_init_inputs(B=1)

        self.generation_config = self.model.tts_model._get_generation_config(guidance_enabled=True)
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        with torch.no_grad():
            if self.use_vllm_eartts:
                self.tts_prompt_token_ids = init_inputs["subword_ids"].squeeze().cpu().numpy().tolist()
                self.tts_init_inputs = init_inputs
                outputs = self.model.tts_model.tts_model(
                    self.tts_init_inputs,
                    request_id="tts_system_prompt_prefill_request",
                    prompt_token_ids=self.tts_prompt_token_ids
                )
                # abort this request
                self.model.tts_model.tts_model.abort_request("tts_system_prompt_prefill_request")
            else:
                outputs = self.model.tts_model.tts_model(**init_inputs)

            code = init_inputs["code"][:, -1:]
            # code, _, _ = self.model.tts_model.tts_model.generate_step(
            #     outputs.hidden_states[:, -1:], **self.generation_config
            # )

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)


        logging.info("TTS warmup state prepared")

    def _update_audio_buffer(self, audio_buffer, buffer_fill_level, new_audio, buffer_size_samples):
        """
        Append incoming samples to the sliding-window buffer and produce the view used for inference.

        Parameters:
            audio_buffer (torch.Tensor): Tensor of shape `[1, buffer_size_samples]` holding the latest audio samples.
            buffer_fill_level (int): Number of valid samples currently stored in `audio_buffer`.
            new_audio (torch.Tensor): Incoming samples of shape `[1, slice_n_samples]` for the current step.
            buffer_size_samples (int): Total capacity of the buffer in samples.

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor]:
                - Updated `audio_buffer` containing the newest samples (always capped to `buffer_size_samples`).
                - Updated `buffer_fill_level`, reflecting how many contiguous samples are valid.
                - `current_buffer`, a view over the valid portion of the buffer used for the model input.

        Notes:
            `audio_buffer` always retains the last `buffer_size_samples` samples even when overfilled,
            whereas `current_buffer` may be shorter during the initial warm-up phase when the buffer
            is not yet full.
        """
        if new_audio.shape[1] == 0:
            current_buffer = audio_buffer[:, :buffer_fill_level]
            return audio_buffer, buffer_fill_level, current_buffer

        remaining = new_audio

        if buffer_fill_level < buffer_size_samples and remaining.shape[1] > 0:
            warmup_take = min(buffer_size_samples - buffer_fill_level, remaining.shape[1])
            if warmup_take > 0:
                audio_buffer[:, buffer_fill_level:buffer_fill_level + warmup_take] = remaining[:, :warmup_take]
                buffer_fill_level += warmup_take
                remaining = remaining[:, warmup_take:]

        if remaining.shape[1] > 0:
            if remaining.shape[1] >= buffer_size_samples:
                audio_buffer = remaining[:, -buffer_size_samples:]
            else:
                audio_buffer = torch.cat([
                    audio_buffer[:, remaining.shape[1]:],
                    remaining
                ], dim=1)
            buffer_fill_level = buffer_size_samples
        current_buffer = audio_buffer if buffer_fill_level == buffer_size_samples else audio_buffer[:, :buffer_fill_level]
        return audio_buffer, buffer_fill_level, current_buffer

    @staticmethod
    def _convert_nums_in_json(obj, _engine=[]):
        """Recursively convert numeric values in a JSON-like object to spoken text."""
        if not _engine:
            import inflect
            _engine.append(inflect.engine())
        eng = _engine[0]

        if isinstance(obj, dict):
            return {k: NemotronVoicechatInferenceWrapper._convert_nums_in_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NemotronVoicechatInferenceWrapper._convert_nums_in_json(item) for item in obj]
        elif isinstance(obj, float):
            try:
                int_part = int(obj)
                frac_str = str(obj).split(".")[1] if "." in str(obj) else ""
                result = eng.number_to_words(int_part, andword="")
                if frac_str:
                    result += " point " + " ".join(eng.number_to_words(int(d)) for d in frac_str)
                return result
            except Exception:
                return str(obj)
        elif isinstance(obj, int):
            try:
                return eng.number_to_words(obj, andword="")
            except Exception:
                return str(obj)
        return obj

    def _convert_tool_response_nums_to_text(self, tool_response_text: str) -> str:
        """Convert digit numbers in a TOOL_RESPONSE string to spoken text."""
        import json
        tag_pattern = re.compile(r'(<TOOL_RESPONSE>)(.*?)(</TOOL_RESPONSE>)', re.DOTALL)

        def _replace(match):
            open_tag, json_str, close_tag = match.group(1), match.group(2), match.group(3)
            try:
                parsed = json.loads(json_str)
                converted = self._convert_nums_in_json(parsed)
                return open_tag + json.dumps(converted) + close_tag
            except (json.JSONDecodeError, TypeError):
                return match.group(0)

        result = tag_pattern.sub(_replace, tool_response_text)
        if result != tool_response_text:
            logging.info("[FC] Converted numbers to text in tool response")
            logging.debug("[FC]   Before: %s", tool_response_text[:200])
            logging.debug("[FC]   After:  %s", result[:200])
        return result

    # ------------------------------------------------------------------
    # TTS PAD-silence substitution: per-stream "agent idle" state
    # (Option B — multi-stream-safe via S2SStreamingState).
    #
    # The pipeline calls `_set_streaming_state_getter(self.get_or_create_state)`
    # once at __init__ time. From then on, the three TTS sites in this
    # wrapper read/write the per-stream `agent_idle` flag through the
    # helpers below — so two concurrent user sessions never share state.
    #
    # If the getter has not been installed (e.g. older pipeline), the
    # helpers fall back to the wrapper-level `self._agent_idle` and the
    # fix degrades gracefully to single-stream behavior instead of breaking.
    # ------------------------------------------------------------------
    def _set_streaming_state_getter(self, getter):
        """Install a `callable(stream_id) -> S2SStreamingState | None` lookup.

        Called once by the pipeline at wiring time. Example:
            wrapper._set_streaming_state_getter(pipeline.get_or_create_state)
        Storing the callable (not the dict) avoids tight coupling to the
        pipeline's internal data structure.
        """
        self._streaming_state_getter = getter

    def _get_agent_idle(self, stream_id=None):
        """Read the per-stream `agent_idle` flag for the TTS substitution gate.

        Returns True iff the agent is currently idle (no open turn — session
        just started, or last non-PAD token on the agent text channel was
        EOS / abort fired). In that state, PAD frames on the agent text
        channel correspond to genuine silence (no turn is open, TTS has
        nothing to render), so silencing them is safe.

        Priority:
          1. Locked read via S2SStreamingState.is_agent_idle() (Caveat 3
             — guards against background FC-async thread writes racing with
             main-thread reads).
          2. Direct `state.agent_idle` attribute access (older state).
          3. Wrapper-level `self._agent_idle` fallback (single-stream).
        Returns True if nothing is set (safe default — silence is benign
        when idle; we only need to avoid it within an open turn).
        """
        if stream_id is not None and self._streaming_state_getter is not None:
            try:
                st = self._streaming_state_getter(stream_id)
                if st is not None:
                    if hasattr(st, "is_agent_idle"):
                        return st.is_agent_idle()
                    if hasattr(st, "agent_idle"):
                        return st.agent_idle
            except Exception:
                pass
        return getattr(self, "_agent_idle", True)

    def _set_agent_idle(self, value, stream_id=None):
        """Write the per-stream `agent_idle` flag (also updates wrapper fallback).

        `value=True`  → agent turn ended (will substitute silence on PAD)
        `value=False` → agent turn opened (will NOT substitute — TTS may
                        still be rendering audio for the in-turn PAD trail).

        Always updates `self._agent_idle` first so single-stream callers
        still see the change; then routes the write to the per-stream
        S2SStreamingState via its locked helpers (`mark_agent_idle()` /
        `mark_agent_active()`) — Caveat 3. Falls back to direct attribute
        assignment for older state objects.
        """
        # Always update the wrapper-level fallback so single-stream usage works.
        self._agent_idle = bool(value)
        if stream_id is not None and self._streaming_state_getter is not None:
            try:
                st = self._streaming_state_getter(stream_id)
                if st is not None:
                    if value and hasattr(st, "mark_agent_idle"):
                        st.mark_agent_idle()
                    elif (not value) and hasattr(st, "mark_agent_active"):
                        st.mark_agent_active()
                    else:
                        # Legacy / older state object — direct assignment.
                        st.agent_idle = bool(value)
            except Exception:
                pass

    def _mark_agent_idle(self, stream_id=None):
        """Convenience hook for "implicit" turn endings (Caveat 2).

        Use whenever an agent turn ends WITHOUT a natural EOS landing on
        the agent text channel. Today that means:
          - user barge-in interrupting agent speech
          - abort_event firing during the FC async loop

        Without this hook, `agent_idle` could stay False forever after
        such an event, and subsequent silent PAD frames would NOT get the
        substitution — bringing back the garbled-syllables bug.

        Equivalent to `_set_agent_idle(True, stream_id)`.
        """
        self._set_agent_idle(True, stream_id)

    def _build_fc_response_tokens(self, tool_response_text: str) -> list:
        """Tokenize a tool response string.

        The text is expected to already contain <TOOL_RESPONSE>...</TOOL_RESPONSE>
        tags (matching the training format), so we tokenize it as-is without
        adding extra wrapper tokens.
        """
        if not tool_response_text:
            return []
        if self._fc_convert_num_to_text:
            tool_response_text = self._convert_tool_response_nums_to_text(tool_response_text)
        return self.tokenizer.text_to_ids(tool_response_text)

    def _apply_fc_state_machine(
        self,
        fc_state: dict,
        function_predicted_token: torch.Tensor,
        gen_text: torch.Tensor,
        gen_function_text: torch.Tensor,
        current_frame_idx: int,
        frame_offset: int,
        predicted_tokens: torch.Tensor,
        function_predicted_tokens: torch.Tensor,
        tool_response_text: str | None,
    ) -> None:
        """Reactive FC state machine: monitor function channel, inject tool responses.

        Mutates *fc_state*, *gen_text*, *gen_function_text*, *predicted_tokens*,
        and *function_predicted_tokens* in-place.

        When fc_async_enabled is True, sets fc_state["trigger_async"] = True on
        SOTC detection so the caller can hand off to _run_fc_async_steps().
        """
        if fc_state is None or self._fc_sotc_id is None:
            return

        pad_id = self.model.stt_model.text_pad_id
        func_tok_val = function_predicted_token.item() if function_predicted_token.dim() == 0 else function_predicted_token[0].item()

        # --- Phase 1: forced token injection (tool response being streamed) ---
        forced = fc_state.get("forced_function_tokens", [])
        if forced:
            override_id = forced.pop(0)
            gen_function_text[:, current_frame_idx] = override_id
            function_predicted_tokens[:, frame_offset] = override_id

            gen_text[:, current_frame_idx] = pad_id
            predicted_tokens[:, frame_offset] = pad_id
            fc_state["injecting_response"] = True
            return

        if fc_state.get("injecting_response", False):
            fc_state["injecting_response"] = False

        # --- Phase 2: monitor natural function head output ---
        if func_tok_val == self._fc_sotc_id:
            fc_state["active"] = True
            fc_state["call_tokens"] = []
            logging.info(f"FC: SOTC detected at frame {current_frame_idx}")
            if self._fc_async_enabled:
                fc_state["trigger_async"] = True
        elif func_tok_val == self._fc_eotc_id and fc_state.get("active", False):
            fc_state["active"] = False
            call_text = self.tokenizer.ids_to_text(fc_state.get("call_tokens", []))
            fc_state.setdefault("completed_calls", []).append(call_text)
            logging.info(f"FC: EOTC detected at frame {current_frame_idx}. Call: {call_text}")

            if tool_response_text:
                response_tokens = self._build_fc_response_tokens(tool_response_text)
                fc_state["forced_function_tokens"] = response_tokens
                logging.info(
                    f"FC: Queued {len(response_tokens)} response tokens for injection"
                )
        elif fc_state.get("active", False) and func_tok_val != pad_id:
            fc_state.setdefault("call_tokens", []).append(func_tok_val)

    def _run_tts_reminder(
        self,
        reminder_tokens: list,
        tts_state: dict,
        tts_audio_output_queue: "queue.Queue | None" = None,
        abort_event: "threading.Event | None" = None,
        request_id: str | None = None,
        stream_id: "int | None" = None,
    ) -> int:
        """Play reminder tokens through TTS only (no LLM steps) during tool API wait.

        Feeds each token in reminder_tokens to EarTTS one step at a time, decodes audio,
        and pushes chunks to tts_audio_output_queue so the client hears "still working"
        audio instead of silence while the tool API call blocks.

        Mutates tts_state in-place (code, past_key_values, codec_cache).
        Returns total audio samples generated so the caller can sleep for playback duration.
        """
        pad_id = self.model.stt_model.text_pad_id
        total_samples = 0
        for tok_id in reminder_tokens:
            if abort_event is not None and abort_event.is_set():
                break
            try:
                _tts_subword = torch.tensor([[tok_id]], device=self.device, dtype=torch.long)
                _tts_sw_mask_val = torch.ones(1, 1, device=self.device, dtype=torch.bool)
                _tts_inputs = {
                    "current_subword_id": _tts_subword,
                    "prev_subword_id": _tts_subword,
                    "current_subword_mask": _tts_sw_mask_val,
                    "prev_audio_tokens": tts_state["code"],
                    "past_key_values": tts_state["past_key_values"],
                    "guidance_enabled": True,
                    "generation_config": self.generation_config,
                    "ignore_eos_flag_stop": True,
                }
                if self.use_vllm_eartts:
                    _tts_inputs["request_id"] = request_id or self.request_id
                _tts_code_new, _tts_pkv_new = self.model.tts_model.infer_codes_one_step(**_tts_inputs)

                # ====== TTS PAD-silence substitution — SITE 2 (_run_tts_reminder) ======
                # INTENTIONALLY NO SUBSTITUTION HERE.
                #
                # Why: this function renders a self-contained ack/reminder message
                # whose token sequence is [BOS, content_tokens, PAD×17, EOS]. The
                # trailing PADs are MID-UTTERANCE for the TTS — they give the TTS
                # codec time to finish rendering the ack content (codec output is
                # autoregressive; one text frame can produce multiple codec frames).
                # If we substituted silence on these PADs, the tail of the ack
                # audio would be chopped off and the user would hear something
                # like "Looking up the wea—" instead of "Looking up the weather."
                #
                # Sites 1 (FC async warmup) and 3 (main loop) DO substitute on
                # PAD frames because those PADs occur when the agent is idle,
                # not mid-ack.
                tts_state["code"] = _tts_code_new
                tts_state["past_key_values"] = _tts_pkv_new

                _tts_codec_cache = tts_state.get("codec_cache")
                if _tts_codec_cache is not None:
                    with fp32_precision(), torch.no_grad():
                        _tts_new_codes = _tts_code_new.unsqueeze(0) if _tts_code_new.dim() == 2 else _tts_code_new
                        if hasattr(self.model.tts_model, '_control_codes'):
                            from nemo.collections.speechlm2.models.duplex_ear_tts import replace_control_speech_codes
                            _tts_new_codes = replace_control_speech_codes(
                                _tts_new_codes,
                                self.model.tts_model._control_codes,
                                getattr(self.model.tts_model, 'codec_silence_tokens', None),
                            )
                        _tts_code_len = torch.tensor([1], dtype=torch.long, device=self.device)
                        _decoded, _ = self.model.tts_model.audio_codec.decode(
                            _tts_new_codes, _tts_code_len, cache=_tts_codec_cache,
                        )
                        _chunk_cpu = _decoded.detach().cpu()
                        total_samples += _chunk_cpu.shape[-1]
                        if tts_audio_output_queue is not None:
                            tts_audio_output_queue.put(_chunk_cpu)
            except Exception as _exc:
                logging.warning("[FC Reminder] TTS step failed: %s", _exc)
                break
        return total_samples

    def infer_one_step(self,
                       audio_input,
                       num_frames_per_chunk,
                       frame_idx,
                       gen_text,
                       audio_toks_buffer,
                       input_embeds_history,
                       dynamic_cache,
                       past_key_values=None,
                       code=None,
                       subword_mask=None,
                       gen_asr_text=None,
                       gen_function_text=None,
                       request_id: Optional[str] = None,
                       perception_cache: Optional[PerceptionCacheState] = None,
                       has_prompt: bool = False,
                       codec_cache=None,
                       rnnt_partial_hypotheses=None,
                       fc_state: dict | None = None,
                       tool_response_text: str | None = None,
                       stream_id: Optional[int] = None):

        # Set up effective request ID for vLLM streaming
        effective_request_id = request_id or self.request_id

        start_time_one_step = time.time()
        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]

        pad_id = self.model.stt_model.text_pad_id
        predicted_tokens = torch.full((batch_size, num_frames_per_chunk), pad_id, dtype=gen_text.dtype, device=gen_text.device)
        asr_predicted_tokens = torch.full((batch_size, num_frames_per_chunk), pad_id, dtype=gen_text.dtype, device=gen_text.device)
        function_predicted_tokens = torch.full((batch_size, num_frames_per_chunk), pad_id, dtype=gen_text.dtype, device=gen_text.device)

        # Do "perception" step outside the for-loop
        start_perception = time.time()
        asr_emb = None

        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            # Cache-aware perception (also returns raw encoder embeddings for RNNT)
            source_encoded, perception_cache, asr_emb = self.perception_cache_mgr.step(
                audio_input=audio_input,
                frame_idx=frame_idx,
                num_frames_per_chunk=num_frames_per_chunk,
                perception_cache=perception_cache,
            )
        else:
            # Standard perception (full buffer processing)
            buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
            source_encoded, _, asr_emb = self.model.stt_model.perception(
                input_signal=audio_input,
                input_signal_length=buffer_len,
                return_encoder_emb=True,
            )

        torch.cuda.synchronize()
        time_perception = time.time() - start_perception
        logging.info(f"Time taken for perception: {time_perception:.3f}s")
        source_encoded = source_encoded.to(self.dtype)
        total_encoded_frames = source_encoded.shape[1]

        # Determine embedding position based on whether we're using cache
        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            # With cache: we get exactly num_frames_per_chunk output frames
            # Use all of them directly
            embedding_position = 0
            newest_frame_index = total_encoded_frames - 1
            base_frame_index = 0
        else:
            # Without cache: Use the second-to-last encoded frame (-2) as the "newest" frame embedding.
            # This is because the model's expects the chunk sizes to be size 10ms, 80ms, 80ms, 80ms, ....,
            # but we pass in always 80ms, 80ms, 80ms....
            # e.g.
            # (1) if we pass in just one 80ms chunk -> the model treats it as 10ms, then 70ms with 10ms silence padding at the end.
            # (2) if we pass 80ms, 80ms -> the model treats it as 10ms, 80ms, 70ms with 10ms silence padding at the end.
            # => we do not want to use the final embedding due to containing silence padding. We want to use the second-to-last embedding.
            embedding_position = -2
            newest_frame_index = total_encoded_frames + embedding_position
            base_frame_index = newest_frame_index - (num_frames_per_chunk - 1)
            base_frame_index = max(base_frame_index, 0)

        new_input_embeds = []
        new_codes_for_decode = []
        for frame_offset in range(num_frames_per_chunk):
            current_frame_idx = frame_idx + frame_offset
            current_frame_index = base_frame_index + frame_offset
            current_frame_index = min(current_frame_index, total_encoded_frames - 1)
            current_frame_embedding = source_encoded[:, current_frame_index:current_frame_index + 1, :]

            has_fc = gen_function_text is not None

            # Build per-channel embeddings, then fuse — mirrors training's fusion_module usage
            user_audio_emb = current_frame_embedding  # [B, 1, D]
            self._last_user_audio_emb = user_audio_emb.detach()

            if current_frame_idx == 0 and not has_prompt:
                agent_text_emb = self._get_bos_embedding().unsqueeze(0)  # [1, 1, H]
                user_text_emb = self._get_asr_bos_embedding().unsqueeze(0) if self._has_asr_head else None
                if has_fc:
                    pad_id = self.model.stt_model.text_pad_id
                    fc_pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                    function_emb = self.model.stt_model.embed_tokens(fc_pad_token).to(dtype=self.dtype).unsqueeze(0)
                else:
                    function_emb = None
            elif current_frame_idx == 0 and has_prompt:
                pad_id = self.model.stt_model.text_pad_id
                pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                agent_text_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype).unsqueeze(0)
                if self._has_asr_head:
                    user_text_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype).unsqueeze(0)
                else:
                    user_text_emb = None
                if has_fc:
                    function_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype).unsqueeze(0)
                else:
                    function_emb = None
            else:
                agent_text_emb = self.model.stt_model.embed_tokens(gen_text[:, current_frame_idx - 1]).unsqueeze(1)
                if self._has_asr_head:
                    user_text_emb = self.model.stt_model.embed_asr_tokens(gen_asr_text[:, current_frame_idx - 1]).unsqueeze(1)
                else:
                    user_text_emb = None
                if has_fc:
                    function_emb = self.model.stt_model.embed_tokens(
                        gen_function_text[:, current_frame_idx - 1]
                    ).to(dtype=self.dtype).unsqueeze(1)
                else:
                    function_emb = None

            current_input_emb = self.fusion_module(
                agent_text_embeds=agent_text_emb,
                user_audio_embeds=user_audio_emb,
                user_text_embeds=user_text_emb,
                function_embeds=function_emb,
            )

            start_stt_model = time.time()

            if use_cache or self.use_vllm_llm:
                if self.use_vllm_llm:
                    # vLLM requires request_id
                    ans = self.model_llm_interface(
                        current_input_emb,
                        request_id=effective_request_id,
                        generated_tokens=gen_text,
                        current_step=current_frame_idx
                    )
                else:
                    ans = self.model_llm_interface(
                        current_input_emb,
                        cache=dynamic_cache,
                        generated_tokens=gen_text,
                        current_step=current_frame_idx
                    )
                dynamic_cache = ans["cache"]
            else:
                new_input_embeds.append(current_input_emb)
                full_input_embeds = torch.cat(input_embeds_history + new_input_embeds, dim=1)
                ans = self.model_llm_interface(
                    full_input_embeds,
                    cache=None,
                    generated_tokens=gen_text,
                    current_step=current_frame_idx
                )

            torch.cuda.synchronize()
            time_stt_model = time.time() - start_stt_model
            logging.info(f"Time taken for stt_model: {time_stt_model:.3f}s")

            predicted_token = ans["predicted_token"]
            asr_predicted_token = ans["asr_predicted_token"]

            gen_text[:, current_frame_idx] = predicted_token
            predicted_tokens[:, frame_offset] = predicted_token

            gen_asr_text[:, current_frame_idx] = asr_predicted_token
            asr_predicted_tokens[:, frame_offset] = asr_predicted_token

            if "function_predicted_token" in ans:
                function_predicted_tokens[:, frame_offset] = ans["function_predicted_token"]
                if gen_function_text is not None:
                    gen_function_text[:, current_frame_idx] = ans["function_predicted_token"]

            # FC state machine: detect SOTC/EOTC on function channel, inject tool response
            if fc_state is not None and gen_function_text is not None:
                self._apply_fc_state_machine(
                    fc_state=fc_state,
                    function_predicted_token=gen_function_text[:, current_frame_idx],
                    gen_text=gen_text,
                    gen_function_text=gen_function_text,
                    current_frame_idx=current_frame_idx,
                    frame_offset=frame_offset,
                    predicted_tokens=predicted_tokens,
                    function_predicted_tokens=function_predicted_tokens,
                    tool_response_text=tool_response_text,
                )

            # Detect when tool-response injection just finished so RNNT allows the post-TC BOS.
            if fc_state is not None:
                _injecting_now = bool(fc_state.get("forced_function_tokens")) or fc_state.get("injecting_response", False)
                if not _injecting_now and fc_state.get("_was_injecting", False):
                    self._post_tc_bos_exempt = True
                    logging.info("[FC] Injection complete → post-TC BOS exempt enabled")
                fc_state["_was_injecting"] = _injecting_now

            # Silence text channel while FC is active — prevents speech tokens from
            # sub-steps after SOTC fires reaching the audio codec before FC async takes over.
            _fc_in_progress = (fc_state is not None and (
                fc_state.get("active", False) or
                fc_state.get("forced_function_tokens") or
                fc_state.get("injecting_response", False)
            ))
            if _fc_in_progress:
                gen_text[:, current_frame_idx] = pad_id
                predicted_tokens[:, frame_offset] = pad_id

            # Apply forced turn-taking. Skip when FC is in progress.
            if not _fc_in_progress:
                self._maybe_apply_forced_turn_taking(current_frame_idx, gen_text, gen_asr_text, rnnt_state=rnnt_partial_hypotheses)

            predicted_tokens[:, frame_offset] = gen_text[:, current_frame_idx]

            # RNNT-based EOU/BOU turn-taking (ga/eou approach: decode then turn-take same frame)
            if rnnt_partial_hypotheses is not None and asr_emb is not None:
                # asr_emb is always [B, T, D]: perception_cache.step() transposes internally
                rnnt_enc = asr_emb
                emb_idx = frame_offset if (self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized()) else current_frame_index
                emb_idx = min(emb_idx, rnnt_enc.shape[1] - 1)
                rnnt_frame = rnnt_enc[:, emb_idx, :]  # [B, D]
                rnnt_partial_hypotheses, _rnnt_is_blank = self._rnnt_step(rnnt_frame, rnnt_partial_hypotheses)
                if not _fc_in_progress:
                    self._apply_rnnt_turn_taking(current_frame_idx, gen_text, _rnnt_is_blank, rnnt_partial_hypotheses)
                predicted_tokens[:, frame_offset] = gen_text[:, current_frame_idx]

            if self.decode_audio:
                current_subword_id = gen_text[:, current_frame_idx].unsqueeze(-1)

                # do one step inference on Duplex TTS model
                if current_frame_idx == 0:
                    if self.first_context_subword_id is None:
                        raise RuntimeError("first_context_subword_id is not initialized. Ensure TTS warmup ran successfully.")
                    prev_subword_id = self.first_context_subword_id
                else:
                    prev_subword_id = gen_text[:, current_frame_idx-1].unsqueeze(-1)

                # create subword_mask
                current_subword_mask = subword_mask[:, current_frame_idx].unsqueeze(-1)

                if self.generation_config is None:
                    raise RuntimeError("generation_config is not initialized. Ensure TTS warmup ran successfully.")

                start_tts_model = time.time()
                inputs = {
                    "current_subword_id": current_subword_id,
                    "prev_subword_id": prev_subword_id,
                    "current_subword_mask": current_subword_mask,
                    "prev_audio_tokens": code,
                    "past_key_values": past_key_values,
                    "guidance_enabled": True,
                    "generation_config": self.generation_config,
                    "ignore_eos_flag_stop": True,
                }
                if self.use_vllm_eartts:
                    inputs["request_id"] = effective_request_id

                code, past_key_values = self.model.tts_model.infer_codes_one_step(
                        **inputs
                )

                torch.cuda.synchronize()
                time_tts_model = time.time() - start_tts_model
                logging.info(f"Time taken for tts_model: {time_tts_model:.3f}s")

                new_codes_for_decode.append(code.clone())
                # Update sliding-window buffer (only needed for fallback decode when codec_cache is off)
                if audio_toks_buffer is not None:
                    audio_toks_buffer = torch.cat([audio_toks_buffer[:, 1:], code], dim=1)

                # now that we've saved audio_toks_buffer for audio decoding purposes,
                # we can potentially overwrite the audio token with silence tokens (for feeding to the audio token predictor)
                if self.model.cfg.get('inference_force_speech_silence_on_eos', None):
                    silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    code = torch.where(
                        current_subword_id.unsqueeze(-1) == self.model.tts_model.text_eos_id,
                        silence_codes,
                        code,
                    )

                # ====== TTS PAD-silence substitution — SITE 3 (main loop) ======
                # This is the main per-frame TTS step (NOT the FC async warmup).
                # The "post-FC silent gap" bug (Pune / NVDA) lives here: LLM emits
                # PAD for many frames after EOTR; without substitution the TTS
                # would hallucinate garbled syllables on those PAD inputs.
                #
                # Substitute only when AGENT IS IDLE (no open turn) — we MUST
                # NOT silence PAD frames inside an open turn (Branch-B training:
                # the LLM emits dense content then a PAD trail until EOS at the
                # audio-end frame; the TTS is still rendering audio during that
                # trail, so silencing would chop in-turn audio).
                #
                # Extended gate: also silence PAD when an open turn has run
                # past its rendering budget (ratio × content tokens emitted
                # since BOS). Env var S2S_TTS_PAD_TAIL_RATIO (default 3; 0=off).
                _ratio = getattr(self, "_tts_pad_tail_ratio", None)
                if _ratio is None:
                    try:
                        _ratio = float(os.environ.get("S2S_TTS_PAD_TAIL_RATIO", "3"))
                    except ValueError:
                        _ratio = 3.0
                    self._tts_pad_tail_ratio = _ratio
                _c = getattr(self, "_tts_in_turn_content", 0)
                _p = getattr(self, "_tts_in_turn_pads", 0)
                _tail_done = (_ratio > 0
                              and not self._get_agent_idle(stream_id)
                              and _p > _ratio * _c)
                if (self.model.cfg.get('inference_force_speech_silence_on_pad', None)
                        and (self._get_agent_idle(stream_id) or _tail_done)):
                    silence_codes = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(code.shape)
                    code = torch.where(
                        current_subword_id.unsqueeze(-1) == self.model.tts_model.text_pad_id,
                        silence_codes,
                        code,
                    )

                # ----- Update the per-stream "agent idle" flag -----
                # State machine:
                #   BOS  → False (agent has started a turn — turn open)
                #   EOS  → True  (agent has finished a turn — agent idle)
                #   PAD or anything else → no change
                #
                # Caveat 4 — this ALSO handles FTT (Forced-Turn-Taking) injections.
                # `_apply_rnnt_turn_taking` and `_maybe_apply_forced_turn_taking`
                # both write BOS/EOS directly into gen_text BEFORE this point in
                # the loop, so `current_subword_id` (read just above the TTS step)
                # observes them transparently. The state flips correctly whether
                # the BOS/EOS came from the LLM's natural prediction OR from FTT
                # injecting it. Do not move this read earlier in the loop or
                # Caveat 4 will silently break.
                try:
                    _csid_val = current_subword_id.flatten()[0].item()
                    if _csid_val == self.model.tts_model.text_bos_id:
                        self._set_agent_idle(False, stream_id)
                        self._tts_in_turn_content = 0
                        self._tts_in_turn_pads = 0
                    elif _csid_val == self.model.tts_model.text_eos_id:
                        self._set_agent_idle(True, stream_id)
                        self._tts_in_turn_content = 0
                        self._tts_in_turn_pads = 0
                    elif _csid_val == self.model.tts_model.text_pad_id:
                        self._tts_in_turn_pads = _p + 1
                    else:
                        self._tts_in_turn_content = _c + 1
                        self._tts_in_turn_pads = 0
                except Exception:
                    # Don't crash the inference loop on any state-update issue;
                    # worst case is a stale flag for one frame.
                    pass

        # exit for-loop & do audio decoding non-autoregressively (if decode_audio is True)
        if self.decode_audio:
            samples_per_audio_output_frame = self._samples_per_audio_output_frame()
            logging.debug(f"\nDecoding audio for {frame_idx}-th frame  ({num_frames_per_chunk=})")

            start_time_decode = time.time()
            with fp32_precision(), torch.no_grad():
                if codec_cache is not None and new_codes_for_decode:
                    # Incremental decode: feed only the num_frames_per_chunk new tokens
                    # to the codec. CausalConv1dCache maintains all necessary ConvNeXt
                    # and ISTFT overlap state from prior calls, so no history buffer
                    # is needed — this replaces the sliding-window approach entirely.
                    new_codes_tensor = torch.cat(new_codes_for_decode, dim=1)
                    if hasattr(self.model.tts_model, '_control_codes'):
                        from nemo.collections.speechlm2.models.duplex_ear_tts import replace_control_speech_codes
                        new_codes_tensor = replace_control_speech_codes(
                            new_codes_tensor,
                            self.model.tts_model._control_codes,
                            getattr(self.model.tts_model, 'codec_silence_tokens', None),
                        )
                    new_code_len = torch.tensor(
                        [new_codes_tensor.shape[1]], dtype=torch.long, device=self.device
                    )
                    decoded_audio_new, _ = self.model.tts_model.audio_codec.decode(
                        new_codes_tensor, new_code_len, cache=codec_cache,
                    )
                    logging.debug(f"   Incremental decode: {new_codes_tensor.shape[1]} new tokens -> {decoded_audio_new.shape}")
                else:
                    # Fallback: full-buffer sliding-window decode (original behavior)
                    len_audio_toks_buffer = torch.tensor(
                        [self.codec_token_history_size], dtype=torch.long, device=self.device
                    )
                    decoded_audio, decoded_audio_len = self.model.tts_model.audio_codec.decode(
                        audio_toks_buffer, len_audio_toks_buffer,
                    )
                    decoded_audio_new = decoded_audio[:, :, -samples_per_audio_output_frame * num_frames_per_chunk:]
                    logging.debug(f"   Sliding-window decode: extracted {decoded_audio_new.shape} from {decoded_audio.shape}")

            torch.cuda.synchronize()
            time_audio_codec = time.time() - start_time_decode
            logging.info(f"Time taken for audio_codec: {time_audio_codec:.3f}s")

        else:
            audio_toks_buffer = None
            decoded_audio_new = None
            time_tts_model = 0
            time_audio_codec = 0

        # Convert new text tokens to string via tokens_to_text (convert_tokens_to_string)
        # so byte-level BPE is decoded properly (e.g. "Ã©" → "é") and leading spaces
        # from Ġ-prefixed tokens are preserved for correct concatenation of incremental
        # chunks: " Musée" + " National" → " Musée National".
        # NOTE: multi-byte UTF-8 characters whose BPE tokens span two frames will show
        # as replacement chars (�) because each frame is decoded independently. A proper
        # fix would require an incremental UTF-8 decoder that buffers incomplete trailing
        # bytes across frames.
        predicted_text_strs = []
        for predicted_tok_ids_b in predicted_tokens:
            predicted_tok_ids_b = predicted_tok_ids_b.tolist()
            predicted_toks_b = self.tokenizer.ids_to_tokens(predicted_tok_ids_b)
            predicted_toks_b = [tok for tok in predicted_toks_b if tok != '<SPECIAL_12>']
            predicted_text_strs.append(self.tokenizer.tokens_to_text(predicted_toks_b))

        # convert new ASR tokens to string
        asr_predicted_text_strs = []
        for asr_predicted_tok_ids_b in asr_predicted_tokens:
            asr_predicted_tok_ids_b = asr_predicted_tok_ids_b.tolist()
            asr_predicted_toks_b = self.tokenizer.ids_to_tokens(asr_predicted_tok_ids_b)
            asr_predicted_toks_b = [tok for tok in asr_predicted_toks_b if tok != '<SPECIAL_12>']
            asr_predicted_text_strs.append(self.tokenizer.tokens_to_text(asr_predicted_toks_b))

        logging.info(f'frame {frame_idx}: USER\'s asr_predicted_text_strs: {asr_predicted_text_strs}')
        logging.info(f'frame {frame_idx}: --------------------------------AGENT\'s predicted_text_strs: {predicted_text_strs}')
        if self.model.stt_model.function_head is not None:
            fc_tok_ids = function_predicted_tokens[0].tolist()
            pad_id = self.model.stt_model.text_pad_id
            non_pad = [(i, t) for i, t in enumerate(fc_tok_ids) if t != pad_id]
            if non_pad:
                fc_tok_strs = self.tokenizer.ids_to_tokens([t for _, t in non_pad])
                logging.info(f'frame {frame_idx}: FC_HEAD tokens (non-pad): {list(zip([i for i,_ in non_pad], fc_tok_strs))}')
            else:
                logging.info(f'frame {frame_idx}: FC_HEAD tokens: all pad')

        torch.cuda.synchronize()

        time_for_one_step = time.time() - start_time_one_step
        logging.info(f'frame {frame_idx}: Time taken for one step: {time_for_one_step:.3f}s')

        result = {
            'predicted_text_tokens': predicted_tokens,
            'asr_predicted_text_tokens': asr_predicted_tokens,
            'audio_toks_buffer': audio_toks_buffer,
            'decoded_audio_new': decoded_audio_new,
            'predicted_text_strs': predicted_text_strs,
            'asr_predicted_text_strs': asr_predicted_text_strs,
            'input_embeds_history': input_embeds_history + new_input_embeds if not use_cache else input_embeds_history,
            'dynamic_cache': dynamic_cache if use_cache else None,
            'past_key_values': past_key_values,
            'code': code,
            'perception_cache': perception_cache,
            'codec_cache': codec_cache,
            'rnnt_partial_hypotheses': rnnt_partial_hypotheses,
        }
        if self.model.stt_model.function_head is not None:
            result['function_predicted_text_tokens'] = function_predicted_tokens
            fc_text_strs = []
            pad_id = self.model.stt_model.text_pad_id
            for fc_tok_ids_b in function_predicted_tokens:
                fc_ids = [t for t in fc_tok_ids_b.tolist() if t != pad_id]
                fc_text_strs.append(self.tokenizer.ids_to_text(fc_ids) if fc_ids else "")
            result['function_predicted_text_strs'] = fc_text_strs
        if fc_state is not None:
            result['fc_state'] = fc_state
        return result

    def abort_request(self, request_id: Optional[str]) -> bool:
        """
        Abort an in-flight vLLM streaming request if the backend supports it.
        """
        if not request_id:
            return False

        success = False

        # Abort LLM if applicable
        if self.use_vllm_llm:
            abort_fn = getattr(self.model_llm_interface, "abort_request", None)
            if callable(abort_fn):
                try:
                    if abort_fn(request_id):
                        success = True
                    logging.info(f"Aborted LLM request {request_id} successfully.")
                except Exception as exc:
                    logging.warning(f"Failed to abort LLM request {request_id}: {exc}")

        # Abort EarTTS if applicable
        if self.use_vllm_eartts:
            abort_fn = getattr(self.model.tts_model.tts_model, "abort_request", None)
            if callable(abort_fn):
                try:
                    if abort_fn(request_id):
                        success = True
                    logging.info(f"Aborted EarTTS request {request_id} successfully.")
                except Exception as exc:
                    logging.warning(f"Failed to abort EarTTS request {request_id}: {exc}")

        return success


    def _maybe_apply_forced_turn_taking(self, t, gen_text, gen_asr, rnnt_state=None):
        """Apply forced turn-taking rules based on ASR channel tokens.

        First turn uses a shorter silence window (force_turn_taking_pad_window_first_turn)
        so that a short utterance like "Hello" + pause still triggers an agent response.
        """
        if not self.model_cfg.get("force_turn_taking", False):
            return

        threshold        = self.model_cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.model_cfg.get("force_turn_taking_pad_window", 25)
        # Shorter window on the very first turn; falls back to pad_window_steps if unset.
        pad_window_first = self.model_cfg.get("force_turn_taking_pad_window_first_turn", pad_window_steps)

        bos_id      = self.model.stt_model.text_bos_id
        eos_id      = self.model.stt_model.text_eos_id
        pad_id      = self.model.stt_model.text_pad_id
        user_bos_id = self.model.stt_model.user_bos_id

        B = gen_text.size(0)

        for batch_idx in range(B):
            lookback_start    = max(0, t - threshold)
            agent_text_window = gen_text[batch_idx, lookback_start:t]
            current_asr_token = gen_asr[batch_idx, t]

            # First turn = agent has never written a BOS token yet
            is_first_turn = not (gen_text[batch_idx, 0:t] == bos_id).any().item()
            active_window = pad_window_first if is_first_turn else pad_window_steps

            if t < active_window:
                continue

            pad_lookback_start = t - active_window
            asr_recent_tokens  = gen_asr[batch_idx, pad_lookback_start:t]
            has_pad_window = (asr_recent_tokens == pad_id).all() if len(asr_recent_tokens) > 0 else False

            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = gen_asr[batch_idx, pad_lookback_start - 1]
                has_pad_window = (token_before_window != pad_id) and (token_before_window != user_bos_id)
            elif has_pad_window and pad_lookback_start == 0:
                has_pad_window = False

            if has_pad_window:
                if not (agent_text_window == bos_id).any():
                    gen_text[batch_idx, t] = bos_id
                    if rnnt_state is not None and 'forced_bos' in rnnt_state:
                        rnnt_state['forced_bos'][batch_idx] = True
                    logging.info(
                        f"Forced turn-taking at frame {t}: inserted agent BOS "
                        f"(pad_window={active_window}, first_turn={is_first_turn})"
                    )

            # ASR BOS → insert agent EOS if not present in window
            elif current_asr_token == user_bos_id:
                if not (agent_text_window == eos_id).any():
                    gen_text[batch_idx, t] = eos_id
                    logging.info(f"Forced turn-taking at frame {t}: inserted agent EOS (reason: user started speaking)")

    def _rnnt_init_state(self, B: int, device) -> dict:
        """Initialize RNNT streaming state for EOU/BOU detection."""
        return {
            'pred_out':         None,
            'pred_hidden':      None,
            'blank_count':      torch.zeros(B, dtype=torch.long, device=device),
            'nonblank_consec':  torch.zeros(B, dtype=torch.long, device=device),
            'nonblank_total':   torch.zeros(B, dtype=torch.long, device=device),
            'speech_confirmed': torch.zeros(B, dtype=torch.bool, device=device),
            'agent_speaking':   torch.zeros(B, dtype=torch.bool, device=device),
            'first_turn':       torch.ones(B, dtype=torch.bool, device=device),
            'rolling_density':  torch.zeros(B, dtype=torch.float32, device=device),
            'post_eos_fired':   torch.zeros(B, dtype=torch.bool, device=device),
            'forced_bos':       torch.zeros(B, dtype=torch.bool, device=device),
            'y_sequence':       [],
            '_punct_word_acc':  [],    # non-punct token IDs decoded since last punct
            '_punct_bias_val':  0.0,   # accumulated logit boost; increments each blank frame after words
        }

    @torch.no_grad()
    def _rnnt_step(self, encoder_frame: torch.Tensor, rnnt_state: dict):
        """Run one RNNT greedy decode step. Returns (new_state, is_blank [B])."""
        decoder  = getattr(self.model.stt_model, '_rnnt_decoder', None)
        joint    = getattr(self.model.stt_model, '_rnnt_joint', None)
        blank_id = getattr(self.model.stt_model, '_rnnt_blank_id', 1024)
        if decoder is None or joint is None:
            is_blank = torch.ones(encoder_frame.shape[0], dtype=torch.bool, device=encoder_frame.device)
            return rnnt_state, is_blank

        B = encoder_frame.shape[0]
        rnnt_dtype = next(joint.parameters()).dtype
        f = encoder_frame.to(rnnt_dtype).unsqueeze(1)  # [B, 1, D]

        pred_out    = rnnt_state['pred_out']
        pred_hidden = rnnt_state['pred_hidden']
        if pred_out is None:
            pred_out, pred_hidden = decoder.predict(y=None, state=None, add_sos=True, batch_size=B)
        if pred_out.dim() == 3 and pred_out.shape[1] > 1:
            pred_out = pred_out[:, -1:, :]

        logits = joint.joint(f, pred_out)
        tokens = logits.squeeze(1).squeeze(1).argmax(-1)
        # Frame-level blank drives turn-taking counters (blank_count, nonblank_consec) — must stay as first prediction.
        is_blank = (tokens == blank_id)

        # Label loop: keep emitting tokens from this frame until blank or max_symbols, mirroring NeMo master.
        # Predictor advances through every emission; is_blank above is NOT updated so turn-taking is unchanged.
        _cur_pred_out, _cur_pred_hidden = pred_out, pred_hidden
        _loop_tokens, _loop_is_blank = tokens, is_blank
        _emitted: list = []
        _symbols = 0
        # Local copy of bias so we can zero it after the first punct emission
        # within this frame — prevents the same bias from firing repeatedly
        # across multiple label-loop iterations in a single encoder step.
        _pbv = float(rnnt_state.get('_punct_bias_val', 0.0))
        while not _loop_is_blank.all() and _symbols < self._rnnt_max_symbols:
            if B == 1 and not _loop_is_blank[0]:
                _tok = _loop_tokens[0].item()
                _emitted.append(_tok)
                if _tok in self._rnnt_punct_ids_set:
                    _pbv = 0.0   # consumed — no more punct bias this frame
            _y = torch.where(_loop_is_blank, torch.full_like(_loop_tokens, blank_id), _loop_tokens).unsqueeze(1)
            try:
                _np_out, _np_hid = decoder.predict(y=_y, state=_cur_pred_hidden, add_sos=False, batch_size=B)
                if B == 1:
                    if not _loop_is_blank[0]:
                        _cur_pred_out, _cur_pred_hidden = _np_out, _np_hid
                else:
                    _mask = _loop_is_blank.view(B, 1, 1).expand_as(_cur_pred_out)
                    _cur_pred_out = torch.where(_mask, _cur_pred_out, _np_out)
                    _cur_pred_hidden = _np_hid
            except Exception as _e:
                logging.warning(f"RNNT label loop predictor step skipped: {_e}")
                break
            _symbols += 1
            _loop_logits = joint.joint(f, _cur_pred_out)
            _loop_scores = _loop_logits.squeeze(1).squeeze(1)
            # Punct bias: boost punct token logits after words with no punctuation yet.
            # Operates only in the label loop — is_blank (turn-taking) is unaffected.
            if _pbv > 0.0 and self._rnnt_punct_ids:
                for _pid in self._rnnt_punct_ids:
                    _loop_scores[0, _pid] = _loop_scores[0, _pid] + _pbv
            _loop_tokens = _loop_scores.argmax(-1)
            _loop_is_blank = (_loop_tokens == blank_id)
        new_pred_out, new_pred_hidden = _cur_pred_out, _cur_pred_hidden

        # Update punctuation bias state based on what was emitted this frame.
        _pw_acc = list(rnnt_state.get('_punct_word_acc', []))
        _pb_val = float(rnnt_state.get('_punct_bias_val', 0.0))
        if _emitted:
            _emitted_punct    = [t for t in _emitted if t in self._rnnt_punct_ids_set]
            _emitted_nonpunct = [t for t in _emitted if t not in self._rnnt_punct_ids_set]
            if _emitted_punct:
                _pw_acc, _pb_val = [], 0.0           # punct decoded — reset
            elif _emitted_nonpunct:
                _pw_acc.extend(_emitted_nonpunct)    # new words — add to buffer, reset bias
                _pb_val = 0.0
        elif _pw_acc:                                # blank frame after words — increment bias
            _pb_val += self._rnnt_punct_bias_increment

        density_alpha = 0.1
        is_speech_float = (~is_blank).float()
        if rnnt_state.get('agent_speaking') is not None and rnnt_state['agent_speaking'].any():
            new_density = rnnt_state['rolling_density']
        else:
            new_density = density_alpha * is_speech_float + (1.0 - density_alpha) * rnnt_state['rolling_density']

        new_state = {
            'pred_out':         new_pred_out,
            'pred_hidden':      new_pred_hidden,
            'blank_count':      torch.where(is_blank,
                                            rnnt_state['blank_count'] + 1,
                                            torch.zeros_like(rnnt_state['blank_count'])),
            'nonblank_consec':  torch.where(is_blank,
                                            torch.zeros_like(rnnt_state['nonblank_consec']),
                                            rnnt_state['nonblank_consec'] + 1),
            'nonblank_total':   torch.where(is_blank,
                                            rnnt_state['nonblank_total'],
                                            rnnt_state['nonblank_total'] + 1),
            'speech_confirmed': rnnt_state['speech_confirmed'],
            'agent_speaking':   rnnt_state['agent_speaking'],
            'first_turn':       rnnt_state['first_turn'],
            'rolling_density':  new_density,
            'post_eos_fired':   rnnt_state['post_eos_fired'],
            'y_sequence':       rnnt_state.get('y_sequence', []) + _emitted,
            '_punct_word_acc':  _pw_acc,
            '_punct_bias_val':  _pb_val,
        }
        # Carry forward dynamically-added fields (_turn_text_tokens, _agent_talking_frames,
        # forced_bos, etc.) that _apply_rnnt_turn_taking sets in-place but are absent from
        # the fixed schema above. Without this, MaxResponse and TTSRatioCap counters reset
        # to 0 every frame and those safety caps never fire.
        for _k in rnnt_state:
            if _k not in new_state:
                new_state[_k] = rnnt_state[_k]
        return new_state, is_blank

    def _rnnt_decode_text(self, y_sequence: list) -> str:
        """Decode accumulated RNNT token IDs to text via rnnt_joint.vocabulary (SentencePiece, ▁=word boundary)."""
        vocab = getattr(getattr(self.model.stt_model, 'rnnt_joint', None), 'vocabulary', None)
        if not vocab or not y_sequence:
            return ""
        return "".join(vocab[t] for t in y_sequence if 0 <= t < len(vocab)).replace("▁", " ").strip()

    def _apply_rnnt_turn_taking(self, t: int, gen_text: torch.Tensor,
                                is_blank: torch.Tensor, rnnt_state: dict) -> None:
        """Apply RNNT blank/non-blank counts to trigger agent BOS (EOU) or EOS (BOU)."""
        if not self.model_cfg.get("force_turn_taking", False):
            return

        asr_eou         = int(self.model_cfg.get("rnnt_eou_frames",
                              self.model_cfg.get("asr_eou", 15)))
        asr_min_speech  = int(self.model_cfg.get("asr_min_speech_frames", 3))
        user_bos_frames = int(self.model_cfg.get("user_bos_frames",
                              self.model_cfg.get("asr_bou",
                              self.model_cfg.get("rnnt_bou_frames", 4))))
        threshold = int(self.model_cfg.get("force_turn_taking_threshold", 40))
        bos_id    = self.model.stt_model.text_bos_id
        eos_id    = self.model.stt_model.text_eos_id

        B = gen_text.size(0)
        for b in range(B):
            # Redirect token injection: when max response duration was exceeded,
            # inject pre-tokenized "How can I help you?" tokens frame-by-frame.
            # These run before normal RNNT logic so TTS picks up the injected token.
            if self._redirect_tokens_queue:
                next_tok = self._redirect_tokens_queue.pop(0)
                gen_text[b, t] = next_tok
                if next_tok == bos_id:
                    rnnt_state['agent_speaking'][b] = True
                elif next_tok == eos_id:
                    rnnt_state['agent_speaking'][b] = False
                    self._agent_eos_just_fired = True
                continue  # skip normal RNNT logic this frame

            lookback_start = max(0, t - threshold)
            agent_window   = gen_text[b, lookback_start:t]

            blank_cnt        = rnnt_state['blank_count'][b].item()
            nonblank_cnt     = rnnt_state['nonblank_consec'][b].item()
            nonblank_total   = rnnt_state['nonblank_total'][b].item()
            speech_confirmed = rnnt_state['speech_confirmed'][b].item()
            first_turn       = rnnt_state['first_turn'][b].item()
            rolling_density  = rnnt_state['rolling_density'][b].item()

            if first_turn:
                effective_min_speech = int(self.model_cfg.get("asr_min_speech_frames_first_turn", 2))
                effective_eou        = int(self.model_cfg.get("asr_eou_first_turn", asr_eou))
            else:
                effective_min_speech = asr_min_speech
                effective_eou        = asr_eou

            density_threshold = float(self.model_cfg.get("density_speech_threshold", 0.15))
            density_low_min   = int(self.model_cfg.get("density_low_min_speech", 6))
            density_adapted   = False
            if not first_turn and rolling_density < density_threshold and rolling_density > 0.0:
                effective_min_speech = density_low_min
                density_adapted = True

            agent_speaking = rnnt_state['agent_speaking'][b].item()
            current_tok = gen_text[b, t].item()

            # Self-play suppression: block LLM-native BOS when agent finished speaking
            # and user hasn't started. MUST run before the BOS-detection block below
            # (which sets agent_speaking=True, making a post-detection check unreachable).
            # Forced-turn-taking BOS is exempted via the 'forced_bos' flag set by
            # _maybe_apply_forced_turn_taking. RNNT-EOU BOS is safe: it is injected
            # internally (token was PAD at function entry) and returns before this path.
            _forced_bos_flags = rnnt_state.get('forced_bos')
            _is_forced_bos = (_forced_bos_flags is not None and _forced_bos_flags[b].item())
            if _forced_bos_flags is not None:
                rnnt_state['forced_bos'][b] = False  # consume flag each frame
            if (not agent_speaking and not first_turn and not speech_confirmed
                    and current_tok == bos_id and not _is_forced_bos
                    and self.model_cfg.get("rnnt_self_play_suppression", False)):
                if self._post_tc_bos_exempt:
                    self._post_tc_bos_exempt = False
                    logging.info(f"RNNT post-TC BOS exempt at t={t}: allowing agent BOS after tool call")
                    # fall through — do not suppress
                else:
                    gen_text[b, t] = self.model.stt_model.text_pad_id
                    logging.debug(f"RNNT self-play suppression t={t}: LLM BOS suppressed (no user speech)")
                    return

            if (agent_window == bos_id).any() or current_tok == bos_id:
                agent_speaking = True
                rnnt_state['agent_speaking'][b] = True
                rnnt_state['first_turn'][b] = False
                first_turn = False
                if current_tok == bos_id:
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    speech_confirmed = False
                    nonblank_total = 0
                    # Reset token counter for the new agent turn.
                    if '_turn_text_tokens' not in rnnt_state:
                        rnnt_state['_turn_text_tokens'] = {}
                    rnnt_state['_turn_text_tokens'][b] = 0
            if (agent_window == eos_id).any() or current_tok == eos_id:
                agent_speaking = False
                rnnt_state['agent_speaking'][b] = False
                rnnt_state['speech_confirmed'][b] = False
                rnnt_state['nonblank_total'][b] = 0
                speech_confirmed = False
                nonblank_total = 0

            noise_reset_frames = int(self.model_cfg.get("nonblank_reset_after_silence", 10))
            if blank_cnt >= noise_reset_frames and not speech_confirmed and not agent_speaking:
                rnnt_state['nonblank_total'][b] = 0
                nonblank_total = 0

            if (nonblank_cnt >= effective_min_speech or nonblank_total >= effective_min_speech) and not agent_speaking:
                rnnt_state['speech_confirmed'][b] = True
                speech_confirmed = True

            first_turn_fallback = int(self.model_cfg.get("first_turn_fallback_frames", 50))
            if first_turn and t >= first_turn_fallback and speech_confirmed and not (agent_window == bos_id).any():
                gen_text[b, t] = bos_id
                rnnt_state['speech_confirmed'][b] = False
                rnnt_state['nonblank_total'][b] = 0
                rnnt_state['agent_speaking'][b] = True
                rnnt_state['first_turn'][b] = False
                logging.info(f"RNNT first-turn fallback t={t}: agent BOS (no EOU in {first_turn_fallback} frames)")
                return

            # EOU: N blank frames after confirmed speech → inject agent BOS
            if blank_cnt >= effective_eou and speech_confirmed and not agent_speaking:
                if not (agent_window == bos_id).any() and current_tok != bos_id:
                    gen_text[b, t] = bos_id
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    rnnt_state['agent_speaking'][b] = True
                    rnnt_state['first_turn'][b] = False
                    logging.info(
                        f"RNNT EOU t={t}: agent BOS (blank_cnt={blank_cnt}, "
                        f"nonblank_total={nonblank_total}, density={rolling_density:.3f})"
                    )
                else:
                    logging.debug(f"RNNT EOU suppressed at t={t}: LLM already has BOS in window")
                return

            force_eos_frames = int(self.model_cfg.get("force_eos_after_frames", 0))
            if (force_eos_frames > 0 and agent_speaking and not speech_confirmed
                    and not first_turn and not (agent_window == eos_id).any()):
                if blank_cnt >= force_eos_frames:
                    gen_text[b, t] = eos_id
                    rnnt_state['agent_speaking'][b] = False
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    self._agent_eos_just_fired = True
                    logging.info(f"RNNT force-EOS t={t}: agent stopped (no user speech in {blank_cnt} blanks)")
                    return

            post_eos_fallback = int(self.model_cfg.get("post_eos_fallback_frames", 0))
            if (post_eos_fallback > 0 and not first_turn and not agent_speaking
                    and not speech_confirmed and not rnnt_state['post_eos_fired'][b]):
                if blank_cnt >= post_eos_fallback:
                    gen_text[b, t] = bos_id
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    rnnt_state['agent_speaking'][b] = True
                    rnnt_state['first_turn'][b] = False
                    rnnt_state['post_eos_fired'][b] = True
                    logging.info(f"RNNT post-EOS fallback t={t}: BOS (blank_cnt={blank_cnt})")
                    return

            if (post_eos_fallback > 0 and not first_turn and not speech_confirmed
                    and rnnt_state['post_eos_fired'][b]):
                _pad_id = self.model.stt_model.text_pad_id
                if gen_text[b, t].item() == bos_id:
                    gen_text[b, t] = _pad_id
                    rnnt_state['agent_speaking'][b] = False
                    return

            # Max agent response duration: force EOS after too many continuous talking frames.
            if (self._max_agent_response_frames > 0 and agent_speaking
                    and current_tok != eos_id
                    and not (agent_window == eos_id).any()):
                _talking_key = '_agent_talking_frames'
                if _talking_key not in rnnt_state:
                    rnnt_state[_talking_key] = {}
                _cnt = rnnt_state[_talking_key].get(b, 0) + 1
                rnnt_state[_talking_key][b] = _cnt
                if _cnt >= self._max_agent_response_frames:
                    gen_text[b, t] = eos_id
                    rnnt_state['agent_speaking'][b] = False
                    rnnt_state[_talking_key][b] = 0
                    self._agent_eos_just_fired = True
                    if self._max_response_redirect_tokens:
                        self._redirect_tokens_queue = list(self._max_response_redirect_tokens)
                    logging.info(
                        "[MaxResponse] Agent talking %d frames (>= max %d) → forced EOS + redirect queued",
                        _cnt, self._max_agent_response_frames,
                    )
                    return
            elif not agent_speaking:
                _talking_key = '_agent_talking_frames'
                if _talking_key in rnnt_state:
                    rnnt_state[_talking_key][b] = 0

            # TTS token-ratio cap (Edresson's fix): force EOS after ratio × text_token_count
            # frames to prevent TTS hallucination when EOS is missing or delayed.
            # Ratio and min_tokens are configurable; 0 ratio = disabled.
            _tts_ratio = float(self.model_cfg.get("tts_text_token_ratio_cap", 6.0))
            _tts_min_tokens = int(self.model_cfg.get("tts_text_token_min", 5))
            if (_tts_ratio > 0 and agent_speaking
                    and current_tok != eos_id
                    and not (agent_window == eos_id).any()):
                _pad_id_r = self.model.stt_model.text_pad_id
                # Count non-PAD, non-BOS, non-EOS text tokens produced this turn.
                if '_turn_text_tokens' not in rnnt_state:
                    rnnt_state['_turn_text_tokens'] = {}
                if current_tok not in (bos_id, eos_id, _pad_id_r):
                    rnnt_state['_turn_text_tokens'][b] = rnnt_state['_turn_text_tokens'].get(b, 0) + 1
                _txt_cnt = rnnt_state['_turn_text_tokens'].get(b, 0)
                _talking_frames = rnnt_state.get('_agent_talking_frames', {}).get(b, 0)
                if (_txt_cnt >= _tts_min_tokens
                        and _talking_frames >= _tts_ratio * _txt_cnt):
                    gen_text[b, t] = eos_id
                    rnnt_state['agent_speaking'][b] = False
                    rnnt_state['_turn_text_tokens'][b] = 0
                    if '_agent_talking_frames' in rnnt_state:
                        rnnt_state['_agent_talking_frames'][b] = 0
                    self._agent_eos_just_fired = True
                    logging.info(
                        "[TTSRatioCap] Forced EOS: %d talking frames >= %.1f × %d text tokens",
                        _talking_frames, _tts_ratio, _txt_cnt,
                    )
                    return

            # Barge-in: N consecutive non-blank frames while agent speaking → EOS
            if nonblank_cnt >= user_bos_frames and agent_speaking:
                if not (agent_window == eos_id).any() and current_tok != eos_id:
                    gen_text[b, t] = eos_id
                    rnnt_state['nonblank_consec'][b] = 0
                    rnnt_state['nonblank_total'][b] = 0
                    rnnt_state['agent_speaking'][b] = False
                    self._redirect_tokens_queue = []  # cancel pending redirect if user barges in
                    self._agent_eos_just_fired = True
                    logging.info(
                        f"RNNT barge-in t={t}: agent EOS (nonblank_consec={nonblank_cnt})"
                    )

    def _reset_rnnt_turn_taking_state(self):
        self._agent_eos_just_fired = False
        self._redirect_tokens_queue = []
        self._post_tc_bos_exempt = False

    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, num_frames_per_chunk: int = None, request_id: Optional[str] = None, pad_audio_to_sec: Optional[float] = None, pad_silence_ratio: Optional[float] = None, pad_audio_by_sec: Optional[float] = None, system_prompt: Optional[str] = None, tool_executor_fn: Optional[callable] = None):
        """
        Perform realtime streaming inference simulating microphone capture.

        Args:
            audio_path: Path to input audio file (simulates microphone input)
            num_frames_per_chunk: Number of frames to process per inference step (default: 1)
            request_id: Optional request ID for vLLM streaming
            pad_audio_to_sec: Optional duration to pad audio to (in seconds)
            pad_silence_ratio: Optional ratio of original duration to append as silence (e.g. 0.2 = 20%)
            pad_audio_by_sec: Optional fixed number of extra seconds of silence to append
            system_prompt: Optional system prompt to provide context to the model
            tool_executor_fn: Optional callable (str) -> str for two-phase async FC.
                When fc_async_two_phase is enabled, this function is called with the
                tool call text after EOTC. It should execute the tool and return the
                response string. If None and two-phase is enabled, the async loop
                exits at EOTC without response injection.

        Returns:
            Dictionary with 'text', 'tokens_text', 'tokens_audio', 'audio', 'audio_len', 'system_prompt'
        """
        # Use provided value or default
        if num_frames_per_chunk is None:
            num_frames_per_chunk = DEFAULT_NUM_FRAMES_PER_CHUNK
        if num_frames_per_chunk < 1:
            raise ValueError("num_frames_per_chunk must be at least 1")
        start_time = time.time()

        logging.info("\n" + "=" * 70)
        logging.info("STARTING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)

        # Set up request ID for vLLM streaming
        stream_request_id = request_id or self.request_id

        buffer_size_frames = int(self.model_cfg.get("buffer_size_frames", DEFAULT_BUFFER_SIZE_FRAMES))
        buffer_size_samples = buffer_size_frames * FRAME_SIZE_SAMPLES
        if num_frames_per_chunk > buffer_size_frames:
            raise ValueError(
                f"num_frames_per_chunk ({num_frames_per_chunk}) must be "
                f"less than or equal to buffer_size_frames ({buffer_size_frames})."
            )

        att_context_size = self.model.stt_model.perception.encoder._cfg.att_context_size
        if self.use_perception_cache:
            min_buffer = num_frames_per_chunk * (att_context_size[1] + 1) + 2
            reason = (
                f"must be >= num_frames_per_chunk * (att_context_size[1] + 1) + 2 = "
                f"{num_frames_per_chunk} * ({att_context_size[1]} + 1) + 2 = {min_buffer} "
                f"when using perception cache (+2 to minimize windowing artifacts)"
            )
        else:
            min_buffer = att_context_size[0] + att_context_size[1] + 1
            reason = (
                f"must be >= att_context_size[0] + att_context_size[1] + 1 = "
                f"{att_context_size[0]} + {att_context_size[1]} + 1 = {min_buffer} "
                f"without perception cache"
            )
        if buffer_size_frames < min_buffer:
            raise ValueError(
                f"buffer_size_frames ({buffer_size_frames}) is too small: {reason}."
            )
        if self.decode_audio and not self.use_codec_cache and num_frames_per_chunk > self.codec_token_history_size:
            raise ValueError(
                f"num_frames_per_chunk ({num_frames_per_chunk}) must be "
                f"<= codec_token_history_size ({self.codec_token_history_size}) when decode_audio=True "
                f"and use_codec_cache=False. "
                f"Either reduce num_frames_per_chunk, increase codec_token_history_size, or enable use_codec_cache."
            )
        logging.info(f"Buffer size: {buffer_size_frames} frames ({buffer_size_frames * FRAME_SIZE_SEC}s)")
        logging.info(f"Frames per inference step: {num_frames_per_chunk}")

        # Load audio file (simulating microphone stream)
        logging.info(f"Loading audio file: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio_signal)
        total_duration = total_samples / SAMPLE_RATE

        logging.info(f"   Total duration: {total_duration:.2f}s")
        logging.info(f"   Total samples: {total_samples}")

        # Optionally pad audio (at most one of these is set; enforced by caller)
        if pad_audio_to_sec is not None and pad_audio_to_sec > total_duration:
            target_samples = int(pad_audio_to_sec * SAMPLE_RATE)
            audio_signal = np.pad(audio_signal, (0, target_samples - total_samples), mode='constant')
            total_samples = len(audio_signal)
            logging.info(f"   Padded to {pad_audio_to_sec:.2f}s ({total_samples} samples)")
        elif pad_silence_ratio is not None:
            extra_samples = int(total_duration * pad_silence_ratio * SAMPLE_RATE)
            audio_signal = np.pad(audio_signal, (0, extra_samples), mode='constant')
            total_samples = len(audio_signal)
            logging.info(f"   Padded with {pad_silence_ratio*100:.1f}% extra silence ({extra_samples} samples)")
        elif pad_audio_by_sec is not None:
            extra_samples = int(pad_audio_by_sec * SAMPLE_RATE)
            audio_signal = np.pad(audio_signal, (0, extra_samples), mode='constant')
            total_samples = len(audio_signal)
            logging.info(f"   Padded with {pad_audio_by_sec:.2f}s extra silence ({extra_samples} samples)")

        # derive num_inference_steps
        total_frames_maybe = int(np.ceil(total_samples / FRAME_SIZE_SAMPLES)) # "maybe" because we might need to add padding
        num_inference_steps = (total_frames_maybe // num_frames_per_chunk)
        if total_frames_maybe % num_frames_per_chunk != 0:
            num_inference_steps += 1
        total_frames = num_inference_steps * num_frames_per_chunk

        # pad audio signal so that it is divisible by num_inference_steps
        padded_total_samples = num_inference_steps * num_frames_per_chunk * FRAME_SIZE_SAMPLES
        if padded_total_samples > total_samples:
            audio_signal = np.pad(audio_signal, (0, padded_total_samples - total_samples), mode='constant')
            logging.info(f"   Padded to: {padded_total_samples} samples")
        logging.info(f" {num_frames_per_chunk=} => {total_frames=}, {num_inference_steps=}")

        # convert audio signal to tensor
        audio_signal_tensor = torch.tensor(audio_signal, dtype=self.dtype, device=self.device).unsqueeze(0)

        # Check if Nemotron (no cache support)
        use_cache = 'Nemotron' not in self.model.stt_model.cfg.pretrained_llm
        logging.info(f"Model: {self.model.stt_model.cfg.pretrained_llm}")
        logging.info(f"   Use cache: {use_cache}")

        # Initialize buffer and state
        audio_buffer = torch.zeros(1, buffer_size_samples, dtype=self.dtype, device=self.device)
        buffer_fill_level = 0  # How many samples currently in buffer

        # Initialize LLM cache
        if use_cache:
            llm_cache = DynamicCache()
        else:
            llm_cache = None
            input_embeds_history = []  # For no-cache mode

        # Process system prompt if provided (before streaming audio)
        prompt_embedded = None
        prompt_len = 0
        
        if system_prompt:
            start_get_prompt_embeddings = time.time()
            prompt_embedded, prompt_len = self._prepare_system_prompt_embeddings(system_prompt)
            logging.info(f"Time taken to get prompt embeddings: {time.time() - start_get_prompt_embeddings:.3f}s")
            if prompt_embedded is not None and "vllm" in self.engine_type.lower():
                # Prepare token IDs for the prompt
                prompt_token_ids = (
                    [self.tokenizer.bos_id] +
                    self.tokenizer.text_to_ids(system_prompt) +
                    [self.tokenizer.eos_id]
                )

                # For vLLM mode: use efficient BATCH prefill (~20x faster than sequential)
                logging.info(f"   Batch prefilling {prompt_len} prompt embeddings...")
                start_batch_prefill = time.time()
                with torch.no_grad():
                    success = self.model_llm_interface(
                        prompt_embedded,
                        request_id=stream_request_id,
                        decode_steps=0,
                        prompt_token_ids=prompt_token_ids,
                    )
                logging.info(f"Time taken to batch prefill stt model: {time.time() - start_batch_prefill:.3f}s")
                if success:
                    logging.info(f" System prompt prefilled ({prompt_len} tokens)")
                else:
                    raise RuntimeError("vLLM batch prefill for system prompt failed.")
            elif prompt_embedded is not None and not use_cache:
                # For no-cache mode (Nemotron): add prompt embeddings to history
                # Split into individual frames for consistent processing
                for t in range(prompt_len):
                    input_embeds_history.append(prompt_embedded[:, t:t+1, :])
                logging.info(f"   Added {prompt_len} prompt embeddings to input_embeds_history")
            elif prompt_embedded is not None and use_cache:
                # For cache mode: process prompt through LLM to update cache
                with torch.no_grad():
                    ans = self.model.stt_model(prompt_embedded, cache=llm_cache)
                    llm_cache = ans.get("cache", llm_cache)
                logging.info(f"   System prompt processed, cache updated")

        # Initialize TTS
        code = None
        past_key_values = None
        subword_mask = None
        audio_toks_buffer = None
        if self.decode_audio and hasattr(self.model, 'tts_model'):

            # Sliding-window buffer is only needed when codec_cache is off
            if not self.use_codec_cache:
                audio_toks_buffer = self.model.tts_model.codec_silence_tokens.view(1, 1, -1).expand(
                    -1, self.codec_token_history_size, -1
                ).to(self.device)

            if (
                self.first_context_subword_id is None
                or self.generation_config is None
                or self.first_tts_code_input is None
                or self.first_tts_past_key_values_input is None
            ) and not self.use_vllm_eartts:
                raise RuntimeError("TTS warmup state was not prepared during initialization.")

            if not self.use_vllm_eartts:
                past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
                code = self.first_tts_code_input.detach().clone()
            else:
                start_batch_prefill = time.time()
                logging.info(f"   Batch prefilling TTS model with speaker embedding...")
                # use speaker embedding to prefill EarTTS's vLLM
                tts_result = self.model.tts_model.tts_model(
                    self.tts_init_inputs,
                    request_id=stream_request_id,
                    prompt_token_ids=self.tts_prompt_token_ids
                )
                code = self.first_tts_code_input.detach().clone()
                past_key_values = None
                logging.info(f"Time taken to batch prefill tts model: {time.time() - start_batch_prefill:.3f}s")
                # Initialize subword_mask for vLLM path as well
            subword_mask = torch.ones(1, timeline_capacity, device=self.device, dtype=torch.bool)
            logging.info(f"TTS initialized")

        # Initialize perception cache if enabled
        perception_cache = None
        if self.use_perception_cache:
            perception_cache = self.perception_cache_mgr.get_initial_state(batch_size=1)
            logging.info(f"Perception cache initialized")

        # Initialize codec streaming cache to remove clicking sounds and wasted inference computation
        codec_cache = None
        if self.decode_audio and self.use_codec_cache:
            from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
            codec_cache = CausalConv1dCache()
            logging.info(f"Codec streaming cache initialized")

        # When fc_async is enabled, extra timeline slots are needed for FC tokens
        # that run ahead of the audio. Pre-allocate generous headroom.
        fc_async_headroom = 2000 if (self._fc_async_enabled and self.model.stt_model.function_head is not None) else 0
        timeline_capacity = total_frames + fc_async_headroom

        gen_text = torch.full((1, timeline_capacity), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = torch.full((1, timeline_capacity), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)
        has_function_head = self.model.stt_model.function_head is not None
        if has_function_head:
            gen_function_text = torch.full((1, timeline_capacity), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)

        # initialize list to which we will append generated audio segments
        audio_segments = []

        # Initialize RNNT state for user transcription
        has_rnnt = getattr(self.model.stt_model, '_rnnt_decoder', None) is not None
        rnnt_partial_hypotheses = self._rnnt_init_state(1, self.device) if has_rnnt else None

        # FC async state: tracks the FC state machine across frames
        fc_state = {"active": False} if has_function_head and self._fc_async_enabled else None
        fc_total_async_steps = 0

        logging.info("\n" + "=" * 70)
        logging.info("STARTING FRAME-BY-FRAME PROCESSING")
        logging.info("=" * 70)
        if has_rnnt:
            logging.info("RNNT user transcription enabled")
        if self._fc_async_enabled and has_function_head:
            logging.info("FC async mode ENABLED — text will run at full LLM speed during tool calls")
            if self._fc_async_two_phase:
                logging.info("FC async two-phase ENABLED — will pause at EOTC for tool execution%s",
                             " (tool_executor_fn provided)" if tool_executor_fn else " (no tool_executor_fn)")

        # frame_idx corresponds to index of the first frame passed to infer_one_step
        # (we need this distinction in the case that num_frames_per_chunk > 1)
        frame_idx = 0
        while frame_idx < total_frames:
            slice_start = frame_idx * FRAME_SIZE_SAMPLES
            slice_n_samples = num_frames_per_chunk * FRAME_SIZE_SAMPLES
            slice_end = slice_start + slice_n_samples
            new_audio = audio_signal_tensor[:, slice_start:slice_end]

            audio_buffer, buffer_fill_level, current_buffer = self._update_audio_buffer(
                audio_buffer, buffer_fill_level, new_audio, buffer_size_samples
            )

            result = self.infer_one_step(
                audio_input=current_buffer,
                num_frames_per_chunk=num_frames_per_chunk,
                frame_idx=frame_idx,
                gen_text=gen_text,
                audio_toks_buffer=audio_toks_buffer if self.decode_audio else None,
                input_embeds_history=input_embeds_history if not use_cache else [],
                dynamic_cache=llm_cache if use_cache else None,
                past_key_values=past_key_values if self.decode_audio else None,
                code=code if self.decode_audio else None,
                subword_mask=subword_mask if self.decode_audio else None,
                gen_asr_text=gen_asr_text,
                gen_function_text=gen_function_text if has_function_head else None,
                request_id=stream_request_id,
                perception_cache=perception_cache,
                has_prompt=(prompt_len > 0),
                codec_cache=codec_cache,
                rnnt_partial_hypotheses=rnnt_partial_hypotheses,
                fc_state=fc_state,
                tool_response_text=None,
            )

            # handle results from infer_one_step
            if has_function_head and 'function_predicted_text_tokens' in result:
                for fi in range(num_frames_per_chunk):
                    gen_function_text[:, frame_idx + fi] = result['function_predicted_text_tokens'][:, fi]
            input_embeds_history = result['input_embeds_history']
            llm_cache = result['dynamic_cache']
            if self.use_perception_cache:
                perception_cache = result.get('perception_cache', perception_cache)
            rnnt_partial_hypotheses = result.get('rnnt_partial_hypotheses', rnnt_partial_hypotheses)
            if self.decode_audio:
                audio_toks_buffer = result['audio_toks_buffer']
                decoded_audio_new = result['decoded_audio_new']
                if decoded_audio_new is not None:
                    audio_segments.append(decoded_audio_new)

                past_key_values = result['past_key_values']
                code = result['code']
                codec_cache = result.get('codec_cache', codec_cache)
            else:
                decoded_audio_new = None

            # FC Async: if SOTC was detected in infer_one_step, run the async loop
            if fc_state is not None and fc_state.pop("trigger_async", False):
                sotc_frame = frame_idx + num_frames_per_chunk - 1  # frame where SOTC was detected
                logging.info(
                    "[FC Async] Triggered at frame %d — entering async loop",
                    sotc_frame,
                )
                fc_wall_start = time.time()

                # Build realtime audio context: during the async loop, new
                # audio frames become available at the real-time rate (one
                # per 80 ms of wall-clock time), simulating a live mic.
                rt_audio_ctx = {
                    "audio_signal_tensor": audio_signal_tensor,
                    "next_audio_frame": frame_idx + num_frames_per_chunk,
                    "total_audio_frames": total_frames,
                    "audio_buffer": audio_buffer.clone(),
                    "buffer_fill_level": buffer_fill_level,
                    "buffer_size_samples": buffer_size_samples,
                    "wall_start": fc_wall_start,
                    "frames_consumed": 0,
                }

                # Phase 1: generate call tokens (SOTC → ... → EOTC) at full LLM speed
                async_steps, llm_cache, _, _, _ = self._run_fc_async_steps(
                    fc_state=fc_state,
                    gen_text=gen_text,
                    gen_asr_text=gen_asr_text,
                    gen_function_text=gen_function_text,
                    current_frame_idx=sotc_frame,
                    dynamic_cache=llm_cache,
                    input_embeds_history=input_embeds_history if not use_cache else [],
                    tool_response_text=None,
                    realtime_audio=rt_audio_ctx,
                    request_id=stream_request_id,
                )
                fc_total_async_steps += async_steps
                logging.info(
                    "[FC Async] Phase 1 completed: %d steps (total async: %d)",
                    async_steps, fc_total_async_steps,
                )

                # Phase 2 (two-phase mode): execute tool, then re-enter async to
                # inject response tokens at full LLM speed
                if fc_state.pop("awaiting_response", False):
                    call_text = fc_state.get("last_call_text", "")
                    phase2_start_t = fc_state.pop("phase1_end_t", sotc_frame + async_steps)
                    tool_response = None

                    if tool_executor_fn is not None:
                        logging.info(
                            "[FC Async] Phase 2: executing tool for call: %s", call_text
                        )
                        try:
                            tool_response = tool_executor_fn(call_text)
                            logging.info(
                                "[FC Async] Phase 2: tool returned %d chars",
                                len(tool_response) if tool_response else 0,
                            )
                        except Exception as e:
                            logging.error(
                                "[FC Async] Phase 2: tool_executor_fn raised %s: %s",
                                type(e).__name__, e,
                            )
                    else:
                        logging.info(
                            "[FC Async] Phase 2: no tool_executor_fn provided, "
                            "skipping response injection"
                        )

                    if tool_response:
                        async_steps_p2, llm_cache, _, _, _ = self._run_fc_async_steps(
                            fc_state=fc_state,
                            gen_text=gen_text,
                            gen_asr_text=gen_asr_text,
                            gen_function_text=gen_function_text,
                            current_frame_idx=phase2_start_t - 1,
                            dynamic_cache=llm_cache,
                            input_embeds_history=input_embeds_history if not use_cache else [],
                            tool_response_text=tool_response,
                            realtime_audio=rt_audio_ctx,
                            request_id=stream_request_id,
                        )
                        fc_total_async_steps += async_steps_p2
                        logging.info(
                            "[FC Async] Phase 2 completed: %d steps (total async: %d)",
                            async_steps_p2, fc_total_async_steps,
                        )

                fc_wall_elapsed = time.time() - fc_wall_start

                # Adopt the audio buffer state that was updated by the async
                # loop (it may have consumed real frames and shifted the
                # sliding window forward).
                rt_consumed = rt_audio_ctx["frames_consumed"]
                if rt_consumed > 0:
                    audio_buffer = rt_audio_ctx["audio_buffer"]
                    buffer_fill_level = rt_audio_ctx["buffer_fill_level"]
                    frame_idx += rt_consumed
                    logging.info(
                        "[FC Async] Consumed %d real audio frames during async "
                        "(frame_idx advanced to %d)",
                        rt_consumed, frame_idx,
                    )

                # Insert silence audio to reflect the real-time wait during FC.
                # In deployment the user hears silence while the tool executes;
                # we replicate that in the saved audio so playback is faithful.
                if self.decode_audio:
                    tts_rate = getattr(self, "target_sample_rate", TTS_SAMPLE_RATE)
                    silence_samples = int(fc_wall_elapsed * tts_rate)
                    if silence_samples > 0:
                        silence_audio_chunk = torch.zeros(
                            1, 1, silence_samples,
                            device=self.device, dtype=torch.float32,
                        )
                        audio_segments.append(silence_audio_chunk)
                        logging.info(
                            "[FC Async] Inserted %.2fs (=%d samples @ %dHz) silence "
                            "into audio output to reflect FC wall-clock time",
                            fc_wall_elapsed, silence_samples, tts_rate,
                        )

                logging.info(
                    "[FC Async] FC async done (wall=%.3fs, rt_audio=%d frames). "
                    "Audio resumes at frame %d",
                    fc_wall_elapsed, rt_consumed,
                    frame_idx + num_frames_per_chunk,
                )

            if frame_idx % 10 == 0 or frame_idx < 3 or gen_text[:, frame_idx].item() == self.model.stt_model.text_eos_id:
                token_str = self.tokenizer.ids_to_text([gen_text[0, frame_idx].item()])
                buffer_status = f"{buffer_fill_level}/{buffer_size_samples}" if buffer_fill_level < buffer_size_samples else "FULL"
                special_label = ""
                if gen_text[0, frame_idx].item() == self.model.stt_model.text_bos_id:
                    special_label = " [BOS]"
                elif gen_text[0, frame_idx].item() == self.model.stt_model.text_eos_id:
                    special_label = " [EOS]"
                elif gen_text[0, frame_idx].item() == self.model.stt_model.text_pad_id:
                    special_label = " [PAD]"
                logging.info(f"Frame {frame_idx:3d}/{total_frames} | Buffer: {buffer_status:20s} | Token: {gen_text[0, frame_idx].item():5d}{special_label} | '{token_str}'")

            frame_idx += num_frames_per_chunk

        # Prepare results
        elapsed_time = time.time() - start_time
        # Effective timeline length = audio frames + any async FC steps that ran ahead
        effective_total_frames = total_frames + fc_total_async_steps

        logging.info("\n" + "=" * 70)
        logging.info("STREAMING INFERENCE COMPLETED")
        logging.info("=" * 70)
        logging.info(f"Total time: {elapsed_time:.2f}s")
        logging.info(f"Audio duration: {total_duration:.2f}s")
        logging.info(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        logging.info(f"Processed frames: {total_frames} (audio) + {fc_total_async_steps} (async FC) = {effective_total_frames}")

        # Trim to effective length (audio frames + async FC frames)
        gen_text = gen_text[:, :effective_total_frames]
        gen_asr_text = gen_asr_text[:, :effective_total_frames]

        # Decode text
        lengths = torch.tensor([effective_total_frames], dtype=torch.long, device=self.device)
        text_output = tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        # Decode ASR text
        asr_text_output = tokens_to_str(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        # Also create raw versions with <SPECIAL_12> kept for comparison
        text_output_raw = tokens_to_str_raw(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)
        asr_text_output_raw = tokens_to_str_raw(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)

        logging.info(f"Generated text: {text_output[0]}")
        logging.info(f"Generated ASR text: {asr_text_output[0]}")

        # Finalize RNNT transcript (rnnt_partial_hypotheses is now a step-state dict, no text extraction)
        rnnt_asr_text = None

        # Decode function calling channel
        if has_function_head:
            gen_function_text = gen_function_text[:, :effective_total_frames]
            function_text_output = tokens_to_str(gen_function_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=False)
            function_text_output_raw = tokens_to_str_raw(gen_function_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)
            logging.info(f"Generated function text: {function_text_output[0]}")

        ans = {
            "text": text_output,
            "text_raw": text_output_raw,
            "tokens_text": gen_text,
            "tokens_len": lengths,
            "audio": torch.cat(audio_segments, dim=-1) if audio_segments else None,
            "asr_text": asr_text_output,
            "asr_text_raw": asr_text_output_raw,
            "asr_tokens": gen_asr_text,
            "system_prompt": system_prompt if system_prompt else "",
            "rnnt_asr_text": rnnt_asr_text,
        }
        if has_function_head:
            ans["function_text"] = function_text_output
            ans["function_text_raw"] = function_text_output_raw
            ans["function_tokens"] = gen_function_text

        if self.use_vllm_llm or self.use_vllm_eartts:
            self.abort_request(stream_request_id)

        return ans


def main():
    parser = argparse.ArgumentParser(description="Realtime Streaming Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to eartts's checkpoint with TTS (HF format)")
    parser.add_argument("--llm_checkpoint_path", type=str, required=True,
                       help="Path to checkpoint with LLM/perception (HF format)")
    parser.add_argument("--audio_path", type=str, default=None,
                       help="Path to input audio file (for single-file mode)")
    parser.add_argument("--input_json", type=str, default=None,
                       help="Path to input JSON file containing list of records with audio_filepath and text fields (for batch mode)")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Path to output JSON file with predictions")
    parser.add_argument("--output_dir", type=str, default="output_streaming",
                       help="Output directory for audio files and JSON results")
    parser.add_argument("--pad_audio_to_sec", type=float, default=None,
                       help="Pad audio to this duration in seconds (useful for consistent buffer behavior)")
    parser.add_argument("--pad_silence_ratio", type=float, default=None,
                       help="Append silence equal to this ratio of the original audio duration (e.g. 0.2 = 20%% extra)")
    parser.add_argument("--pad_audio_by_sec", type=float, default=None,
                       help="Append this many seconds of extra silence after the audio")
    parser.add_argument("--speaker_reference", type=str, required=True,
                       help="Path to speaker reference audio file")
    parser.add_argument("--buffer_size_frames", type=int, default=DEFAULT_BUFFER_SIZE_FRAMES,
                       help=f"Size of audio buffer in frames (each frame = 80ms, default: {DEFAULT_BUFFER_SIZE_FRAMES})")
    parser.add_argument("--num_frames_per_chunk", type=int, default=DEFAULT_NUM_FRAMES_PER_CHUNK,
                       help="Number of frames per inference step (default: 1)")
    parser.add_argument("--decode_audio", action="store_true",
                       help="Whether to decode audio")
    parser.add_argument("--combine_inp_out_audio", action="store_true",
                       help="Whether to combine input and output audio into a stereo file")

    # Deterministic inference
    parser.add_argument("--deterministic", action="store_true",
                       help="Enable fully deterministic inference (disables FlashAttention, forces deterministic "
                            "CUDA algorithms). Useful for reproducible benchmarking. Not compatible with vLLM engines. "
                            "Note: results may differ slightly from non-deterministic mode due to different compute path.")

    # Perception cache argument
    parser.add_argument("--use_perception_cache", action="store_true",
                       help="Enable cache-aware streaming for perception encoder")
    parser.add_argument("--use_perception_cudagraph", action="store_true",
                       help="Use CUDA graphs for perception encoder (requires --use_perception_cache)")
    # Codec streaming cache argument
    parser.add_argument("--use_codec_cache", action="store_true",
                       help="Enable incremental codec decode to remove clicking sounds and wasted inference computation (recommended)")

    # vLLM arguments
    parser.add_argument("--engine_type", type=str, default="native", choices=["native", "vllm_llm", "vllm_eartts", "vllm_llm_vllm_eartts"],
                       help="Engine type for inference (default: native)")
    parser.add_argument("--vllm_llm_engine_path", type=str, default=None,
                       help="Path to vLLM-compatible model checkpoint if the path not exists, it will be auto-converted")
    parser.add_argument("--vllm_max_model_len", type=int, default=768,
                       help="Maximum sequence length for vLLM (default: 768)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, nargs='+', default=[0.4],
                       help="GPU memory utilization for vLLM. Single value shared by both engines; two values assign to LLM and TTS respectively.")
    parser.add_argument("--vllm_llm_dtype", type=str, default="bfloat16",
                       help="Data type for vLLM (default: bfloat16)")

    # vLLM EarTTS arguments
    parser.add_argument("--vllm_eartts_engine_path", type=str, default=None,
                       help="Path to vLLM-compatible EarTTS model checkpoint if the path not exists, it will be auto-converted")
    parser.add_argument("--vllm_eartts_dtype", type=str, default="float32",
                       help="Data type for vLLM (default: float32)")

    # Sampling parameters
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty for generated tokens. 1.0 disables it. Default: 1.0. Recommended: 1.2")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for sampling. 1.0 = no change, <1.0 = sharper, >1.0 = flatter, 0.0 = greedy. Default: 1.0")

    # Turn-taking
    parser.add_argument("--force_turn_taking", action="store_true",
                       help="Enable forced turn-taking based on ASR channel tokens")
    parser.add_argument("--force_turn_taking_threshold", type=int, default=40,
                       help="Number of lookback steps for turn-taking detection (default: 40)")
    parser.add_argument("--force_turn_taking_pad_window", type=int, default=25,
                       help="Number of consecutive ASR pad tokens to trigger turn-taking (default: 25)")

    # Inference logit boosts
    parser.add_argument("--inference_pad_boost", type=float, default=None,
                       help="Boost for agent pad logit at inference time")
    parser.add_argument("--inference_bos_boost", type=float, default=None,
                       help="Boost for agent BOS logit at inference time")
    parser.add_argument("--inference_eos_boost", type=float, default=None,
                       help="Boost for agent EOS logit at inference time")
    parser.add_argument("--inference_user_pad_boost", type=float, default=None,
                       help="Boost for ASR pad logit at inference time")
    parser.add_argument("--inference_user_bos_boost", type=float, default=None,
                       help="Boost for ASR BOS logit at inference time")
    parser.add_argument("--inference_user_eos_boost", type=float, default=None,
                       help="Boost for ASR EOS logit at inference time")

    # System prompt
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt to provide context to the model. Can also be specified per-record in input JSON.")
    parser.add_argument("--tts_system_prompt", type=str, default=None,
                       help="System prompt for EARTTS model.")

    # FC async mode
    parser.add_argument("--fc_async_enabled", action="store_true",
                       help="Enable async function calling: text tokens run at full LLM speed during "
                            "tool calls using cached silence embeddings, decoupled from 80ms audio frames")
    parser.add_argument("--fc_async_two_phase", action="store_true", default=False,
                       help="Enable two-phase async FC: Phase 1 generates call tokens at LLM speed, "
                            "pauses at EOTC for tool execution, then Phase 2 injects response at LLM "
                            "speed. Requires --fc_async_enabled. Default: False (single-phase)")
    args = parser.parse_args()

    # Validate arguments: either audio_path OR input_json must be provided
    if args.audio_path is None and args.input_json is None:
        parser.error("Either --audio_path (single-file mode) or --input_json (batch mode) must be provided")
    if args.audio_path is not None and args.input_json is not None:
        parser.error("Cannot use both --audio_path and --input_json at the same time")

    if sum(x is not None for x in [args.pad_audio_to_sec, args.pad_silence_ratio, args.pad_audio_by_sec]) > 1:
        raise ValueError("Set at most one of: --pad_audio_to_sec, --pad_silence_ratio, --pad_audio_by_sec")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error(f"--temperature must be a finite value >= 0.0, got {args.temperature}")
    if args.fc_async_two_phase and not args.fc_async_enabled:
        parser.error("--fc_async_two_phase requires --fc_async_enabled")

    try:
        import json
        import soundfile as sf

        model_cfg_dict = {
            "model_path": args.model_path,
            "llm_checkpoint_path": args.llm_checkpoint_path,
            "speaker_reference": args.speaker_reference,
            "buffer_size_frames": args.buffer_size_frames,
            "decode_audio": bool(args.decode_audio),
            "engine_type": args.engine_type,
            "deterministic": bool(args.deterministic),
            "use_perception_cache": bool(args.use_perception_cache),
            "use_perception_cudagraph": bool(args.use_perception_cudagraph),
            "use_codec_cache": bool(args.use_codec_cache),
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "tts_system_prompt": args.tts_system_prompt,
            "force_turn_taking": args.force_turn_taking,
            "force_turn_taking_threshold": args.force_turn_taking_threshold,
            "force_turn_taking_pad_window": args.force_turn_taking_pad_window,
            "inference_pad_boost": args.inference_pad_boost,
            "inference_bos_boost": args.inference_bos_boost,
            "inference_eos_boost": args.inference_eos_boost,
            "inference_user_pad_boost": args.inference_user_pad_boost,
            "inference_user_bos_boost": args.inference_user_bos_boost,
            "inference_user_eos_boost": args.inference_user_eos_boost,
            "fc_async_enabled": bool(args.fc_async_enabled),
            "fc_async_two_phase": bool(args.fc_async_two_phase),
        }

        # Pop GPU memory utilization values: first for LLM, second (or same) for TTS
        _gpu_mem = list(args.vllm_gpu_memory_utilization)
        gpu_mem_llm = _gpu_mem.pop(0)
        gpu_mem_tts = _gpu_mem.pop(0) if _gpu_mem else gpu_mem_llm

        # Add vLLM configuration if using vLLM engine
        if "vllm_llm" in args.engine_type:
            model_cfg_dict["vllm_llm_config"] = {
                "model_path": args.model_path,
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": gpu_mem_llm,
                "dtype": args.vllm_llm_dtype,
                "engine_path": args.vllm_llm_engine_path,  # Will auto-convert if needed
                "pretrained_llm": args.llm_checkpoint_path,
            }

        if "vllm_eartts" in args.engine_type:
            model_cfg_dict["vllm_tts_config"] = {
                "model_path": args.model_path, # we use exactly the same whole duplexs2s ckpt
                "max_model_len": args.vllm_max_model_len,
                "gpu_memory_utilization": gpu_mem_tts,
                "dtype": args.vllm_eartts_dtype,
                "engine_path": args.vllm_eartts_engine_path,
                "pretrained_llm": None,
                "skip_tokenizer_init": True
            }

        model_cfg = OmegaConf.create(model_cfg_dict)

        model = NemotronVoicechatInferenceWrapper(model_cfg=model_cfg)

        # =========================================
        # Load input records (from JSON manifest or single audio file)
        # =========================================
        if args.input_json is not None:
            logging.info(f"Loading input JSON: {args.input_json}")
            with open(args.input_json, 'r') as f:
                input_records = [json.loads(line) for line in f]
        else:
            input_records = [{"audio_filepath": args.audio_path, "text": ""}]

        logging.info(f"Found {len(input_records)} records to process")

        os.makedirs(args.output_dir, exist_ok=True)

        if args.output_json:
            base_path = args.output_json.rsplit('.', 1)[0] if '.' in args.output_json else args.output_json
            output_json_processed = f"{base_path}_processed.json"
            output_json_raw = f"{base_path}_raw.json"
        else:
            output_json_processed = os.path.join(args.output_dir, "output_results_processed.json")
            output_json_raw = os.path.join(args.output_dir, "output_results_raw.json")

        logging.info(f"Output will be saved incrementally to:")
        logging.info(f"   Processed: {output_json_processed}")
        logging.info(f"   Raw: {output_json_raw}")
        output_file_processed = open(output_json_processed, 'w', encoding='utf-8')
        output_file_raw = open(output_json_raw, 'w', encoding='utf-8')

        output_records = []
        wer_scores = []

        try:
            for idx, record in enumerate(input_records):
                logging.info("\n" + "=" * 70)
                logging.info(f"Processing record {idx + 1}/{len(input_records)}")
                logging.info("=" * 70)

                audio_path = record.get('audio_filepath')
                ground_truth_text = record.get('text', '')
                record_system_prompt = record.get('system_prompt', args.system_prompt)

                if not audio_path:
                    logging.warning(f"Record {idx} missing audio_filepath, skipping...")
                    continue

                if not os.path.exists(audio_path):
                    logging.warning(f"Audio file not found: {audio_path}, skipping...")
                    continue

                logging.info(f"   Audio: {audio_path}")
                logging.info(f"   Ground truth: {ground_truth_text}")

                audio_id = os.path.splitext(os.path.basename(audio_path))[0]

                results = model.inference_realtime_streaming(
                    audio_path,
                    num_frames_per_chunk=args.num_frames_per_chunk,
                    pad_audio_to_sec=args.pad_audio_to_sec,
                    pad_silence_ratio=args.pad_silence_ratio,
                    pad_audio_by_sec=args.pad_audio_by_sec,
                    request_id=f"streaming_request_{idx}",
                    system_prompt=record_system_prompt,
                )

                pred_asr_text = results['asr_text'][0] if 'asr_text' in results else ''
                pred_asr_text_raw = results['asr_text_raw'][0] if 'asr_text_raw' in results else ''
                pred_text = results['text'][0] if 'text' in results else ''
                pred_text_raw = results['text_raw'][0] if 'text_raw' in results else ''
                rnnt_asr_text = results.get('rnnt_asr_text')
                pred_rnnt_text = rnnt_asr_text[0] if rnnt_asr_text else ''

                # Use RNNT transcript for WER if available, otherwise fall back to ASR head
                wer_pred_text = pred_rnnt_text if pred_rnnt_text else pred_asr_text

                try:
                    cleaned_pred = clean_pred_text(wer_pred_text)
                    cleaned_gt = clean_pred_text(ground_truth_text)
                    if cleaned_gt.strip() and cleaned_pred.strip():
                        utterance_wer = wer(cleaned_gt, cleaned_pred)
                        wer_scores.append(utterance_wer)
                    else:
                        utterance_wer = None
                except Exception as e:
                    utterance_wer = None
                    logging.warning(f"Error calculating WER: {e}")

                if utterance_wer is not None:
                    logging.info(f"WER for utterance {idx + 1}: {utterance_wer:.4f} ({utterance_wer * 100:.2f}%)")

                pred_audio_path = None
                if args.decode_audio and 'audio' in results and results['audio'] is not None:
                    input_basename = os.path.splitext(os.path.basename(audio_path))[0]
                    audio_filename = f"{idx:04d}_{input_basename}_output.wav"
                    pred_audio_path = os.path.join(args.output_dir, audio_filename)

                    audio_np = results['audio'].float().cpu().numpy().flatten()

                    sf.write(pred_audio_path, audio_np, model.target_sample_rate)
                    logging.info(f"Audio saved: {pred_audio_path}")

                    if args.combine_inp_out_audio:
                        stereo_filename = f"{idx:04d}_{input_basename}_combined.wav"
                        stereo_path_out = os.path.join(args.output_dir, stereo_filename)

                        inp_audio, sr = librosa.load(audio_path, sr=model.target_sample_rate)

                        delay_samples = int(args.num_frames_per_chunk * FRAME_SIZE_SEC * model.target_sample_rate)
                        out_audio_delayed = np.concatenate([np.zeros(delay_samples, dtype=audio_np.dtype), audio_np])

                        max_len = max(len(inp_audio), len(out_audio_delayed))
                        inp_audio_padded = np.pad(inp_audio, (0, max_len - len(inp_audio)))
                        out_audio_padded = np.pad(out_audio_delayed, (0, max_len - len(out_audio_delayed)))

                        stereo_audio = np.stack([inp_audio_padded, out_audio_padded], axis=1)
                        sf.write(stereo_path_out, stereo_audio, model.target_sample_rate)
                        logging.info(f"Stereo audio saved: {stereo_path_out}")

                result_system_prompt = results.get('system_prompt', '')

                output_record_processed = {
                    'id': audio_id,
                    'target_text': '',
                    'pred_audio': pred_audio_path,
                    'src_text': ground_truth_text,
                    'pred_src_text': pred_asr_text,
                    'pred_src_text_rnnt': pred_rnnt_text,
                    'pred_text': pred_text,
                    'system_prompt': result_system_prompt,
                }

                output_record_raw = {
                    'id': audio_id,
                    'target_text': '',
                    'pred_audio': pred_audio_path,
                    'src_text': ground_truth_text,
                    'pred_src_text': pred_asr_text_raw,
                    'pred_src_text_rnnt': pred_rnnt_text,
                    'pred_text': pred_text_raw,
                    'system_prompt': result_system_prompt,
                }

                output_records.append(output_record_processed)

                json.dump(output_record_processed, output_file_processed, ensure_ascii=False)
                output_file_processed.write('\n')
                output_file_processed.flush()

                json.dump(output_record_raw, output_file_raw, ensure_ascii=False)
                output_file_raw.write('\n')
                output_file_raw.flush()

                logging.info(f"Record {idx + 1} completed and saved")

        finally:
            output_file_processed.close()
            output_file_raw.close()

        logging.info("\n" + "=" * 70)
        logging.info("ALL RESULTS SAVED")
        logging.info("=" * 70)
        logging.info(f"Results saved to:")
        logging.info(f"   Processed: {output_json_processed}")
        logging.info(f"   Raw: {output_json_raw}")
        logging.info(f"   Processed {len(output_records)}/{len(input_records)} records successfully")

        if wer_scores:
            avg_wer = np.mean(wer_scores)
            logging.info("\n" + "=" * 70)
            logging.info("WER STATISTICS")
            logging.info("=" * 70)
            logging.info(f"   Total utterances with WER: {len(wer_scores)}")
            logging.info(f"   Average WER: {avg_wer:.4f} ({avg_wer * 100:.2f}%)")
            logging.info(f"   Min WER: {np.min(wer_scores):.4f} ({np.min(wer_scores) * 100:.2f}%)")
            logging.info(f"   Max WER: {np.max(wer_scores):.4f} ({np.max(wer_scores) * 100:.2f}%)")

        logging.info("=" * 70)
        logging.info("ALL DONE!")
        logging.info("=" * 70)

    except Exception as e:
        logging.error(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

