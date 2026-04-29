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
#
# Author: Harishchandra Dubey (hdubey@nvidia.com)
# RNNT EOU/BOU turn-taking: _rnnt_init_state, _rnnt_step, _apply_rnnt_turn_taking
# Mechanisms: rolling density, adaptive min_speech, post-EOS fallback, force-EOS, BOS suppression

import torch
import yaml
from omegaconf import OmegaConf, DictConfig
import numpy as np
import librosa
import time
from transformers import DynamicCache
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

        # RNNT EOU event log: timestamped file written whenever EOU or BOU fires
        import datetime
        _eou_log = model_cfg.get("rnnt_eou_log", None)
        self._rnnt_eou_log_path = _eou_log
        if _eou_log:
            os.makedirs(os.path.dirname(os.path.abspath(_eou_log)), exist_ok=True)
            with open(_eou_log, "w") as _f:
                _f.write(f"# RNNT EOU/BOU event log — started {datetime.datetime.now().isoformat()}\n")
                _f.write("# format: ISO-timestamp | event | audio_frame | t_sec | blank_cnt | nonblank_total | notes\n")

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

        # Convert to dict for model initialization
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Step 3: Initialize model structure
        logging.info("Initializing model structure...")
        start_DuplexS2S_init = time.time()
        self.model = NemotronVoiceChat(cfg_dict)
        logging.info(f"Time taken to initialize NemotronVoiceChat: {time.time() - start_DuplexS2S_init} seconds")
        logging.info("  Model structure initialized")

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

        # Optionally move TTS to a separate GPU to split memory load across devices.
        # Config: s2s.tts_device_id=1 sends EarTTS to GPU 1, leaving STT/LLM on GPU 0.
        tts_device_id = self.model_cfg.get("tts_device_id", None)
        if tts_device_id is not None and hasattr(self.model, 'tts_model'):
            self.tts_device = torch.device(f"cuda:{tts_device_id}")
            self.model.tts_model.to(self.tts_device)
            logging.info(f"TTS model moved to {self.tts_device} (STT/LLM stays on {self.device})")
        else:
            self.tts_device = self.device

        # Convert only the S2S components to the configured dtype, not the TTS model
        logging.info(f"Converting S2S components to {self.dtype} (keeping TTS in float32)...")
        if self.model.stt_model.llm is not None:
            self.model.stt_model.llm = self.model.stt_model.llm.to(self.dtype)
        self.model.stt_model.lm_head = self.model.stt_model.lm_head.to(self.dtype)
        self.model.stt_model.embed_tokens = self.model.stt_model.embed_tokens.to(self.dtype)
        # When use_separate_asr_head=False, asr_head and embed_asr_tokens are absent.
        # Alias them to lm_head/embed_tokens so downstream inference code works unchanged.
        if not hasattr(self.model.stt_model, 'asr_head'):
            self.model.stt_model.asr_head = self.model.stt_model.lm_head
        if not hasattr(self.model.stt_model, 'embed_asr_tokens'):
            self.model.stt_model.embed_asr_tokens = self.model.stt_model.embed_tokens
        self.model.stt_model.asr_head = self.model.stt_model.asr_head.to(self.dtype)
        self.model.stt_model.embed_asr_tokens = self.model.stt_model.embed_asr_tokens.to(self.dtype)
        if self.model.stt_model.function_head is not None:
            self.model.stt_model.function_head = self.model.stt_model.function_head.to(self.dtype)
            logging.info("function_head converted to %s", self.dtype)
        #self.model.stt_model.perception = self.model.stt_model.perception.to(self.dtype)
        logging.info("S2S components converted, TTS kept in float32")
        logging.info("new update, perception also is kept in float32")

        # commenting this out to avoid error when try vllm tts
        # and anyway - when sticking to "native", saw no difference in output
        # with and without this call
        #self.model.on_train_epoch_start()
        self.tokenizer = self.model.stt_model.tokenizer


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
            )

            logging.info("VllmLLMModel interface created")
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

        # Setup perception cache if enabled
        if self.use_perception_cache:
            self.perception_cache_mgr = PerceptionCacheManager(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                use_cudagraph=self._use_perception_cudagraph,
            )
            if not self.perception_cache_mgr.setup():
                self.use_perception_cache = False
                self.perception_cache_mgr = None

        # Move RNNT decoder+joint to inference device (set via object.__setattr__, not auto-moved by .to())
        stt = self.model.stt_model
        if hasattr(stt, '_rnnt_decoder'):
            stt._rnnt_decoder.to(self.device)
            stt._rnnt_joint.to(self.device)
            logging.info(f"RNNT decoder+joint moved to {self.device}")

    def _get_bos_embedding(self):
        """Get beginning of sequence embedding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

    def _get_asr_bos_embedding(self) -> torch.Tensor:
        """Get ASR BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.model.stt_model.text_pad_id, device=self.device)
        input_embeds = self.model.stt_model.embed_asr_tokens(text_bos)
        return input_embeds.to(dtype=self.dtype)

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
        prompt_token_ids = (
            [self.tokenizer.bos_id] +
            self.tokenizer.text_to_ids(system_prompt) +
            [self.tokenizer.eos_id]
        )
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, prompt_len]
        prompt_len = prompt_tokens.shape[1]

        logging.info(f"   Prompt length: {prompt_len} tokens")

        # Step 2: Embed the prompt tokens (this acts as the "audio channel" for prompt positions)
        prompt_embedded = self.model.stt_model.embed_tokens(prompt_tokens)  # [1, prompt_len, H]
        prompt_embedded = prompt_embedded.to(dtype=self.dtype)

        # Step 3: Add pad embeddings for text and ASR channels (for positions t > 0)
        # In offline_inference, prompt positions use gen_text[:, t-1] = pad_id
        pad_id = self.model.stt_model.text_pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
        pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)  # [1, H]
        pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)  # [1, H]

        # For positions t > 0, add pad embeddings (simulating gen_text[:, t-1] = pad_id)
        has_fc = self.model.stt_model.function_head is not None
        if prompt_len > 1:
            prompt_embedded[:, 1:, :] += pad_emb
            prompt_embedded[:, 1:, :] += pad_asr_emb
            if has_fc:
                prompt_embedded[:, 1:, :] += pad_emb  # FC channel also uses pad at t > 0

        # Step 4: For position 0, add BOS embeddings
        bos_emb = self._get_bos_embedding()  # [1, H]
        asr_bos_emb = self._get_asr_bos_embedding()  # [1, H]
        prompt_embedded[:, 0, :] += bos_emb.squeeze(0)
        prompt_embedded[:, 0, :] += asr_bos_emb.squeeze(0)
        if has_fc:
            prompt_embedded[:, 0, :] += pad_emb.squeeze(0)  # FC channel uses pad at t=0

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

        tts_dev = getattr(self, 'tts_device', self.device)
        speaker_audio = speaker_audio.to(tts_dev)
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)], device=tts_dev).long()

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
                       rnnt_state: Optional[dict] = None):

        # Set up effective request ID for vLLM streaming
        effective_request_id = request_id or self.request_id

        start_time_one_step = time.time()
        use_cache = dynamic_cache is not None
        batch_size = gen_text.shape[0]

        predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)
        asr_predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)
        function_predicted_tokens = torch.empty((batch_size, num_frames_per_chunk), dtype=gen_text.dtype, device=gen_text.device)

        # Do "perception" step outside the for-loop
        start_perception = time.time()

        used_perception_cache = (self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized())
        encoder_emb_raw = None  # raw [B, T, 1024] for RNNT EOU/BOU

        if used_perception_cache:
            # Cache-aware perception
            source_encoded, encoder_emb_raw, perception_cache = self.perception_cache_mgr.step(
                audio_input=audio_input,
                frame_idx=frame_idx,
                num_frames_per_chunk=num_frames_per_chunk,
                perception_cache=perception_cache,
            )
        else:
            # Standard perception (full buffer processing)
            buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
            source_encoded, _, encoder_emb_raw = self.model.stt_model.perception(
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

            current_input_emb = current_frame_embedding.clone()

            has_fc = gen_function_text is not None

            if current_frame_idx == 0 and not has_prompt:
                # Only add BOS if there's no prompt (BOS is already in prompt's position 0)
                current_input_emb += self._get_bos_embedding()
                current_input_emb += self._get_asr_bos_embedding()
                if has_fc:
                    pad_id = self.model.stt_model.text_pad_id
                    fc_pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                    current_input_emb += self.model.stt_model.embed_tokens(fc_pad_token).to(dtype=self.dtype)
            elif current_frame_idx == 0 and has_prompt:
                # With prompt: first audio frame uses pad embedding (like offline_inference)
                # gen_text[:, -1] from prompt positions is pad_id
                pad_id = self.model.stt_model.text_pad_id
                pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
                pad_emb = self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
                pad_asr_emb = self.model.stt_model.embed_asr_tokens(pad_token).to(dtype=self.dtype)
                current_input_emb += pad_emb
                current_input_emb += pad_asr_emb
                if has_fc:
                    current_input_emb += self.model.stt_model.embed_tokens(pad_token).to(dtype=self.dtype)
            else:
                # t > 0: add embeddings from model's own predictions at t-1
                _vocab = self.model.stt_model.embed_tokens.weight.shape[0]
                last_token_emb = self.model.stt_model.embed_tokens(gen_text[:, current_frame_idx - 1].clamp(0, _vocab - 1))
                current_input_emb += last_token_emb
                _asr_vocab = self.model.stt_model.embed_asr_tokens.weight.shape[0]
                last_asr_token_emb = self.model.stt_model.embed_asr_tokens(gen_asr_text[:, current_frame_idx - 1].clamp(0, _asr_vocab - 1))
                current_input_emb += last_asr_token_emb
                if has_fc:
                    last_fc_token_emb = self.model.stt_model.embed_tokens(gen_function_text[:, current_frame_idx - 1].clamp(0, _vocab - 1))
                    current_input_emb += last_fc_token_emb.to(dtype=self.dtype)

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

            # Clamp to embedding table size — GA FC checkpoint returns vLLM text tokens in
            # the ASR slot (no real separate ASR head); RNNT provides turn-taking separately.
            _asr_vocab = self.model.stt_model.embed_asr_tokens.weight.shape[0]
            asr_predicted_token = asr_predicted_token.clamp(0, _asr_vocab - 1)
            gen_asr_text[:, current_frame_idx] = asr_predicted_token
            asr_predicted_tokens[:, frame_offset] = asr_predicted_token

            if "function_predicted_token" in ans:
                _fc_vocab = self.model.stt_model.embed_tokens.weight.shape[0]
                fc_token = ans["function_predicted_token"].clamp(0, _fc_vocab - 1)
                function_predicted_tokens[:, frame_offset] = fc_token
                if gen_function_text is not None:
                    gen_function_text[:, current_frame_idx] = fc_token

            # Apply forced turn taking based on ASR results
            self._maybe_apply_forced_turn_taking(current_frame_idx, gen_text, gen_asr_text)
            # Update predicted_tokens with any changes made by forced turn taking
            predicted_tokens[:, frame_offset] = gen_text[:, current_frame_idx]

            # RNNT-based EOU/BOU turn-taking
            if rnnt_state is not None and encoder_emb_raw is not None:
                # encoder_emb_raw format depends on code path:
                #   non-cache: perception.py returns [B, T, D] (transposed at source)
                #   cache:     perception_cache.py returns [B, D, T] (channels-first, NOT transposed)
                # Normalise to [B, T, D] so frame extraction is consistent.
                rnnt_enc = encoder_emb_raw.transpose(1, 2) if used_perception_cache else encoder_emb_raw
                emb_idx = frame_offset if used_perception_cache else current_frame_index
                emb_idx = min(emb_idx, rnnt_enc.shape[1] - 1)
                rnnt_frame = rnnt_enc[:, emb_idx, :]  # [B, D]
                tok_before = gen_text[0, current_frame_idx].item()
                rnnt_state, rnnt_is_blank = self._rnnt_step(rnnt_frame, rnnt_state)
                self._apply_rnnt_turn_taking(current_frame_idx, gen_text, rnnt_is_blank, rnnt_state)
                tok_after = gen_text[0, current_frame_idx].item()
                predicted_tokens[:, frame_offset] = gen_text[:, current_frame_idx]
                # Print when RNNT EOU/BOU actively changed the voicechat response token
                if tok_after != tok_before:
                    import datetime as _dt
                    bos_id = self.model.stt_model.text_bos_id
                    eos_id = self.model.stt_model.text_eos_id
                    _evt = ("BOS→agent-start" if tok_after == bos_id else
                            "EOS→agent-stop(barge-in)" if tok_after == eos_id else f"tok={tok_after}")
                    _t_sec = round(current_frame_idx * 0.08, 2)
                    _density_val = rnnt_state['rolling_density'][0].item()
                    print(
                        f"[RNNT-CTRL {_dt.datetime.now().isoformat()}] "
                        f"frame={current_frame_idx} t={_t_sec}s "
                        f"RNNT overrode LLM: llm_tok={tok_before} → {_evt} "
                        f"(blank={rnnt_is_blank.item()}, "
                        f"blank_cnt={rnnt_state['blank_count'][0].item()}, "
                        f"nonblank_total={rnnt_state['nonblank_total'][0].item()}, "
                        f"density={_density_val:.3f})",
                        flush=True
                    )

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
                # If TTS is on a different device, move tensors there and back.
                tts_dev = getattr(self, 'tts_device', self.device)
                tts_xdev = (tts_dev != self.device)
                inputs = {
                    "current_subword_id": current_subword_id.to(tts_dev),
                    "prev_subword_id": prev_subword_id.to(tts_dev),
                    "current_subword_mask": current_subword_mask.to(tts_dev),
                    "prev_audio_tokens": code.to(tts_dev) if code is not None else None,
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

                # Move outputs back to main device if TTS was on a different device.
                if tts_xdev:
                    code = code.to(self.device)

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
            'rnnt_state': rnnt_state,
        }
        if self.model.stt_model.function_head is not None:
            result['function_predicted_text_tokens'] = function_predicted_tokens
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


    def _maybe_apply_forced_turn_taking(self, t, gen_text, gen_asr):
        """Apply forced turn-taking rules based on ASR channel tokens."""
        if not self.model_cfg.get("force_turn_taking", False):
            return

        threshold = self.model_cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.model_cfg.get("force_turn_taking_pad_window", 25)

        B = gen_text.size(0)

        for batch_idx in range(B):
            lookback_start = max(0, t - threshold)
            agent_text_window = gen_text[batch_idx, lookback_start:t]
            current_asr_token = gen_asr[batch_idx, t]

            # ASR EOS or ~1 sec of pad tokens → insert agent BOS if not present in window
            # Skip if we don't have enough tokens at the beginning
            if t < pad_window_steps:
                continue

            pad_lookback_start = t - pad_window_steps
            asr_recent_tokens = gen_asr[batch_idx, pad_lookback_start:t]
            has_pad_window = (asr_recent_tokens == self.model.stt_model.text_pad_id).all() if len(asr_recent_tokens) > 0 else False

            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = gen_asr[batch_idx, pad_lookback_start - 1]
                has_pad_window = (token_before_window != self.model.stt_model.text_pad_id) and (token_before_window != self.model.stt_model.user_bos_id)
            elif has_pad_window and pad_lookback_start == 0:
                # If the pad window starts at position 0, it doesn't meet the requirement
                has_pad_window = False

            if has_pad_window:
                if not (agent_text_window == self.model.stt_model.text_bos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_bos_id
                    logging.info(f"Forced turn-taking at frame {t}: inserted agent BOS (reason: pad window)")

            # ASR BOS → insert agent EOS if not present in window
            elif current_asr_token == self.model.stt_model.user_bos_id:
                if not (agent_text_window == self.model.stt_model.text_eos_id).any():
                    gen_text[batch_idx, t] = self.model.stt_model.text_eos_id
                    logging.info(f"Forced turn-taking at frame {t}: inserted agent EOS (reason: user started speaking)")

    def _rnnt_init_state(self, B: int, device) -> dict:
        """Initialize RNNT streaming state for EOU/BOU detection."""
        return {
            'pred_out':         None,
            'pred_hidden':      None,
            'blank_count':      torch.zeros(B, dtype=torch.long, device=device),
            'nonblank_consec':  torch.zeros(B, dtype=torch.long, device=device),
            # nonblank_total: cumulative non-blank frames since last reset. Resets on EOU,
            # agent-speaking, or after long silence (to prevent noise accumulation).
            # Unlike nonblank_consec, blanks mid-word don't reset it, so short words like
            # "hello" still accumulate enough count to confirm real speech.
            'nonblank_total':   torch.zeros(B, dtype=torch.long, device=device),
            # speech_confirmed: True when real user speech detected (not echo/noise).
            # Set when nonblank_consec OR nonblank_total reaches the speech threshold.
            'speech_confirmed': torch.zeros(B, dtype=torch.bool, device=device),
            # Persistent agent-speaking flag — not window-bounded.
            'agent_speaking':   torch.zeros(B, dtype=torch.bool, device=device),
            # First turn: True until agent speaks for the first time. Uses lower thresholds
            # so the user doesn't need to say many words before the agent responds.
            'first_turn':       torch.ones(B, dtype=torch.bool, device=device),
            # rolling_density: exponential moving average of non-blank rate (0.0–1.0).
            # Tracks how speech-dense this speaker is. Low density → accent/noise/sparse
            # speech → adaptive lower min_speech_frames threshold.
            'rolling_density':  torch.zeros(B, dtype=torch.float32, device=device),
            # post_eos_fired: True after post_eos_fallback fires once for the current agent turn.
            # Prevents repeated post_eos triggers to silence. Resets when agent starts speaking.
            'post_eos_fired':   torch.zeros(B, dtype=torch.bool, device=device),
        }

    @torch.no_grad()
    def _rnnt_step(self, encoder_frame: torch.Tensor, rnnt_state: dict):
        """Run one RNNT step. Returns (new_state, is_blank [B])."""
        decoder  = getattr(self.model.stt_model, '_rnnt_decoder', None)
        joint    = getattr(self.model.stt_model, '_rnnt_joint', None)
        blank_id = getattr(self.model.stt_model, '_rnnt_blank_id', 1024)
        if decoder is None or joint is None:
            is_blank = torch.ones(encoder_frame.shape[0], dtype=torch.bool, device=encoder_frame.device)
            return rnnt_state, is_blank

        B = encoder_frame.shape[0]
        f = encoder_frame.float().unsqueeze(1)  # [B, 1, 1024]

        pred_out    = rnnt_state['pred_out']
        pred_hidden = rnnt_state['pred_hidden']
        if pred_out is None:
            pred_out, pred_hidden = decoder.predict(y=None, state=None, add_sos=True, batch_size=B)

        # add_sos=True may return pred_out with T>1 (e.g. [B, 2, D]).
        # The RNNT joint expects [B, 1, D] — keep only the last decoder step.
        if pred_out.dim() == 3 and pred_out.shape[1] > 1:
            pred_out = pred_out[:, -1:, :]   # [B, 1, D]

        logits = joint.joint(f, pred_out)           # [B, 1, 1, V]
        tokens = logits.squeeze(1).squeeze(1).argmax(-1)  # [B]
        is_blank = (tokens == blank_id)

        new_pred_out    = pred_out
        new_pred_hidden = pred_hidden
        if not is_blank.all():
            non_blank_tokens = torch.where(is_blank, torch.full_like(tokens, blank_id), tokens)
            y = non_blank_tokens.unsqueeze(1)  # [B, 1]
            try:
                new_pred_out_cand, new_pred_hidden_cand = decoder.predict(
                    y=y, state=pred_hidden, add_sos=False, batch_size=B
                )
                if B == 1:
                    if not is_blank[0]:
                        new_pred_out    = new_pred_out_cand
                        new_pred_hidden = new_pred_hidden_cand
                else:
                    mask = is_blank.view(B, 1, 1).expand_as(pred_out)
                    new_pred_out    = torch.where(mask, pred_out, new_pred_out_cand)
                    new_pred_hidden = new_pred_hidden_cand
            except Exception as _e:
                # Decoder state update failed (e.g., hidden-state shape mismatch on first
                # non-blank token). EOU blank/nonblank counting still works correctly;
                # only the decoder conditioning is affected.
                logging.warning(f"RNNT decoder update skipped: {_e}")

        # EMA density: track fraction of non-blank frames (speaker-level, not reset on turns).
        # Alpha=0.1 means ~10-frame (0.8s) effective window for density estimation.
        # During agent speech, RNNT sees the agent's TTS audio — we want user density only,
        # so only update density when agent is NOT speaking.
        density_alpha = 0.1
        is_speech_float = (~is_blank).float()
        if rnnt_state.get('agent_speaking') is not None and rnnt_state['agent_speaking'].any():
            # Don't corrupt density estimate with agent's own TTS audio
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
            # nonblank_total increments on non-blank, holds on blank (reset by _apply_rnnt_turn_taking)
            'nonblank_total':   torch.where(is_blank,
                                            rnnt_state['nonblank_total'],
                                            rnnt_state['nonblank_total'] + 1),
            # speech_confirmed, agent_speaking, and first_turn are managed in _apply_rnnt_turn_taking
            'speech_confirmed': rnnt_state['speech_confirmed'],
            'agent_speaking':   rnnt_state['agent_speaking'],
            'first_turn':       rnnt_state['first_turn'],
            'rolling_density':  new_density,
            'post_eos_fired':   rnnt_state['post_eos_fired'],
        }
        return new_state, is_blank

    def _apply_rnnt_turn_taking(self, t: int, gen_text: torch.Tensor,
                                 is_blank: torch.Tensor, rnnt_state: dict) -> None:
        """Apply RNNT blank/non-blank counts to trigger agent BOS (EOU) or EOS (BOU)."""
        if not self.model_cfg.get("rnnt_eou_enabled", False):
            return

        asr_eou          = int(self.model_cfg.get("asr_eou", 4))           # blank frames → EOU (N × 80ms silence)
        asr_min_speech   = int(self.model_cfg.get("asr_min_speech_frames", 3))  # min speech frames before EOU can fire (noise guard)
        user_bos_frames  = int(self.model_cfg.get("user_bos_frames",       # consecutive non-blank frames → barge-in
                                self.model_cfg.get("asr_bou", 4)))
        threshold = int(self.model_cfg.get("force_turn_taking_threshold", 40))
        bos_id   = self.model.stt_model.text_bos_id
        eos_id   = self.model.stt_model.text_eos_id

        B = gen_text.size(0)
        for b in range(B):
            lookback_start = max(0, t - threshold)
            agent_window   = gen_text[b, lookback_start:t]

            blank_cnt        = rnnt_state['blank_count'][b].item()
            nonblank_cnt     = rnnt_state['nonblank_consec'][b].item()
            nonblank_total   = rnnt_state['nonblank_total'][b].item()
            speech_confirmed = rnnt_state['speech_confirmed'][b].item()
            first_turn       = rnnt_state['first_turn'][b].item()

            # First-turn thresholds: before agent ever speaks, use lower bar so a single
            # short word like "hello" + brief pause triggers the response.
            # After first turn: use normal (stricter) configured thresholds.
            if first_turn:
                effective_min_speech = int(self.model_cfg.get("asr_min_speech_frames_first_turn", 2))
                effective_eou        = int(self.model_cfg.get("asr_eou_first_turn", 2))
            else:
                effective_min_speech = asr_min_speech
                effective_eou        = asr_eou

            # Adaptive speech density: if this speaker produces sparse non-blank tokens
            # (accent, noise, synthetic speech), lower effective_min_speech automatically.
            # rolling_density is EMA of non-blank rate; values <density_threshold indicate
            # a sparse-speech speaker who needs a lower confirmation threshold.
            density_threshold  = float(self.model_cfg.get("density_speech_threshold", 0.15))
            density_low_min    = int(self.model_cfg.get("density_low_min_speech", 1))
            rolling_density    = rnnt_state['rolling_density'][b].item()
            density_adapted    = False
            if not first_turn and rolling_density < density_threshold and rolling_density > 0.0:
                effective_min_speech = density_low_min
                density_adapted = True

            # Sync persistent agent_speaking flag from BOS/EOS in the recent window.
            # Any BOS in the window (from RNNT EOU or force_turn_taking) ends the first turn.
            agent_speaking = rnnt_state['agent_speaking'][b].item()
            if (agent_window == bos_id).any():
                agent_speaking = True
                rnnt_state['agent_speaking'][b] = True
                rnnt_state['first_turn'][b] = False
                first_turn = False
            if (agent_window == eos_id).any():
                agent_speaking = False
                rnnt_state['agent_speaking'][b] = False
                # Turn-boundary reset: clear accumulated signal from during agent speech.
                # This is not echo suppression — it's a clean slate for the next user turn.
                # Without this, nonblank_total from noise/signal during agent speech would
                # immediately re-trigger EOU the moment the agent stops.
                rnnt_state['speech_confirmed'][b] = False
                rnnt_state['nonblank_total'][b] = 0
                speech_confirmed = False
                nonblank_total = 0

            # Noise-accumulation guard: after long silence (10 × 80ms = 800ms) without
            # confirmed speech, reset nonblank_total so isolated noise spikes don't
            # slowly add up across minutes to falsely trigger speech_confirmed.
            noise_reset_frames = int(self.model_cfg.get("nonblank_reset_after_silence", 10))
            if blank_cnt >= noise_reset_frames and not speech_confirmed and not agent_speaking:
                rnnt_state['nonblank_total'][b] = 0
                nonblank_total = 0

            # Confirm real user speech when (a) N consecutive non-blank frames (strict),
            # OR (b) N total non-blank frames since last reset (catches short words where
            # RNNT inserts blanks between phonemes, breaking the consecutive streak).
            if (nonblank_cnt >= effective_min_speech or nonblank_total >= effective_min_speech) and not agent_speaking:
                rnnt_state['speech_confirmed'][b] = True
                speech_confirmed = True

            # First-turn timer fallback: if the user hasn't triggered EOU via RNNT after
            # first_turn_fallback_frames (default 50 × 80ms = 4s), inject BOS anyway so
            # the agent always responds on the first turn even if RNNT misses the speech.
            # This is force_turn_taking scoped to first_turn only — safe alongside RNNT.
            first_turn_fallback = int(self.model_cfg.get("first_turn_fallback_frames", 50))
            if first_turn and t >= first_turn_fallback and speech_confirmed and not (agent_window == bos_id).any():
                gen_text[b, t] = bos_id
                rnnt_state['speech_confirmed'][b] = False
                rnnt_state['nonblank_total'][b] = 0
                rnnt_state['agent_speaking'][b] = True
                rnnt_state['first_turn'][b] = False
                logging.info(
                    f"RNNT first-turn fallback t={t}: agent BOS "
                    f"(no EOU detected in {first_turn_fallback} frames)"
                )
                return

            # EOU: N blank frames after confirmed user speech → inject agent BOS.
            if blank_cnt >= effective_eou and speech_confirmed and not agent_speaking:
                if not (agent_window == bos_id).any():
                    # CONFLICT CHECK: RNNT is about to overwrite a non-pad LLM token.
                    # This happens when the LLM backbone already placed a speech/text token
                    # at frame t while RNNT is trying to write BOS — the two disagree on
                    # what should happen at this frame.
                    existing_tok = gen_text[b, t].item()
                    _pad_id = self.model.stt_model.text_pad_id
                    if existing_tok != _pad_id and existing_tok != bos_id:
                        import datetime
                        _conflict_msg = (
                            f"CONFLICT [RNNT-EOU overwrites LLM token] t={t} t_sec={t*0.08:.2f}: "
                            f"LLM placed token={existing_tok} at frame {t}, "
                            f"RNNT overwriting with BOS (blank_cnt={blank_cnt}, "
                            f"nonblank_total={nonblank_total})"
                        )
                        logging.warning(_conflict_msg)
                        if getattr(self, '_rnnt_eou_log_path', None):
                            with open(self._rnnt_eou_log_path, 'a') as _lf:
                                _lf.write(
                                    f"{datetime.datetime.now().isoformat()} | CONFLICT_EOU_OVERWRITE | frame={t} | "
                                    f"t_sec={t*0.08:.2f} | llm_token={existing_tok} | "
                                    f"blank_cnt={blank_cnt} | nonblank_total={nonblank_total}\n"
                                )
                    gen_text[b, t] = bos_id
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    rnnt_state['agent_speaking'][b] = True
                    rnnt_state['first_turn'][b] = False
                    _density_note = (
                        f" [ADAPTIVE density={rolling_density:.3f}<{density_threshold} → min_speech={density_low_min}]"
                        if density_adapted else ""
                    )
                    _msg = (
                        f"RNNT EOU t={t}: agent BOS "
                        f"(first_turn={first_turn}, blank_count={blank_cnt}, "
                        f"nonblank_consec={nonblank_cnt}, nonblank_total={nonblank_total}, "
                        f"density={rolling_density:.3f}){_density_note}"
                    )
                    logging.info(_msg)
                    if getattr(self, '_rnnt_eou_log_path', None):
                        import datetime
                        with open(self._rnnt_eou_log_path, 'a') as _lf:
                            _lf.write(
                                f"{datetime.datetime.now().isoformat()} | EOU | frame={t} | "
                                f"t_sec={t*0.08:.2f} | blank_cnt={blank_cnt} | "
                                f"nonblank_total={nonblank_total} | first_turn={first_turn} | "
                                f"density={rolling_density:.3f} | adapted={density_adapted}\n"
                            )
                else:
                    # CONFLICT: RNNT EOU would fire but LLM backbone already placed BOS in
                    # the lookback window (premature LLM BOS suppresses the RNNT EOU).
                    # This is the primary failure mode when force_turn_taking fires at 0.72s
                    # before the user finishes speaking.
                    import datetime
                    _llm_bos_offsets = (agent_window == bos_id).nonzero(as_tuple=True)[0].tolist()
                    _llm_bos_frames = [lookback_start + off for off in _llm_bos_offsets]
                    _conflict_msg = (
                        f"CONFLICT [RNNT-EOU suppressed by LLM-BOS] t={t} t_sec={t*0.08:.2f}: "
                        f"LLM already has BOS at frames {_llm_bos_frames} "
                        f"(t_sec={[f*0.08 for f in _llm_bos_frames]}), "
                        f"RNNT wanted EOU (blank_cnt={blank_cnt}, nonblank_total={nonblank_total})"
                    )
                    logging.warning(_conflict_msg)
                    if getattr(self, '_rnnt_eou_log_path', None):
                        with open(self._rnnt_eou_log_path, 'a') as _lf:
                            _lf.write(
                                f"{datetime.datetime.now().isoformat()} | CONFLICT_EOU_SUPPRESSED | frame={t} | "
                                f"t_sec={t*0.08:.2f} | blank_cnt={blank_cnt} | "
                                f"nonblank_total={nonblank_total} | "
                                f"llm_bos_frames={_llm_bos_frames} | "
                                f"llm_bos_t_sec={[round(f*0.08,2) for f in _llm_bos_frames]}\n"
                            )
                return  # EOU takes priority over user BOS interrupt

            # Force-EOS: if the agent has been speaking for too long (blank_cnt >= threshold)
            # with zero user speech detected (speech_confirmed=False), the LLM is generating
            # indefinitely without natural EOS (common for zero-RNNT-detection speakers like
            # Kevin/Edresson/Moshi whose audio produces 0 RNNT non-blank tokens). Inject EOS to
            # stop the agent turn so that post_eos_fallback can then trigger a new BOS response.
            # NOT triggered for control speakers: their barge-ins reset blank_cnt before threshold.
            force_eos_frames = int(self.model_cfg.get("force_eos_after_frames", 0))  # 0=disabled
            if (force_eos_frames > 0 and agent_speaking and not speech_confirmed
                    and not first_turn and not (agent_window == eos_id).any()):
                if blank_cnt >= force_eos_frames:
                    import datetime as _dt_fef
                    gen_text[b, t] = eos_id
                    rnnt_state['agent_speaking'][b] = False
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    _fef_msg = (
                        f"RNNT force-EOS t={t}: agent EOS injected "
                        f"(agent speaking > {force_eos_frames} blank frames with no user speech, "
                        f"blank_cnt={blank_cnt}, density={rolling_density:.3f})"
                    )
                    logging.info(_fef_msg)
                    print(
                        f"[RNNT-CTRL {_dt_fef.datetime.now().isoformat()}] "
                        f"frame={t} t={t*0.08:.2f}s "
                        f"FORCE-EOS: agent stopped (no user speech in {blank_cnt} blanks, "
                        f"density={rolling_density:.3f})",
                        flush=True
                    )
                    if getattr(self, '_rnnt_eou_log_path', None):
                        with open(self._rnnt_eou_log_path, 'a') as _lf:
                            _lf.write(
                                f"{_dt_fef.datetime.now().isoformat()} | FORCE_EOS | frame={t} | "
                                f"t_sec={t*0.08:.2f} | blank_cnt={blank_cnt} | "
                                f"density={rolling_density:.3f} | adapted={density_adapted}\n"
                            )
                    return

            # Post-EOS fallback: if agent finished speaking and blank_count accumulates past
            # post_eos_fallback_frames without ANY confirmed user speech, inject BOS unconditionally.
            # This handles speakers whose acoustic properties produce 0 RNNT non-blank tokens
            # (e.g., heavy accent, synthetic speech) — Kratos can't detect them, so we fall back
            # to a time-based response trigger. Only fires when speech_confirmed=False (if RNNT
            # DID detect user speech, the normal EOU path above handles it).
            post_eos_fallback = int(self.model_cfg.get("post_eos_fallback_frames", 0))  # 0=disabled
            if (post_eos_fallback > 0 and not first_turn and not agent_speaking
                    and not speech_confirmed and not rnnt_state['post_eos_fired'][b]):
                if blank_cnt >= post_eos_fallback:
                    import datetime as _dt_pef
                    gen_text[b, t] = bos_id
                    rnnt_state['speech_confirmed'][b] = False
                    rnnt_state['nonblank_total'][b] = 0
                    rnnt_state['agent_speaking'][b] = True
                    rnnt_state['first_turn'][b] = False
                    rnnt_state['post_eos_fired'][b] = True  # fire at most once per agent turn
                    _pef_msg = (
                        f"RNNT post-EOS fallback t={t}: agent BOS "
                        f"(blank_cnt={blank_cnt}>={post_eos_fallback}, "
                        f"density={rolling_density:.3f}, speech_confirmed=False)"
                    )
                    logging.info(_pef_msg)
                    print(
                        f"[RNNT-CTRL {_dt_pef.datetime.now().isoformat()}] "
                        f"frame={t} t={t*0.08:.2f}s "
                        f"POST-EOS fallback: BOS injected (no speech detected in {blank_cnt} blanks, "
                        f"density={rolling_density:.3f})",
                        flush=True
                    )
                    if getattr(self, '_rnnt_eou_log_path', None):
                        with open(self._rnnt_eou_log_path, 'a') as _lf:
                            _lf.write(
                                f"{_dt_pef.datetime.now().isoformat()} | POST_EOS_FALLBACK | frame={t} | "
                                f"t_sec={t*0.08:.2f} | blank_cnt={blank_cnt} | "
                                f"density={rolling_density:.3f} | adapted={density_adapted}\n"
                            )
                    return

            # BOS suppression: after post_eos fired once and RNNT still detects no user speech,
            # suppress any LLM-native BOS tokens that would start hallucinated self-play turns.
            # post_eos already gave the agent one response; additional LLM-generated BOS tokens
            # are a runaway conversation loop — overwrite with pad to stop generation.
            # speech_confirmed=True bypasses suppression so barge-ins from real users still work.
            if (post_eos_fallback > 0 and not first_turn and not speech_confirmed
                    and rnnt_state['post_eos_fired'][b]):
                _pad_id = self.model.stt_model.text_pad_id
                if gen_text[b, t].item() == bos_id:
                    gen_text[b, t] = _pad_id
                    rnnt_state['agent_speaking'][b] = False
                    return

            # Barge-in: N consecutive non-blank frames while agent speaking → agent EOS.
            if nonblank_cnt >= user_bos_frames and agent_speaking:
                if not (agent_window == eos_id).any():
                    # CONFLICT CHECK: RNNT barge-in about to overwrite a non-pad LLM token.
                    existing_tok = gen_text[b, t].item()
                    _pad_id = self.model.stt_model.text_pad_id
                    if existing_tok != _pad_id and existing_tok != eos_id:
                        import datetime
                        _conflict_msg = (
                            f"CONFLICT [RNNT-barge-in overwrites LLM token] t={t} t_sec={t*0.08:.2f}: "
                            f"LLM placed token={existing_tok}, RNNT overwriting with EOS "
                            f"(nonblank_consec={nonblank_cnt})"
                        )
                        logging.warning(_conflict_msg)
                        if getattr(self, '_rnnt_eou_log_path', None):
                            with open(self._rnnt_eou_log_path, 'a') as _lf:
                                _lf.write(
                                    f"{datetime.datetime.now().isoformat()} | CONFLICT_BARGEIN_OVERWRITE | frame={t} | "
                                    f"t_sec={t*0.08:.2f} | llm_token={existing_tok} | "
                                    f"nonblank_consec={nonblank_cnt}\n"
                                )
                    gen_text[b, t] = eos_id
                    rnnt_state['nonblank_consec'][b] = 0
                    rnnt_state['nonblank_total'][b] = 0
                    rnnt_state['agent_speaking'][b] = False
                    logging.info(
                        f"RNNT barge-in t={t}: agent EOS "
                        f"(nonblank_consec={nonblank_cnt}, user_bos_frames={user_bos_frames})"
                    )
                else:
                    # CONFLICT: barge-in EOS suppressed because LLM already placed EOS in window.
                    import datetime
                    _llm_eos_offsets = (agent_window == eos_id).nonzero(as_tuple=True)[0].tolist()
                    _llm_eos_frames = [lookback_start + off for off in _llm_eos_offsets]
                    _conflict_msg = (
                        f"CONFLICT [RNNT-barge-in suppressed by LLM-EOS] t={t} t_sec={t*0.08:.2f}: "
                        f"LLM already has EOS at frames {_llm_eos_frames}, "
                        f"RNNT wanted barge-in EOS (nonblank_consec={nonblank_cnt})"
                    )
                    logging.warning(_conflict_msg)
                    if getattr(self, '_rnnt_eou_log_path', None):
                        with open(self._rnnt_eou_log_path, 'a') as _lf:
                            _lf.write(
                                f"{datetime.datetime.now().isoformat()} | CONFLICT_BARGEIN_SUPPRESSED | frame={t} | "
                                f"t_sec={t*0.08:.2f} | nonblank_consec={nonblank_cnt} | "
                                f"llm_eos_frames={_llm_eos_frames}\n"
                            )

    @torch.no_grad()
    def inference_realtime_streaming(self, audio_path: str, num_frames_per_chunk: int = None, request_id: Optional[str] = None, pad_audio_to_sec: Optional[float] = None, pad_silence_ratio: Optional[float] = None, pad_audio_by_sec: Optional[float] = None, system_prompt: Optional[str] = None):
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
            subword_mask = torch.ones(1, total_frames, device=self.device, dtype=torch.bool)
            logging.info(f"TTS initialized")

        # Initialize perception cache if enabled
        perception_cache = None
        if self.use_perception_cache:
            perception_cache = self.perception_cache_mgr.get_initial_state(batch_size=1)
            logging.info(f"Perception cache initialized")

        # Initialize RNNT EOU/BOU state
        rnnt_state = None
        if hasattr(self.model.stt_model, '_rnnt_decoder') and self.model_cfg.get('rnnt_eou_enabled', False):
            rnnt_state = self._rnnt_init_state(1, self.device)
            logging.info(
                f"RNNT EOU/BOU enabled: asr_eou={self.model_cfg.get('asr_eou',4)} frames "
                f"({self.model_cfg.get('asr_eou',4)*80}ms silence→EOU), "
                f"user_bos_frames={self.model_cfg.get('user_bos_frames', self.model_cfg.get('asr_bou',4))} frames "
                f"({self.model_cfg.get('user_bos_frames', self.model_cfg.get('asr_bou',4))*80}ms speech→interrupt)"
            )

        # Initialize codec streaming cache to remove clicking sounds and wasted inference computation
        codec_cache = None
        if self.decode_audio and self.use_codec_cache:
            from nemo.collections.speechlm2.modules.ear_tts_vae_codec import CausalConv1dCache
            codec_cache = CausalConv1dCache()
            logging.info(f"Codec streaming cache initialized")

        gen_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)
        has_function_head = self.model.stt_model.function_head is not None
        if has_function_head:
            gen_function_text = torch.full((1, total_frames), self.model.stt_model.text_pad_id, device=self.device, dtype=torch.long)

        # initialize list to which we will append generated audio segments
        audio_segments = []

        logging.info("\n" + "=" * 70)
        logging.info("STARTING FRAME-BY-FRAME PROCESSING")
        logging.info("=" * 70)

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
                rnnt_state=rnnt_state,
            )

            # handle results from infer_one_step
            if has_function_head and 'function_predicted_text_tokens' in result:
                for fi in range(num_frames_per_chunk):
                    gen_function_text[:, frame_idx + fi] = result['function_predicted_text_tokens'][:, fi]
            input_embeds_history = result['input_embeds_history']
            llm_cache = result['dynamic_cache']
            if self.use_perception_cache:
                perception_cache = result.get('perception_cache', perception_cache)
            rnnt_state = result.get('rnnt_state', rnnt_state)
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
        logging.info("\n" + "=" * 70)
        logging.info("STREAMING INFERENCE COMPLETED")
        logging.info("=" * 70)
        logging.info(f"Total time: {elapsed_time:.2f}s")
        logging.info(f"Audio duration: {total_duration:.2f}s")
        logging.info(f"RTF (Real-Time Factor): {elapsed_time / total_duration:.2f}x")
        logging.info(f"Processed frames: {total_frames}")

        # Trim to actual length
        # TODO: this is currently redundant since we iterate over all frames in the while loop
        gen_text = gen_text[:, :total_frames]
        gen_asr_text = gen_asr_text[:, :total_frames]

        # Decode text
        lengths = torch.tensor([total_frames], dtype=torch.long, device=self.device)
        text_output = tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        # Decode ASR text
        asr_text_output = tokens_to_str(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id, eval_text_turn_taking=True)

        # Also create raw versions with <SPECIAL_12> kept for comparison
        text_output_raw = tokens_to_str_raw(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)
        asr_text_output_raw = tokens_to_str_raw(gen_asr_text, lengths, tokenizer=self.tokenizer, pad_id=self.model.stt_model.text_pad_id)

        logging.info(f"Generated text: {text_output[0]}")
        logging.info(f"Generated ASR text: {asr_text_output[0]}")

        # Decode function calling channel
        if has_function_head:
            gen_function_text = gen_function_text[:, :total_frames]
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

                try:
                    cleaned_pred = clean_pred_text(pred_asr_text)
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
                    'pred_text': pred_text,
                    'system_prompt': result_system_prompt,
                }

                output_record_raw = {
                    'id': audio_id,
                    'target_text': '',
                    'pred_audio': pred_audio_path,
                    'src_text': ground_truth_text,
                    'pred_src_text': pred_asr_text_raw,
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

