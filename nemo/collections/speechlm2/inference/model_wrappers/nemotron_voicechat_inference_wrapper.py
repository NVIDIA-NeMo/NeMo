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

import copy
import time

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.speechlm2.inference.model_wrappers.backend.llm import LlmStepResult
from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.eartts import PyTorchEarTTS
from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.llm import PyTorchLLM
from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.eartts import VllmEarTTS
from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.llm import VllmLLM
from nemo.collections.speechlm2.inference.model_wrappers.capabilities import (
    AuxiliaryOutputCapabilities,
    derive_auxiliary_output_capabilities,
)
from nemo.collections.speechlm2.inference.model_wrappers.config_overrides import apply_model_cfg_overrides
from nemo.collections.speechlm2.inference.model_wrappers.decode_state import (
    InferenceStepResult,
    IntermediateResultLogger,
    NullIntermediateResultLogger,
    NullTimingSummary,
    StreamingDecodeState,
    TimingSummary,
)
from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import (
    VLLM_OMNI,
    native_weight_skip_prefixes,
    precision_matches_cfg,
    reject_unimplemented_vllm,
    reject_unsupported_determinism,
    resolve_engine_types,
)
from nemo.collections.speechlm2.inference.model_wrappers.codec import AudioCodec
from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
    PerceptionCacheManager,
    PerceptionCacheState,
)
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.collections.speechlm2.parts.text_utils import (
    _decode_tokens_with_specials,
    get_special_token_ids,
    get_special_token_strings,
)
from nemo.utils import logging, str_to_dtype

# --- Configuration ---
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming Parameters ---
SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08  # 80ms per frame
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)  # 1280 samples

TTS_SAMPLE_RATE = 22050


class NemotronVoicechatInferenceWrapper:
    """
    Inference wrapper for NemotronVoiceChat models.
    Uses a sliding window buffer and processes audio frame by frame.
    """

    def __init__(self, model_cfg: DictConfig):
        """
        Initialize the model for realtime streaming inference.

        Args:
            model_cfg (DictConfig): Configuration describing the model paths and inference parameters.
        """
        if model_cfg is None:
            raise ValueError("model_cfg must be provided")
        if not isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.create(model_cfg)

        # Precision and determinism are process globals scoped by the caller's
        # `inference_precision_from_cfg`. Validated again here so constructing the
        # wrapper directly still rejects an impossible combination.
        self.llm_engine_type, self.tts_engine_type = resolve_engine_types(model_cfg)
        self._deterministic = bool(model_cfg.get("deterministic", False))
        reject_unsupported_determinism(self.llm_engine_type, self.tts_engine_type, self._deterministic)
        reject_unimplemented_vllm(self.llm_engine_type, self.tts_engine_type)
        if not precision_matches_cfg(model_cfg):
            # Direct construction warns: only S2SPipelineBuilder requires the
            # precision scope. These torch globals are not applied here.
            logging.warning(
                "This process is not configured the way model_cfg asks: allow_tf32, "
                "matmul_precision and deterministic are torch process globals and are NOT in "
                "effect for this model, so sampling and numerics may differ from a configured "
                "run. Wrap construction in `inference_precision_from_cfg(model_cfg)`, or use "
                "S2SPipelineBuilder.build_pipeline, which requires it."
            )

        self.model_cfg = model_cfg

        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("`model_cfg.model_path` must be provided.")

        self.decode_audio = bool(model_cfg.get("decode_audio", True))

        if model_cfg.get("speaker_reference"):
            raise ValueError(
                "s2s.speaker_reference is not supported. Checkpoints only generate with "
                "speakers registered at export (register_speaker_dict in "
                "examples/speechlm2/nemotron_voicechat_to_hf.py). Set s2s.speaker_name "
                "to a registered speaker."
            )
        self.speaker_name = model_cfg.get("speaker_name", None)
        if self.decode_audio and not self.speaker_name:
            raise ValueError(
                "`model_cfg.speaker_name` must be provided when decode_audio is enabled. "
                "It must match a speaker registered in the checkpoint."
            )

        self.dtype = str_to_dtype(model_cfg.get("compute_dtype", "bfloat16"))

        device = model_cfg.get("device")
        device_id = model_cfg.get("device_id")
        if device is None:
            self.device = DEFAULT_DEVICE
        else:
            device_str = str(device)
            if device_id is not None and device_str.startswith("cuda") and ":" not in device_str:
                device_str = f"{device_str}:{device_id}"
            self.device = torch.device(device_str)

        logging.info("=" * 70)
        logging.info("INITIALIZING REALTIME STREAMING INFERENCE")
        logging.info("=" * 70)
        logging.info(f"Frame size: {FRAME_SIZE_SEC}s ({FRAME_SIZE_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
        logging.info(f"Device: {self.device}")
        logging.info(f"Compute dtype: {self.dtype}")
        logging.info(f"Decode audio: {self.decode_audio}")
        logging.info(f"Engine types: LLM={self.llm_engine_type}, TTS={self.tts_engine_type}")
        logging.info(
            f"Sampling - top_p: {model_cfg.get('top_p', 0.5)}, repetition_penalty: {model_cfg.get('repetition_penalty', 1.1)}, temperature: {model_cfg.get('temperature', 0.3)}"
        )
        logging.info(f"Precision (configured): deterministic={self._deterministic}")
        logging.info(
            f"Precision (effective): float32_matmul_precision={torch.get_float32_matmul_precision()}, cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        )
        logging.info("=" * 70)

        # Profiling: when True, a TimingSummary (extending NamedTimer with
        # sync_cuda=True) is attached to each decode state, recording
        # per-stage wall-clock times.  Disabled by default to avoid
        # unnecessary GPU stalls in production.
        self._profile_timing = bool(model_cfg.get("profile_timing", False))

        # Cached TTS helpers populated during initialization/warmup
        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None

        self.model = None
        # Selected per-component backends (DuplexLLM / DuplexTTS).
        self.llm_backend = None
        self.tts_backend = None
        # Codec decode is always PyTorch, for every LLM/TTS engine pair.
        self.codec: AudioCodec | None = None
        # Native objects only; None when that component runs on vLLM.
        self.model_llm_interface = None
        self.model_eartts_interface = None
        self.tokenizer = None
        self.special_token_ids: set[int] = set()

        self._output_capabilities = AuxiliaryOutputCapabilities(
            has_asr_head=False,
            has_function_head=False,
        )
        self.request_id = "streaming_request_0"  # For vLLM streaming

        # vLLM-Omni runtime + speaker latent (selected components only).
        self.vllm_omni_config = model_cfg.get("vllm_omni_config", None)
        self.omni_runtime = None
        self.omni_wrapper_dir: str | None = None
        self.omni_speaker_latent: torch.Tensor | None = None
        self.omni_guidance_enabled = True

        # Sampling parameters (defaults match s2s_streaming.yaml)
        self.top_p = float(model_cfg.get("top_p", 0.5))
        self.repetition_penalty = float(model_cfg.get("repetition_penalty", 1.1))
        self.temperature = float(model_cfg.get("temperature", 0.3))

        # Native LLM KV cache (vLLM engines manage their own cache and ignore this).
        self.use_llm_cache = bool(model_cfg.get("use_llm_cache", False))

        # Perception cache configuration (defaults match s2s_streaming.yaml)
        self.use_perception_cache = bool(model_cfg.get("use_perception_cache", True))
        use_perception_cudagraph = bool(model_cfg.get("use_perception_cudagraph", True))
        if use_perception_cudagraph and not self.use_perception_cache:
            raise ValueError(
                "use_perception_cudagraph requires use_perception_cache to be enabled. "
                "Please also set use_perception_cache=True."
            )
        self.perception_cache_mgr: PerceptionCacheManager | None = None
        self._use_perception_cudagraph = use_perception_cudagraph

        self._initialize_model()

        logging.info("NemotronVoicechatInferenceWrapper initialized successfully.")

    # ``llm_engine_type`` and ``tts_engine_type`` are the only stored selection;
    # everything else derives from them so the two cannot drift apart.
    # Perception, codec and tokenization stay on PyTorch in every combination.

    @property
    def use_vllm_llm(self) -> bool:
        return self.llm_engine_type == VLLM_OMNI

    @property
    def use_vllm_tts(self) -> bool:
        return self.tts_engine_type == VLLM_OMNI

    @property
    def use_vllm_omni(self) -> bool:
        return self.use_vllm_llm or self.use_vllm_tts

    @property
    def output_capabilities(self) -> AuxiliaryOutputCapabilities:
        """Optional checkpoint heads this run can produce."""
        return self._output_capabilities

    def _initialize_model(self):
        """Initialize the NemotronVoiceChat model from an HF checkpoint."""
        logging.info("Initializing model structure...")
        start_model_init = time.time()

        # Tell from_pretrained to skip loading checkpoint weights for
        # submodules that vLLM will replace — avoids wasted I/O and memory.
        # The streaming pipeline also does not instantiate the auxiliary RNN-T
        # decoder some checkpoints bundle; its weights are independent of the
        # duplex text/audio path.
        skip_prefixes = native_weight_skip_prefixes(self.llm_engine_type, self.tts_engine_type)

        self.model = NemotronVoiceChat.from_pretrained(
            self.model_path,
            skip_prefixes=skip_prefixes,
        )
        logging.info(f"NemotronVoiceChat initialized in {time.time() - start_model_init:.1f}s")

        # Remove skipped submodules (still on meta device / uninitialized)
        if self.use_vllm_llm:
            del self.model.stt_model.llm
            self.model.stt_model.llm = None
        if self.use_vllm_tts:
            del self.model.tts_model.tts_model

        self.model.to(self.device)
        self.model.safe_cast_to(self.dtype)
        self.model.eval()

        self.tokenizer = self.model.stt_model.tokenizer

        # Bridge the config keys that shared model code reads off its own cfg.
        # See config_overrides for the full set of keys and which
        # backends honour each one.
        effective_overrides = apply_model_cfg_overrides(
            self.model,
            self.model_cfg,
            llm_engine_type=self.llm_engine_type,
            tts_engine_type=self.tts_engine_type,
        )
        if self.model.stt_model.cfg.get("force_turn_taking", False) and not self.model.stt_model.predict_user_text:
            logging.warning(
                "Disabling force_turn_taking because this checkpoint has no ASR head. "
                "The model's learned duplex turn-taking remains active."
            )
            OmegaConf.update(self.model.stt_model.cfg, "force_turn_taking", False)
            effective_overrides["force_turn_taking"] = False
        logging.info(f"Effective model config overrides: {effective_overrides}")

        stt = self.model.stt_model
        self._output_capabilities = derive_auxiliary_output_capabilities(stt)
        logging.info(f"Auxiliary output capabilities: {self._output_capabilities.to_dict()}")
        self.special_token_ids = get_special_token_ids(
            stt.tokenizer,
            stt.text_pad_id,
            model_cfg=stt.cfg,
        )
        if self.use_vllm_omni:
            self._initialize_vllm_omni_backend()

        # One implementation per component, chosen here and nowhere else. The
        # PyTorch objects stay reachable as model_*_interface because stream
        # setup needs cache creation, prompt prefill and abort, which are not
        # on the shared contracts.
        if self.use_vllm_llm:
            self.llm_backend = VllmLLM()
        else:
            self.model_llm_interface = PyTorchLLM(
                model=self.model,
                special_token_ids=self.special_token_ids,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                use_llm_cache=self.use_llm_cache,
            )
            self.llm_backend = self.model_llm_interface
        logging.info(f"LLM backend: {type(self.llm_backend).__name__}")

        if self.use_vllm_tts:
            self.tts_backend = VllmEarTTS(device=self.device)
        else:
            self.model_eartts_interface = PyTorchEarTTS(tts_model=self.model.tts_model)
            self.tts_backend = self.model_eartts_interface

            # PyTorch TTS-only speedups are delegated to its backend.
            if bool(self.model_cfg.get("use_tts_torch_compile", False)):
                self.model_eartts_interface.compile()
            self.model_eartts_interface.setup_subword_cache(self.model_cfg)
        logging.info(f"TTS backend: {type(self.tts_backend).__name__}")

        # Codec decode is always PyTorch, regardless of which TTS engine
        # produced the codes.
        if hasattr(self.model, "tts_model"):
            self.target_fps = self.model.tts_model.target_fps
            self.target_sample_rate = self.model.tts_model.target_sample_rate
            self.codec = AudioCodec(self.model.tts_model, self.device)
            self.codec.log_configuration()
            if self.decode_audio and not self.use_vllm_tts:
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

    # ------------------------------------------------------------------
    # vLLM-Omni backend
    # ------------------------------------------------------------------

    def _initialize_vllm_omni_backend(self):
        """Start the AsyncOmni runtime for selected vLLM components.

        Not implemented in this PR. The parent commit on
        ``duplex-vllm-omni-on-main`` has the wrapper-checkpoint conversion,
        ``OmniRuntime``, and session wiring. Kept as the construction hook so
        the native ``_initialize_model`` path already matches the combined form.
        """
        reject_unimplemented_vllm(self.llm_engine_type, self.tts_engine_type)

    def start_vllm_omni_session(
        self,
        state: StreamingDecodeState,
        system_prompt: str | None,
        *,
        request_id: str,
        sampling_params: dict[str, float] | None = None,
    ) -> None:
        """Attach a per-stream vLLM session (not implemented in this PR)."""
        del state, system_prompt, request_id, sampling_params
        reject_unimplemented_vllm(self.llm_engine_type, self.tts_engine_type)

    # ------------------------------------------------------------------
    # Per-stream lifecycle
    # ------------------------------------------------------------------

    def begin_stream(
        self,
        state: StreamingDecodeState,
        system_prompt: str | None,
        *,
        request_id: str,
        sampling_params: dict[str, float] | None = None,
    ) -> None:
        """Prepare per-stream backend state before the first audio frame.

        The caller does not need to know which components run on vLLM. This
        opens a vLLM session for those that do and prefills the system prompt
        natively when the LLM is native. A vLLM Nemotron consumes the prompt
        inside its own long-lived request, so there is nothing to prefill.
        """
        if self.use_vllm_omni:
            self.start_vllm_omni_session(
                state,
                system_prompt or "",
                request_id=request_id,
                sampling_params=sampling_params,
            )
            logging.info(f"vllm_omni: started streaming session (request_id={request_id!r}).")

        if self.use_vllm_llm or not system_prompt:
            return
        self._prefill_system_prompt_native(state, system_prompt)

    def end_stream(self, state: StreamingDecodeState | None, *, request_id: str) -> None:
        """Release per-stream backend state. Idempotent.

        Must run before the decode state is discarded: a vLLM session's
        consumer task hangs off the state and would otherwise leak until
        process exit, with asyncio reporting a pending destroyed task.
        """
        session = state.omni_session if state is not None else None
        if session is not None:
            state.omni_session = None
            self._close_session(session, request_id)

        # The session above covers whichever components run on vLLM; only the
        # native backends still hold a request to abort.
        if not self.use_vllm_llm:
            self.model_llm_interface.abort_request(request_id)
        if not self.use_vllm_tts:
            self.model_eartts_interface.abort_request(request_id)

    @staticmethod
    def _close_session(session, request_id: str) -> None:
        """Close a vLLM session, falling back to a hard abort.

        The only teardown step that is allowed to fail: ``finish`` waits for the
        engine's consumer task to drain, which can time out or raise if the
        engine is already unhealthy. ``abort`` drops the request without
        waiting, so it is the correct second attempt rather than a blanket
        except. The native aborts above are local bookkeeping and are left to
        raise.
        """
        try:
            session.finish()
            return
        except Exception as exc:
            logging.warning(f"vllm_omni session.finish() failed for request {request_id}: {exc}; aborting instead.")
        try:
            session.abort()
        except Exception as exc:
            logging.warning(f"vllm_omni session.abort() also failed for request {request_id}: {exc}")

    def _prefill_system_prompt_native(self, state: StreamingDecodeState, system_prompt: str) -> None:
        """Put the system prompt into the native LLM's state for this stream.

        Either warms the KV cache or seeds ``input_embeds_history``, depending
        on whether this stream was given a cache.
        """
        logging.info("Prefilling system prompt...")
        start = time.time()
        prompt_embedded, prompt_len = self._prepare_system_prompt_embeddings(system_prompt)
        logging.debug(f"Time taken to get prompt embeddings: {time.time() - start:.3f}s")

        if prompt_embedded is None:
            logging.warning("System prompt embedding returned None, skipping prefill")
            return

        if state.llm_cache is not None:
            with torch.no_grad():
                cache_position = torch.arange(prompt_len, device=self.device)
                ans = self.model_llm_interface.prefill_prompt(
                    prompt_embedded,
                    cache=state.llm_cache,
                    cache_position=cache_position,
                )
                state.llm_cache = ans.get("cache", state.llm_cache)
            state.llm_cache_position_offset = prompt_len
            logging.info(f"System prompt processed, cache updated ({prompt_len} tokens, offset={prompt_len})")
        else:
            for t in range(prompt_len):
                state.input_embeds_history.append(prompt_embedded[:, t : t + 1, :])
            logging.info(f"Added {prompt_len} prompt embeddings to input_embeds_history")

    def shutdown(self) -> None:
        """Tear down the AsyncOmni runtime. Idempotent; no-op for native engines.

        Called by :meth:`StreamingS2SPipeline.shutdown`, so the engine
        subprocesses and the runtime's daemon thread go away at a known point
        rather than at garbage collection or process exit.
        """
        if self.omni_runtime is None:
            return
        runtime, self.omni_runtime = self.omni_runtime, None
        try:
            runtime.shutdown()
        except Exception as exc:
            # Callers invoke this from ``finally`` / server finalize, where
            # raising would replace whatever error is already unwinding.
            logging.warning(f"OmniRuntime.shutdown raised: {exc!r}")

    def _prepare_system_prompt_embeddings(
        self,
        system_prompt: str,
    ) -> tuple[torch.Tensor | None, int]:
        if not system_prompt or not system_prompt.strip():
            return None, 0

        prompt_token_ids = self._build_prompt_token_ids(system_prompt)
        prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_embedded = self.model.stt_model.embed_tokens(prompt_tokens).to(dtype=self.dtype)
        prompt_len = prompt_tokens.shape[1]

        stt = self.model.stt_model
        pad_id = stt.text_pad_id
        pad_token = torch.full((1,), fill_value=pad_id, device=self.device, dtype=torch.long)
        pad_emb = stt.embed_tokens(pad_token).to(dtype=self.dtype)
        bos_emb = stt._get_bos_embedding().to(dtype=self.dtype)

        if prompt_len > 1:
            prompt_embedded[:, 1:, :] += pad_emb
            if stt.predict_user_text:
                pad_asr_emb = stt.embed_asr_tokens(pad_token).to(dtype=self.dtype)
                prompt_embedded[:, 1:, :] += pad_asr_emb

        prompt_embedded[:, 0, :] += bos_emb.squeeze(0)
        if stt.predict_user_text:
            asr_bos_emb = stt._get_asr_bos_embedding().to(dtype=self.dtype)
            prompt_embedded[:, 0, :] += asr_bos_emb.squeeze(0)
        if stt.use_function_head:
            # Match the channel order in DuplexSTTModel.build_input_embedding.
            prompt_embedded += pad_emb.expand(1, prompt_len, -1) * stt.cfg.get("duplex_function_channel_weight", 1.0)

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
        if hasattr(cache, "__dict__"):
            return copy.deepcopy(cache)
        return cache

    def _build_prompt_token_ids(self, system_prompt: str | None) -> list[int]:
        if not system_prompt or not system_prompt.strip():
            return []
        return [self.tokenizer.bos_id] + self.tokenizer.text_to_ids(system_prompt) + [self.tokenizer.eos_id]

    def _init_token_buffers(self, max_len: int):
        stt_model = self.model.stt_model
        gen_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_asr_text = None
        if stt_model.predict_user_text:
            gen_asr_text = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        gen_function = None
        if stt_model.use_function_head:
            gen_function = torch.full((1, max_len), stt_model.text_pad_id, device=self.device, dtype=torch.long)
        return gen_text, gen_asr_text, gen_function

    def _prepare_tts_initial_state(self):
        if not self.decode_audio:
            return
        if not hasattr(self.model, "tts_model"):
            return

        logging.info("Preparing TTS warmup state...")

        if self.speaker_name not in self.model.tts_model.audio_prompt_latents:
            registered = list(self.model.tts_model.audio_prompt_latents.keys())
            raise ValueError(
                f"Unknown speaker_name {self.speaker_name!r}. Registered speakers: "
                f"{registered or '(none)'}. Register speakers at export with "
                "register_speaker_dict, then pass s2s.speaker_name."
            )
        logging.info(f"Using registered speaker name: {self.speaker_name}")

        self.model.tts_model.set_init_inputs(
            speaker_audio=None,
            speaker_audio_lens=None,
            speaker_name=self.speaker_name,
        )
        init_inputs = self.model.tts_model.get_init_inputs(B=1)

        self.generation_config = self.model.tts_model._get_generation_config(guidance_enabled=True)
        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": True})

        with torch.no_grad():
            outputs = self.model_eartts_interface.prefill_prompt(
                init_inputs,
                prompt_token_ids=None,
                request_id="tts_warmup",
            )
            self.model_eartts_interface.abort_request("tts_warmup")

            code = init_inputs["code"][:, -1:]

        self.first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        self.first_tts_code_input = code.detach().clone()
        self.first_tts_past_key_values_input = self._clone_cache(outputs.past_key_values)
        # The backend needs the frame-0 context token and generation config to
        # run its per-frame step; the wrapper keeps the initial cache/codes
        # because those are seeded per stream in create_decode_state.
        self.model_eartts_interface.set_warmup_state(
            self.first_context_subword_id,
            self.generation_config,
        )

        logging.info("TTS warmup state prepared")

    def create_decode_state(self, max_len: int) -> StreamingDecodeState:
        gen_text, gen_asr_text, gen_function = self._init_token_buffers(max_len)

        llm_cache = None if self.use_vllm_llm else self.model_llm_interface.create_cache()
        # One codec for every backend pairing, so no branch on the TTS engine.
        subword_mask, tts_codec_cache = self.codec.create_state(max_len) if self.decode_audio else (None, None)
        perception_cache = None
        if self.use_perception_cache and self.perception_cache_mgr is not None:
            perception_cache = self.perception_cache_mgr.get_initial_state(batch_size=1)

        tts_past_key_values = None
        tts_code = None
        if self.decode_audio and not self.use_vllm_tts and self.first_tts_code_input is not None:
            tts_past_key_values = self._clone_cache(self.first_tts_past_key_values_input)
            tts_code = self.first_tts_code_input.detach().clone()

        return StreamingDecodeState(
            frame_idx=0,
            gen_text=gen_text,
            gen_asr_text=gen_asr_text,
            gen_function=gen_function,
            input_embeds_history=[],
            llm_cache=llm_cache,
            tts_past_key_values=tts_past_key_values,
            tts_code=tts_code,
            subword_mask=subword_mask,
            perception_cache=perception_cache,
            tts_codec_cache=tts_codec_cache,
            llm_cache_position_offset=0,
            timing=TimingSummary() if self._profile_timing else NullTimingSummary(),
            omni_session=None,
        )

    def infer_one_step(
        self,
        audio_input: torch.Tensor,
        num_frames_per_chunk: int,
        state: StreamingDecodeState,
        *,
        request_id: str | None = None,
        has_prompt: bool = False,
        return_debug: bool = False,
        sampling_params: dict[str, float] | None = None,
    ) -> InferenceStepResult:
        """Run one streaming inference step: perception -> LLM -> TTS -> audio decode.

        All mutable decode state (caches, gen_text, gen_asr_text, code, etc.) is
        updated **in-place** on *state*.  The returned :class:`InferenceStepResult`
        carries only per-step outputs needed by the pipeline.

        Args:
            audio_input (torch.Tensor): Raw audio tensor for this chunk, shape ``(1, samples)``.
            num_frames_per_chunk (int): Number of 80 ms frames in this chunk.
            state (StreamingDecodeState): Mutable decode state (KV caches, token workspaces, etc.).
            request_id (str | None): Unique ID for this stream (used by vLLM engines).
            has_prompt (bool): Whether the LLM state already contains a prefilled
                system prompt. Affects the first-frame embedding (PAD vs BOS).
            return_debug (bool): If True, attach per-step debug info to the result.
            sampling_params (dict[str, float] | None): Optional per-stream sampling overrides
                (``top_p``, ``temperature``, ``repetition_penalty``).
                Keys that are absent fall back to the pipeline-level defaults.
        """
        effective_request_id = request_id or self.request_id
        frame_idx = state.frame_idx

        state.timing.start("total_step")
        has_llm_cache = state.llm_cache is not None
        B = state.gen_text.shape[0]
        if B != 1 and self.use_vllm_omni:
            # The session API is per-stream and its steps take scalars.
            raise ValueError(f"vllm_omni components only support batch size 1 (got gen_text batch={B}).")

        def per_step_tokens(channel: torch.Tensor | None) -> torch.Tensor | None:
            """Pad-filled buffer for one channel, or None if the channel is absent.

            Pad rather than empty: a frame the backend reports no token for
            then decodes to nothing instead of to uninitialized memory.
            """
            if channel is None:
                return None
            return torch.full(
                (B, num_frames_per_chunk),
                self.model.stt_model.text_pad_id,
                dtype=state.gen_text.dtype,
                device=state.gen_text.device,
            )

        predicted_tokens = per_step_tokens(state.gen_text)
        asr_predicted_tokens = per_step_tokens(state.gen_asr_text)
        function_predicted_tokens = per_step_tokens(state.gen_function)

        debug_logger = IntermediateResultLogger() if return_debug else NullIntermediateResultLogger()

        # --- Stage 1: Perception ---
        state.timing.start("perception")
        source_encoded, state.perception_cache = self._run_perception(
            audio_input,
            frame_idx,
            num_frames_per_chunk,
            state.perception_cache,
        )
        state.timing.stop("perception")
        base_frame_index = self._base_frame_index(source_encoded, state, num_frames_per_chunk)

        # --- Stage 2: Per-frame generation loop ---
        new_codes_for_decode = []
        for frame_offset in range(num_frames_per_chunk):
            current_frame_idx = frame_idx + frame_offset
            current_frame_index = min(base_frame_index + frame_offset, source_encoded.shape[1] - 1)
            debug_logger.log_selected_frame_index(current_frame_index)
            frame_embedding = source_encoded[:, current_frame_index : current_frame_index + 1, :]

            ans = self._run_llm_step(
                frame_embedding,
                state,
                frame_offset=frame_offset,
                current_frame_idx=current_frame_idx,
                has_prompt=has_prompt,
                return_debug=return_debug,
                sampling_params=sampling_params,
                debug_logger=debug_logger,
            )

            if ans.text_logits is not None:
                debug_logger.log_text_logits(ans.text_logits[:, -1])
            if ans.asr_logits is not None:
                debug_logger.log_asr_logits(ans.asr_logits[:, -1])

            state.gen_text[:, current_frame_idx] = ans.predicted_token
            if state.gen_asr_text is not None:
                asr_token = ans.asr_predicted_token
                if asr_token is None and self.output_capabilities.has_asr_head:
                    raise RuntimeError("Checkpoint has an ASR head but the LLM backend returned no ASR token")
                if asr_token is not None:
                    state.gen_asr_text[:, current_frame_idx] = asr_token
                    asr_predicted_tokens[:, frame_offset] = asr_token
                    self.model.stt_model.streaming_inference._maybe_apply_forced_turn_taking(
                        current_frame_idx, state.gen_text, state.gen_asr_text
                    )
            if state.gen_function is not None:
                function_token = ans.function_predicted_token
                if function_token is None and self.output_capabilities.has_function_head:
                    raise RuntimeError("Checkpoint has a function head but the LLM backend returned no function token")
                if function_token is not None:
                    state.gen_function[:, current_frame_idx] = function_token
                    function_predicted_tokens[:, frame_offset] = function_token
            # Read back rather than reusing ans: forced turn-taking above may
            # have rewritten this frame's text token in place.
            predicted_tokens[:, frame_offset] = state.gen_text[:, current_frame_idx]

            if self.decode_audio:
                new_code = self._run_tts_step(
                    state,
                    current_frame_idx,
                    effective_request_id,
                )
                new_codes_for_decode.append(new_code)

        # --- Stage 3: Audio decode ---
        # No-op when self.decode_audio is False: _decode_audio returns None immediately.
        decoded_audio_new = self._decode_audio(new_codes_for_decode, state, frame_idx, num_frames_per_chunk)

        # --- Stage 4: Token -> string conversion ---
        predicted_text_strs = self._tokens_to_strings(predicted_tokens)
        asr_predicted_text_strs = (
            self._tokens_to_strings(asr_predicted_tokens) if asr_predicted_tokens is not None else None
        )
        predicted_function_strs = (
            self._tokens_to_strings(function_predicted_tokens) if function_predicted_tokens is not None else None
        )

        logging.debug(f"frame {frame_idx}: USER asr: {asr_predicted_text_strs}")
        logging.debug(f"frame {frame_idx}: FUNCTION: {predicted_function_strs}")
        logging.debug(f"frame {frame_idx}: AGENT txt: {predicted_text_strs}")

        # --- Update remaining state fields ---
        # `input_embeds_history` is extended by the native no-cache backend as
        # it goes; see PyTorchLLM.step.
        if has_llm_cache:
            state.llm_cache_position_offset += num_frames_per_chunk

        state.timing.stop("total_step")

        debug = debug_logger.build_debug_dict(source_encoded, state.gen_text, state.gen_asr_text)

        return InferenceStepResult(
            predicted_text_tokens=predicted_tokens,
            asr_predicted_text_tokens=asr_predicted_tokens,
            predicted_text_strs=predicted_text_strs,
            asr_predicted_text_strs=asr_predicted_text_strs,
            predicted_function_tokens=function_predicted_tokens,
            predicted_function_strs=predicted_function_strs,
            decoded_audio=decoded_audio_new,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # infer_one_step sub-stages
    # ------------------------------------------------------------------

    def _run_llm_step(
        self,
        frame_embedding: torch.Tensor,
        state: StreamingDecodeState,
        *,
        frame_offset: int,
        current_frame_idx: int,
        has_prompt: bool,
        return_debug: bool,
        sampling_params: dict[str, float] | None,
        debug_logger,
    ) -> LlmStepResult:
        """Time one :meth:`DuplexLLM.step` on the selected LLM backend.

        Both backends fill the same :class:`LlmStepResult`, so the caller does
        not need to know which one ran; the optional fields are None when the
        checkpoint has no such head or the backend cannot expose logits.
        """
        state.timing.start("stt_model")
        try:
            return self.llm_backend.step(
                frame_embedding,
                state,
                frame_offset=frame_offset,
                current_frame_idx=current_frame_idx,
                has_prompt=has_prompt,
                return_debug=return_debug,
                sampling_params=sampling_params,
                debug_logger=debug_logger,
            )
        finally:
            state.timing.stop("stt_model")

    def _run_tts_step(
        self,
        state: StreamingDecodeState,
        current_frame_idx: int,
        request_id: str,
    ) -> torch.Tensor:
        """Time one :meth:`DuplexTTS.step` on the selected TTS backend.

        Returns this frame's codes as ``(B, T, num_quantizers)`` for the shared
        native codec to decode. Both backends read the committed text token from
        ``state.gen_text``, so a forced-turn-taking rewrite reaches TTS without
        the caller having to pass it.

        ``inference_force_speech_silence_on_eos`` belongs to the backends, not
        here: each applies it internally to the acoustic input of the step whose
        text token is EOS -- natively in ``DuplexEARTTS.infer_codes_one_step``,
        unconditionally in the vLLM preprocess.
        """
        state.timing.start("tts_model")
        try:
            return self.tts_backend.step(state, current_frame_idx, request_id)
        finally:
            state.timing.stop("tts_model")

    def _decode_audio(
        self,
        new_codes_for_decode: list[torch.Tensor],
        state: StreamingDecodeState,
        frame_idx: int,
        num_frames_per_chunk: int,
    ) -> torch.Tensor | None:
        """Decode accumulated TTS codes into a waveform.

        Returns the decoded audio tensor or *None* when ``decode_audio``
        is disabled or no codes were produced.
        """
        if not self.decode_audio or not new_codes_for_decode:
            return None

        logging.debug(f"Decoding audio for {frame_idx}-th frame  ({num_frames_per_chunk=})")

        state.timing.start("audio_codec")
        try:
            return self.codec.decode(new_codes_for_decode, state.tts_codec_cache)
        finally:
            state.timing.stop("audio_codec")

    def _base_frame_index(
        self,
        source_encoded: torch.Tensor,
        state: StreamingDecodeState,
        num_frames_per_chunk: int,
    ) -> int:
        """Index of the first encoded frame belonging to this chunk."""
        if (
            self.use_perception_cache
            and state.perception_cache is not None
            and state.perception_cache.is_initialized()
        ):
            # With cache: we get exactly num_frames_per_chunk output frames
            return 0
        # Without cache: use the second-to-last encoded frame as the
        # "newest" because the model expects 10ms / 80ms / 80ms ... framing
        # but we always feed 80ms chunks, so the final frame contains
        # silence padding.
        newest = source_encoded.shape[1] - 2
        return max(newest - (num_frames_per_chunk - 1), 0)

    def _run_perception(
        self,
        audio_input: torch.Tensor,
        frame_idx: int,
        num_frames_per_chunk: int,
        perception_cache: PerceptionCacheState | None,
    ) -> tuple[torch.Tensor, PerceptionCacheState | None]:
        """Run the perception encoder and return (source_encoded, updated_cache)."""
        if self.use_perception_cache and perception_cache is not None and perception_cache.is_initialized():
            source_encoded, perception_cache = self.perception_cache_mgr.step(
                audio_input=audio_input,
                frame_idx=frame_idx,
                num_frames_per_chunk=num_frames_per_chunk,
                perception_cache=perception_cache,
            )
        else:
            buffer_len = torch.tensor([audio_input.shape[1]], dtype=torch.long, device=self.device)
            source_encoded, _, _ = self.model.stt_model.perception(
                input_signal=audio_input,
                input_signal_length=buffer_len,
                return_encoder_emb=True,
            )

        source_encoded = source_encoded.to(self.dtype)
        return source_encoded, perception_cache

    def _tokens_to_strings(self, token_ids: torch.Tensor) -> list[str]:
        """Convert a [B, T] tensor of token IDs to a list of strings.

        Uses ``_decode_tokens_with_specials`` so byte-level BPE is decoded
        properly (e.g. ``âĢĻ`` -> ``'``) via HF ``convert_tokens_to_string``.

        Leading spaces are preserved in the output: in byte-level BPE,
        word-initial tokens carry a space prefix that ``convert_tokens_to_string``
        keeps intact.  So callers can concatenate successive chunk strings to
        recover properly spaced text.  A leading space means "new word"; no
        leading space means the token continues the previous word.  For
        example, three chunks producing ``"Hi"``, ``" how can"``,
        ``" I help"`` concatenate to ``"Hi how can I help"`` (not
        ``"Hihow canI help"``).

        NOTE: multi-byte UTF-8 characters whose BPE tokens span two frames
        will show as replacement chars (U+FFFD) because each frame is decoded
        independently.
        """
        pad_token_str = self.tokenizer.ids_to_tokens([self.model.stt_model.text_pad_id])[0]
        result = []
        for tok_ids_b in token_ids:
            toks = self.tokenizer.ids_to_tokens(tok_ids_b.tolist())
            result.append(
                _decode_tokens_with_specials(
                    toks,
                    self.tokenizer,
                    pad_token_str=pad_token_str,
                    keep_pad=False,
                )
            )
        return result

    @property
    def special_token_strings(self) -> set[str]:
        """Token strings that should be stripped from decoded text for clean output."""
        stt = self.model.stt_model
        return get_special_token_strings(stt.tokenizer, stt.text_pad_id, model_cfg=stt.cfg)
