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
from typing import List, Iterable, Tuple
import os
import numpy as np
import torch

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.streaming.framing.s2s_request_options import S2SRequestOptions

import triton_python_backend_utils as pb_utils

from omegaconf import OmegaConf
from nemo.utils import logging
import time


class TritonPythonModel:
    """Triton Python model for streaming S2S generation.
    
    This model uses the NeMo S2S pipeline to generate speech from speech input.
    Every Python model that is created must have "TritonPythonModel" as the class name.
    """
    
    def _resolve_env_overrides(self, cfg):
        """Resolve ??? placeholders in the config from environment variables.
        
        This allows start_triton.sh to control model paths and settings via
        env vars, while sharing the same s2s_streaming.yaml used by the CLI.
        
        Env var mapping (cfg key -> env var, default):
            s2s.model_path             -> S2S_MODEL_PATH (required)
            s2s.llm_checkpoint_path    -> S2S_LLM_CHECKPOINT_PATH (required)
            s2s.speaker_reference      -> S2S_SPEAKER_REFERENCE (required)
            s2s.engine_type            -> S2S_ENGINE_TYPE (default: native)
            s2s.system_prompt          -> S2S_SYSTEM_PROMPT (default: none)
            s2s.tts_system_prompt      -> S2S_TTS_SYSTEM_PROMPT (default: none)
            s2s.use_codec_cache        -> S2S_USE_CODEC_CACHE (default: true)
            streaming.chunk_size_in_secs -> S2S_CHUNK_SIZE_IN_SECS (default: 0.08)
            streaming.buffer_size_in_secs -> S2S_BUFFER_SIZE_IN_SECS (default: 5.6)
        """
        env_overrides = {
            # Required
            "s2s.model_path":             ("S2S_MODEL_PATH", None),
            "s2s.llm_checkpoint_path":    ("S2S_LLM_CHECKPOINT_PATH", None),
            "s2s.speaker_reference":      ("S2S_SPEAKER_REFERENCE", None),
            # Optional (with defaults)
            "s2s.engine_type":            ("S2S_ENGINE_TYPE", "native"),
            "s2s.system_prompt":          ("S2S_SYSTEM_PROMPT", None),
            "s2s.tts_system_prompt":      ("S2S_TTS_SYSTEM_PROMPT", None),
            "s2s.use_codec_cache":        ("S2S_USE_CODEC_CACHE", True),
            "s2s.fc_async_enabled":         ("S2S_FC_ASYNC_ENABLED", False),
            "s2s.fc_async_two_phase":       ("S2S_FC_ASYNC_TWO_PHASE", False),
            "s2s.fc_async_use_real_audio":  ("S2S_FC_ASYNC_USE_REAL_AUDIO", False),
            "s2s.fc_convert_num_to_text":   ("S2S_FC_CONVERT_NUM_TO_TEXT", True),
            "s2s.enable_builtin_tools":   ("S2S_ENABLE_BUILTIN_TOOLS", False),
            "s2s.force_turn_taking":             ("S2S_FORCE_TURN_TAKING", False),
            "s2s.force_turn_taking_pad_window":  ("S2S_FORCE_TURN_TAKING_PAD_WINDOW", 25),
            "s2s.force_turn_taking_threshold":   ("S2S_FORCE_TURN_TAKING_THRESHOLD", 40),
            "s2s.turn_taking_source":            ("S2S_TURN_TAKING_SOURCE", "rnnt"),
            "s2s.rnnt_eou_frames":               ("S2S_RNNT_EOU_FRAMES", 15),   # blank frames → EOU (15×80ms=1.2s)
            "s2s.rnnt_bou_frames":               ("S2S_RNNT_BOU_FRAMES", 3),    # speech frames → BOU/barge-in (3×80ms=240ms)
            "s2s.rnnt_fc_interrupt_ms":          ("S2S_RNNT_FC_INTERRUPT_MS", 240),  # ms of speech to interrupt FC async
            "s2s.use_separate_rnnt_ckpt":        ("S2S_USE_SEPARATE_RNNT_CKPT", False),  # True→use real .nemo RNNT, False→use combined ckpt fake RNNT
            "s2s.force_turn_taking_pad_window_first_turn": ("S2S_FORCE_TURN_TAKING_PAD_WINDOW_FIRST_TURN", 10),
            "s2s.system_prompt_repeat_n":        ("S2S_SYSTEM_PROMPT_REPEAT_N", 2),
            "s2s.fc_random_ack_enabled":         ("S2S_FC_RANDOM_ACK_ENABLED", False),
            "s2s.tool_reminder_enabled":         ("S2S_TOOL_REMINDER_ENABLED", False),
            "s2s.fc_on_hold_messages_path":      ("S2S_FC_ON_HOLD_MESSAGES_PATH", None),
            "s2s.fc_generic_on_hold_messages_path": ("S2S_FC_GENERIC_ON_HOLD_MESSAGES_PATH", None),
            "s2s.fc_tool_timeout_sec":           ("S2S_FC_TOOL_TIMEOUT_SEC", 15.0),
            "s2s.repetition_penalty":            ("S2S_REPETITION_PENALTY", 1.0),
            "s2s.temperature":                   ("S2S_TEMPERATURE", 0.0),
            "s2s.top_p":                         ("S2S_TOP_P", 1.0),
            "s2s.inference_bos_boost":           ("S2S_INFERENCE_BOS_BOOST", 0.0),
            "streaming.chunk_size_in_secs": ("S2S_CHUNK_SIZE_IN_SECS", 0.08),
            "streaming.buffer_size_in_secs": ("S2S_BUFFER_SIZE_IN_SECS", 5.6),
        }
        for cfg_key, (env_var, default) in env_overrides.items():
            val = os.environ.get(env_var)
            if val is not None:
                # Cast to match the default's type (e.g. "0.08" -> float)
                if default is not None and isinstance(default, bool):
                    val = val.lower() in ("true", "1", "yes")
                elif default is not None and isinstance(default, float):
                    val = float(val)
                elif default is not None and isinstance(default, int):
                    val = int(val)
                OmegaConf.update(cfg, cfg_key, val, force_add=True)
            elif default is not None:
                OmegaConf.update(cfg, cfg_key, default, force_add=True)

    def load_model(self, config_path: str):
        """Load the S2S pipeline from a YAML config file.
        
        Args:
            config_path: Path to a shared YAML config file (s2s_streaming.yaml).
                         Fields marked ??? are resolved from environment variables
                         exported by start_triton.sh.
        """
        cfg = OmegaConf.load(config_path)
        self._resolve_env_overrides(cfg)

        self.pipeline = S2SPipelineBuilder.build_pipeline(cfg)
        self.pipeline.open_session()
        
        # Compute chunk size in samples from the pipeline's config
        self.chunk_size = int(self.pipeline.chunk_size_in_secs * self.pipeline.input_sample_rate)
        
        # Track text positions to return only incremental updates
        self.text_positions = {}  # stream_id -> last_text_length
        self.asr_text_positions = {}  # stream_id -> last_asr_text_length
        self.function_text_positions = {}  # stream_id -> last_function_text_length

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Set up file logging so inference logs are saved to demo_record/.
        # NeMo uses a named logger "nemo_logger" with propagate=False, so we
        # must add the handler directly to that logger rather than to root.
        import logging as _stdlib_logging
        import datetime
        _log_dir = "/home/vtrinh/projects/elena_niva_inference/demo_record"
        os.makedirs(_log_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_path = os.path.join(_log_dir, f"duplex_demo_server_audio_{_ts}_text_outputs.log")
        _file_handler = _stdlib_logging.FileHandler(_log_path)
        _file_handler.setLevel(_stdlib_logging.DEBUG)
        _file_handler.setFormatter(_stdlib_logging.Formatter(
            "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
        ))
        # logging is nemo.utils.logging (a Logger singleton); ._logger is the
        # underlying stdlib Logger instance ("nemo_logger").
        logging._logger.addHandler(_file_handler)
        logging.info(f"Inference logs will be saved to: {_log_path}")

        # Config path: set S2S_TRITON_CONFIG_PATH env var (start_triton.sh does this automatically).
        config_path = os.environ.get("S2S_TRITON_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "S2S_TRITON_CONFIG_PATH environment variable is not set. "
                "Use start_triton.sh or set it to the path of s2s_streaming.yaml."
            )
        logging.info(f"Loading S2S Triton model from config: {config_path}")
        self.load_model(config_path)

        # Warm up the inference engine(s) with a throwaway prefill so the
        # first real client request doesn't pay one-time initialization cost.
        self.pipeline.warmup()
    
    def finalize(self) -> None:
        """Finalize the model."""
        # Close the session, clear state pool, and empty CUDA cache
        self.pipeline.close_session()
        torch.cuda.empty_cache()
    
    def validate_and_convert_audio(self, audio_signal: np.ndarray) -> torch.Tensor:
        """Validate that the audio chunk matches the expected size and convert to tensor."""
        if audio_signal.ndim == 2:
            audio_signal = audio_signal.flatten()

        if len(audio_signal) != self.chunk_size:
            expected_frames = self.pipeline.num_frames_per_chunk
            actual_secs = len(audio_signal) / self.pipeline.input_sample_rate
            raise ValueError(
                f"Audio chunk size mismatch: received {len(audio_signal)} samples ({actual_secs:.3f}s) "
                f"but server expects {self.chunk_size} samples "
                f"({self.pipeline.chunk_size_in_secs}s = {expected_frames} frame(s)). "
                f"Make sure the client's num_frames_per_chunk matches the server's "
                f"chunk_size_in_secs={self.pipeline.chunk_size_in_secs}."
            )

        return torch.tensor(audio_signal, dtype=torch.float32)
    
    def triton_requests_to_frames(self, requests: Iterable) -> List[Frame]:
        """
        Convert Triton inference requests into streaming audio Frames.
        
        Extracts audio data and sequence batching controls (START, END, CORRID)
        from each Triton request and wraps them in Frame dataclasses for the
        streaming S2S pipeline.
        
        Since max_batch_size=0, processes one request at a time.
        
        Returns:
            List of Frame objects (one per request)
        """
        frames = []
        
        for request in requests:
            # Get audio input
            audio_signal = pb_utils.get_input_tensor_by_name(request, "audio_signal").as_numpy()
            
            # Extract sequence batching metadata from Triton control inputs
            # These are automatically populated when client uses sequence_start/end/id
            is_first = False
            is_last = False
            stream_id = 0
            
            try:
                start_tensor = pb_utils.get_input_tensor_by_name(request, "START")
                if start_tensor is not None:
                    is_first = bool(start_tensor.as_numpy()[0])
            except Exception:
                pass
            
            try:
                end_tensor = pb_utils.get_input_tensor_by_name(request, "END")
                if end_tensor is not None:
                    is_last = bool(end_tensor.as_numpy()[0])
            except Exception:
                pass
            
            try:
                corrid_tensor = pb_utils.get_input_tensor_by_name(request, "CORRID")
                if corrid_tensor is not None:
                    stream_id = int(corrid_tensor.as_numpy()[0])
            except Exception:
                pass
            
            # Extract optional per-stream options (sent on the first request)
            frame_options = None
            if is_first:
                system_prompt = None
                try:
                    prompt_tensor = pb_utils.get_input_tensor_by_name(request, "system_prompt")
                    if prompt_tensor is not None:
                        raw = prompt_tensor.as_numpy()[0]
                        system_prompt = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                except Exception:
                    pass
                if not system_prompt:
                    system_prompt = self.pipeline.system_prompt

                fc_random_ack = None
                try:
                    ack_tensor = pb_utils.get_input_tensor_by_name(request, "fc_random_ack_enabled")
                    if ack_tensor is not None:
                        raw = ack_tensor.as_numpy()[0]
                        val = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                        fc_random_ack = val.lower() not in ("", "false", "0", "no")
                except Exception:
                    pass
                # Fall back to global config if not provided per-session
                if fc_random_ack is None:
                    env_val = os.environ.get("S2S_FC_RANDOM_ACK_ENABLED", "")
                    if env_val:
                        fc_random_ack = env_val.lower() not in ("false", "0", "no")

                frame_options = S2SRequestOptions(
                    system_prompt=system_prompt,
                    fc_random_ack_enabled=fc_random_ack,
                )

            # Zero-length audio = prefill-only frame; pass through without validation
            if audio_signal.size == 0:
                samples = torch.empty(0, dtype=torch.float32)
            else:
                samples = self.validate_and_convert_audio(audio_signal)

            frames.append(Frame(
                samples=samples,
                stream_id=stream_id,
                is_first=is_first, 
                is_last=is_last,
                options=frame_options,
            ))
        
        return frames
    
    def get_generations(self, frames: List[Frame]) -> List[Tuple]:
        """
        Generate speech for the requests.
        
        Uses StreamingS2SPipeline.generate_step() which updates internal state,
        then extracts results from per-stream S2SStreamingState objects.

        Zero-length first frames are prefill-only: generate_step handles them
        internally and returns early; this method returns empty results for them.
        
        Returns a list of tuples, where each tuple contains:
        - generated audio tensor
        - generated text string (incremental, only new text since last response)
        - generated ASR text string (incremental, only new ASR text since last response)
        """
        _t_generate_step = time.time()
        self.pipeline.generate_step(frames)
        _t_generate_step_done = time.time()
        
        _t_extract = time.time()
        generations = []
        
        for frame in frames:
            stream_id = frame.stream_id

            # Prefill-only frames don't produce audio/text output
            if frame.is_first and frame.samples.numel() == 0:
                generations.append((torch.empty(1, 0), "", "", ""))
                continue
            
            state = self.pipeline.get_or_create_state(stream_id)
            audio = state.audio_buffer
            
            full_text = state.get_output_text()
            full_asr_text = state.get_output_asr_text()
            full_function_text = state.get_output_function_text()
            
            if stream_id not in self.text_positions:
                self.text_positions[stream_id] = 0
            last_position = self.text_positions[stream_id]
            incremental_text = full_text[last_position:]
            self.text_positions[stream_id] = len(full_text)
            
            # Send full ASR text each frame so RNNT revisions display correctly.
            # The UI replaces the current-turn hypothesis rather than appending deltas.
            incremental_asr_text = full_asr_text

            if stream_id not in self.function_text_positions:
                self.function_text_positions[stream_id] = 0
            last_fc_position = self.function_text_positions[stream_id]
            incremental_function_text = full_function_text[last_fc_position:]
            self.function_text_positions[stream_id] = len(full_function_text)
            
            generations.append((audio, incremental_text, incremental_asr_text, incremental_function_text))
            
            state.cleanup_after_response()
            
            if frame.is_last:
                self.pipeline.delete_state(stream_id)
                if stream_id in self.text_positions:
                    del self.text_positions[stream_id]
                if stream_id in self.asr_text_positions:
                    del self.asr_text_positions[stream_id]
                if stream_id in self.function_text_positions:
                    del self.function_text_positions[stream_id]
        _t_extract_done = time.time()
        
        logging.info(f"get_generations breakdown: generate_step={(_t_generate_step_done - _t_generate_step)*1000:.2f}ms, "
                     f"extract+cleanup={(_t_extract_done - _t_extract)*1000:.2f}ms")
       
        return generations
    
    def execute(self, requests: Iterable) -> List[pb_utils.InferenceResponse]:
        """Execute the model and return the responses.
        
        Zero-length audio with ``sequence_start=True`` and a ``system_prompt``
        is treated as a prefill-only request by the pipeline (no fake audio
        needed).  All other requests are normal audio generation.
        
        Returns:
        - output_audio: float32 array of generated audio samples
        - output_text: UTF-8 encoded string of generated text (agent's response)
        - output_asr_text: UTF-8 encoded string of ASR text (user's transcribed speech)
        """
        start_time = time.time()
        
        _t_to_frames = time.time()
        frames = self.triton_requests_to_frames(requests)
        _t_to_frames_done = time.time()
        
        _t_generations = time.time()
        generations = self.get_generations(frames)
        _t_generations_done = time.time()
        
        responses = []
        for audio, text, asr_text, function_text in generations:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy().astype(np.float32)
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            else:
                audio_np = np.zeros((1, 0), dtype=np.float32)
            
            text_np = np.array([text.encode('utf-8')], dtype=object)
            asr_text_np = np.array([asr_text.encode('utf-8')], dtype=object)
            function_text_np = np.array([function_text.encode('utf-8')], dtype=object)
            
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("output_audio", audio_np),
                pb_utils.Tensor("output_text", text_np),
                pb_utils.Tensor("output_asr_text", asr_text_np),
                pb_utils.Tensor("output_function_text", function_text_np),
            ]))

        end_time = time.time()
        logging.info(f"TritonPythonModel.execute time: {end_time - start_time:.2f} seconds")
        logging.info(f"execute() breakdown: triton_requests_to_frames={(_t_to_frames_done - _t_to_frames)*1000:.2f}ms, "
                     f"get_generations={(_t_generations_done - _t_generations)*1000:.2f}ms")
            
        return responses
