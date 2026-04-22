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
            "streaming.chunk_size_in_secs": ("S2S_CHUNK_SIZE_IN_SECS", 0.08),
            "streaming.buffer_size_in_secs": ("S2S_BUFFER_SIZE_IN_SECS", 5.6),
            # RNNT EOU / user-BOS turn-taking
            "s2s.rnnt_eou_enabled":       ("S2S_RNNT_EOU_ENABLED", False),
            "s2s.asr_eou":                ("S2S_ASR_EOU", 4),
            "s2s.asr_bou":                ("S2S_ASR_BOU", 4),
            "s2s.user_bos_frames":        ("S2S_USER_BOS_FRAMES", 4),
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
            
            # Extract optional per-stream system prompt (sent on the first request)
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
                if system_prompt is None:
                    system_prompt = self.pipeline.system_prompt
                frame_options = S2SRequestOptions(system_prompt=system_prompt)

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
                generations.append((torch.empty(1, 0), "", ""))
                continue
            
            state = self.pipeline.get_or_create_state(stream_id)
            audio = state.audio_buffer
            
            full_text = state.get_output_text()
            full_asr_text = state.get_output_asr_text()
            
            if stream_id not in self.text_positions:
                self.text_positions[stream_id] = 0
            last_position = self.text_positions[stream_id]
            incremental_text = full_text[last_position:]
            self.text_positions[stream_id] = len(full_text)
            
            if stream_id not in self.asr_text_positions:
                self.asr_text_positions[stream_id] = 0
            last_asr_position = self.asr_text_positions[stream_id]
            incremental_asr_text = full_asr_text[last_asr_position:]
            self.asr_text_positions[stream_id] = len(full_asr_text)
            
            generations.append((audio, incremental_text, incremental_asr_text))
            
            state.cleanup_after_response()
            
            if frame.is_last:
                self.pipeline.delete_state(stream_id)
                if stream_id in self.text_positions:
                    del self.text_positions[stream_id]
                if stream_id in self.asr_text_positions:
                    del self.asr_text_positions[stream_id]
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
        for audio, text, asr_text in generations:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy().astype(np.float32)
                if audio_np.ndim == 1:
                    audio_np = audio_np.reshape(1, -1)
            else:
                audio_np = np.zeros((1, 0), dtype=np.float32)
            
            text_np = np.array([text.encode('utf-8')], dtype=object)
            asr_text_np = np.array([asr_text.encode('utf-8')], dtype=object)
            
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("output_audio", audio_np),
                pb_utils.Tensor("output_text", text_np),
                pb_utils.Tensor("output_asr_text", asr_text_np),
            ]))

        end_time = time.time()
        logging.info(f"TritonPythonModel.execute time: {end_time - start_time:.2f} seconds")
        logging.info(f"execute() breakdown: triton_requests_to_frames={(_t_to_frames_done - _t_to_frames)*1000:.2f}ms, "
                     f"get_generations={(_t_generations_done - _t_generations)*1000:.2f}ms")
            
        return responses
