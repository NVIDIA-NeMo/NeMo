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

"""
Model Interface for S2S Inference

This module provides an abstract interface for model inference engines,
allowing seamless swapping between different implementations (e.g., native PyTorch, vLLM)
without modifying the inference code.

Usage Example:
    from nemo.collections.speechlm2.inference.model_wrappers.model_factory import create_model

    # Create interface (automatically wraps existing model)
    model_interface = create_model(
        model=your_model,
        engine_type="native"  # or "vllm"
    )

    # Use the interface exactly as you would use self.model()
    ans = model_interface(input_embeds, cache=cache)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Set
import math
import os
import torch
from transformers import DynamicCache
from dataclasses import dataclass

from nemo.utils import logging

class ModelInterface(ABC):
    """
    Base interface for model inference engines with shared sampling utilities.

    This interface defines the contract that all model implementations must follow,
    ensuring consistent behavior across different engine types. It also provides
    concrete implementations of sampling methods (top-p, repetition penalty) that
    can be shared across all engines.
    """

    def __init__(
        self,
        special_token_ids: Optional[Set[int]] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize base interface with sampling parameters.

        Args:
            special_token_ids: Set of special token IDs (pad, eos, bos) that should bypass sampling.
                               These tokens will use greedy decoding and won't be penalized.
            top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
            repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
            temperature: Temperature for sampling. 1.0 = no change, <1.0 = sharper, >1.0 = flatter.
                        0.0 = greedy (argmax). Default: 1.0
        """
        if not math.isfinite(temperature):
            raise ValueError(f"temperature must be finite, got {temperature}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {temperature}")

        self.special_token_ids = special_token_ids if special_token_ids is not None else set()
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

    def _sample_text_token(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        """
        Sample text tokens with optional top-p sampling and repetition penalty.
        Special tokens (pad, eos, bos) bypass sampling - if they have highest probability, return them directly.

        Args:
            logits: Logits tensor of shape (B, V) for vocabulary V.
            generated_tokens: Previously generated tokens of shape (B, T).
            current_step: Current decoding step (used to slice generated_tokens).

        Returns:
            Sampled token ids of shape (B,).
        """
        B, V = logits.shape
        device = logits.device

        # First check greedy tokens (on original logits)
        greedy_tokens = logits.argmax(dim=-1)  # (B,)

        # If no sampling needed (all disabled), return greedy
        if self.top_p >= 1.0 and self.repetition_penalty == 1.0 and (self.temperature == 1.0 or self.temperature == 0.0):
            return greedy_tokens

        # temperature=0 means greedy
        if self.temperature == 0.0:
            return greedy_tokens

        # For each batch, if greedy is special token, use greedy; otherwise sample
        sampled_tokens = greedy_tokens.clone()

        for b in range(B):
            # If greedy token is a special token, keep it (no sampling)
            if greedy_tokens[b].item() in self.special_token_ids:
                continue

            # Not a special token - apply repetition penalty and sampling
            batch_logits = logits[b].clone()  # (V,)

            # Apply repetition penalty
            if self.repetition_penalty != 1.0 and current_step > 0:
                prev_tokens = generated_tokens[b, :current_step]
                unique_prev = prev_tokens.unique()
                # Exclude special tokens from penalty
                if self.special_token_ids:
                    # Use unique_prev.device to ensure tensors are on the same device
                    # (generated_tokens may be on a different device than logits, e.g., vLLM returns CPU logits)
                    special_tensor = torch.tensor(list(self.special_token_ids), device=unique_prev.device)
                    mask = ~torch.isin(unique_prev, special_tensor)
                    unique_prev = unique_prev[mask]

                for token_id in unique_prev:
                    token_id = token_id.item()
                    if batch_logits[token_id] > 0:
                        batch_logits[token_id] = batch_logits[token_id] / self.repetition_penalty
                    else:
                        batch_logits[token_id] = batch_logits[token_id] * self.repetition_penalty

            # Apply temperature scaling
            if self.temperature != 1.0:
                batch_logits = batch_logits / self.temperature

            # Apply top-p sampling
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(batch_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative prob > top_p, keeping at least one
                sorted_indices_to_remove = cumulative_probs > self.top_p
                # Shift to keep the first token that exceeds threshold
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                # Set to -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                batch_logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = torch.softmax(batch_logits, dim=-1)
            sampled_tokens[b] = torch.multinomial(probs, num_samples=1).item()

        return sampled_tokens

    @abstractmethod
    def __call__(
        self,
        input_embeds: torch.Tensor,
        cache: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform model inference.

        Args:
            input_embeds: Input embeddings tensor of shape [batch, seq_len, hidden_dim]
            cache: Optional cache object (e.g., DynamicCache for transformers)
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing:
                - 'text_logits': Logits for text generation [batch, seq_len, vocab_size]
                - 'cache': Updated cache object (if cache was provided)
                - Additional model-specific outputs
        """
        pass

    @abstractmethod
    def to(self, device_or_dtype: Union[torch.device, torch.dtype]) -> 'ModelInterface':
        """Move model to specified device or convert to specified dtype."""
        pass

    @abstractmethod
    def eval(self) -> 'ModelInterface':
        """Set model to evaluation mode."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device of the model."""
        pass


class VllmLLMModel(ModelInterface):
    """
    vLLM-based model interface using LLMStreamingEngine.


    This wraps the LLMStreamingEngine to provide async streaming inference
    while conforming to the ModelInterface contract. Supports multiple concurrent
    requests sharing a single engine instance.

    model = VllmLLMModel(...)

    async def process_stream(embeds, stream_id):
        # Use the async engine directly
        result = await model._async_inference(embeds, f"stream_{stream_id}", seq_len)
        return result

    # Run multiple streams concurrently in same event loop
    async def main():
        results = await asyncio.gather(
            process_stream(embeds1, 1),
            process_stream(embeds2, 2),
            process_stream(embeds3, 3)
        )

    asyncio.run(main())
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 1024,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        dtype: str = "bfloat16",
        engine_path: Optional[str] = None,
        pretrained_llm: Optional[str] = None,
        special_token_ids: Optional[Set[int]] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        model_type: str = "llm",
        text_pad_id: Optional[int] = None,
        device_id: Optional[int] = None,
        **sampling_kwargs
    ):
        """
        Initialize vLLM model interface with LLMStreamingEngine.

        Args:
            model_path: Path to the vLLM-compatible model checkpoint
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            trust_remote_code: Whether to trust remote code in model
            dtype: Data type for embeddings (e.g., "bfloat16", "float16")
            engine_path: Optional path to pre-converted vLLM model
            pretrained_llm: Optional path to pretrained LLM for conversion
            special_token_ids: Set of special token IDs (for potential post-processing)
            top_p: Top-p sampling (currently vLLM uses greedy decoding)
            repetition_penalty: Repetition penalty (currently not used by vLLM engine)
            text_pad_id: PAD token ID used as fallback when no tokens are generated.
                         If None, attempts to extract from vLLM tokenizer or special_token_ids.
            temperature: Temperature for sampling. Applied in _sample_text_token, not in vLLM engine.
            model_type: Type of model for vLLM engine ("llm", "chatglm", etc.)
            **sampling_kwargs: Additional sampling parameters passed to vLLM engine.
                               By default, vLLM uses greedy decoding (temperature=0)
        """
        # Initialize base class with sampling parameters
        super().__init__(
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        import asyncio
        import threading
        from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import LLMStreamingEngine

        self.model_path = model_path
        self.pretrained_llm = pretrained_llm
        self._dtype = dtype
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Force greedy decoding in vLLM by setting temperature=0 if not specified
        if 'temperature' not in sampling_kwargs:
            sampling_kwargs['temperature'] = 0.0

        if engine_path is None:
            # convert model to vLLM format if needed
            dir_name = os.path.basename(os.path.normpath(model_path))
            engine_path = "/tmp/" + dir_name + f"_vllm_converted_{model_type}"
            if os.path.exists(engine_path):
                logging.info(f"Found existing vLLM converted model at {engine_path}")
            else:
                self._convert_ckpt(
                    save_path=engine_path
                )

        from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import create_engine
        # device_id is passed to LLMStreamingEngine, which sets CUDA_VISIBLE_DEVICES
        # inside initialize() right before AsyncLLM spawns its GPU workers.
        self.engine = create_engine(
            engine_type=model_type,
            model_path=engine_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_id=device_id,
            **sampling_kwargs
        )
        # Track request counter
        self._request_counter = 0

        # Lock to serialize run_until_complete calls across threads (e.g. the main
        # Triton execute thread and the FC-async background thread introduced in
        # streaming_s2s_pipeline.py both call the LLM concurrently without this).
        self._loop_lock = threading.Lock()

        # Get or create event loop for this thread
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                raise RuntimeError("closed")
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # Initialize engine immediately to avoid first-call latency
        logging.info("Initializing vLLM engine (this may take a moment)...")
        self._run_async(self.engine.initialize())

        if self.engine.engine.tokenizer is not None and not self.special_token_ids:
            self.special_token_ids = self._get_special_token_ids_from_vllm_tokenizer(self.engine.engine.tokenizer)

        if text_pad_id is not None:
            self._text_pad_id = text_pad_id
        elif self.engine.engine.tokenizer is not None:
            try:
                tid = self.engine.engine.tokenizer.convert_tokens_to_ids('<SPECIAL_12>')
                self._text_pad_id = tid if isinstance(tid, int) else 0
            except Exception:
                self._text_pad_id = 0
        else:
            self._text_pad_id = 0
        logging.info(f"text_pad_id resolved to: {self._text_pad_id}")

        logging.debug(f"Special token IDs: {self.special_token_ids}")
        logging.info("vLLM engine ready!")

    @staticmethod
    def _get_special_token_ids_from_vllm_tokenizer(tokenizer) -> Set[int]:
        """
        Extract special token IDs from a vLLM tokenizer.
        Looks for: '<s>' (bos), '</s>' (eos), '<SPECIAL_12>' (pad).

        Args:
            tokenizer: A vLLM CachedTokenizer instance.

        Returns:
            Set of special token IDs.
        """
        special_ids = set()
        for token in ('<s>', '</s>', '<SPECIAL_12>'):
            try:
                tid = tokenizer.convert_tokens_to_ids(token)
                if isinstance(tid, int):
                    special_ids.add(tid)
            except Exception:
                pass
        return special_ids

    def _convert_ckpt(self, save_path: str):
        """Convert existing checkpoint to vLLM format and save."""
        from nemo.collections.speechlm2.inference.vllm.scripts.convert_nemotronllm_checkpoint import convert_nemo_to_hf_format

        convert_nemo_to_hf_format(
            checkpoint_path=self.model_path,
            output_dir=save_path,
            pretrained_llm=self.pretrained_llm,
            dtype=self._dtype
        )
        logging.info(f"Converted model saved to {save_path}")

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"vllm_request_{self._request_counter}"

    def __call__(
        self,
        input_embeds: torch.Tensor,
        request_id: Optional[str] = "request_id_1",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform inference using vLLM streaming engine.

        Args:
            inputs:
            cache: Optional cache object (currently not used for streaming)
            generated_tokens: Optional tensor of generated tokens
            current_step: Current decoding step
            request_id: Unique request identifier for this generation
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary containing:
                - predicted_token: Last generated text token
                - asr_predicted_token: Last generated ASR token
                - cache: None (vLLM manages cache internally)
                - is_finished: Whether generation is complete
                - request_id: The request identifier
        """
        # Run async inference
        result = self._run_async(self._async_inference(input_embeds, request_id, **kwargs))
        return result

    async def _async_inference(
        self,
        inputs: Union[torch.Tensor, list[torch.Tensor]],
        request_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async inference using the streaming engine.

        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_dim]
            request_id: Unique request identifier
            seq_len: Number of decoding steps to perform

        Returns:
            Dictionary with text_logits and other outputs
        """
        # Check request status and restart if needed
        from nemo.collections.speechlm2.inference.vllm.streaming_llm_engine import StreamStatus

        if request_id not in self.engine.requests:
            await self.engine.start_generation(request_id=request_id)
        else:
            # Check if request is finished and needs restart
            request_state = self.engine.requests[request_id]
            if request_state.status in (StreamStatus.FINISHED, StreamStatus.ABORTED):
                logging.warning(
                    f"Request {request_id} was {request_state.status.value}. "
                    f"Generated {len(request_state.generated_tokens)} tokens before stopping. "
                    "Cleaning up and restarting..."
                )
                # Try to abort cleanly first
                try:
                    await self.engine.abort_generation(request_id)
                except Exception:
                    pass
                # Start fresh
                await self.engine.start_generation(request_id=request_id)

        # Process embeddings to generate tokens
        return await self._process_inputs_to_outputs(inputs, request_id, **kwargs)

    async def _process_inputs_to_outputs(
        self,
        input_embeds: torch.Tensor,
        request_id: str,
        decode_steps: int = 1,
        prompt_token_ids: Optional[list] = None,
        generated_tokens: Optional[torch.Tensor] = None,
        current_step: int = 0
    ) -> Dict[str, Any]:
        """
        Process embeddings sequentially to generate text and ASR tokens.

        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_dim]
            request_id: Request identifier
            decode_steps: Number of decoding steps to perform; decode steps = 0 means prefill
            prompt_token_ids: Optional list of prompt token IDs for prefill
            generated_tokens: Previously generated tokens [batch, num_generated].
                             Required for repetition_penalty. If None, creates empty tensor.
            current_step: Current decoding step. Used for repetition penalty.
        """

        if decode_steps == 0:
            # prefill only, no token generation
            input_embeds = input_embeds.flatten(0, 1)  # [seq_len, hidden_dim]
            result = await self.engine.generate_next_token([input_embeds],
                                                            prompt_token_ids,
                                                            request_id=request_id)
            return True if result is not None else False

        # Process each embedding in sequence
        text_token_ids = []
        asr_token_ids = []
        result = None
        for i in range(decode_steps):
            # Extract single embedding [1, hidden_dim]
            single_embed = input_embeds[:, i:i+1, :].squeeze(1)  # [batch, hidden_dim]

            # Generate next token
            result = await self.engine.generate_next_token([single_embed], request_id=request_id)
            if result is None:
                # No token generated (finished or error)
                break

            text_token_ids.append(result.token_id)
            if result.custom_outputs and "asr_tokens" in result.custom_outputs:
                asr_token_ids.append(result.custom_outputs["asr_tokens"])
            else:
                asr_token_ids.append(0)

            if result.is_finished:
                break

        assert len(text_token_ids) <= decode_steps, "Generated more tokens than input embeddings"
        # Handle case when no tokens were generated
        if not text_token_ids:
            pad_id = self._text_pad_id
            logging.warning(
                "[VLLMModel] No tokens generated for request %s at step %s, returning PAD",
                request_id, current_step,
            )
            return {
                "predicted_token": pad_id,
                "asr_predicted_token": 0,
                "cache": None,
                "is_finished": True,
                "request_id": request_id,
            }

        is_finished = len(text_token_ids) < decode_steps or (result and result.is_finished)

        text_logits = result.custom_outputs["text_logits"] if result else None

        predicted_token = text_token_ids[-1]
        if self.top_p < 1.0 or self.repetition_penalty != 1.0 or (self.temperature != 1.0 and self.temperature != 0.0):
            # Use provided generated_tokens or create empty tensor
            batch_size = text_logits.shape[0]
            if generated_tokens is None:
                gen_tokens = torch.empty(batch_size, 0, device=text_logits.device, dtype=torch.long)
            else:
                gen_tokens = generated_tokens

            # Apply sampling with top-p and repetition penalty
            predicted_token = self._sample_text_token(
                logits=text_logits,
                generated_tokens=gen_tokens,
                current_step=current_step,
            )

        ans = {
            "predicted_token": predicted_token,
            "asr_predicted_token": asr_token_ids[-1],
            "cache": None,  # vLLM manages cache internally
            "is_finished": is_finished,
            "request_id": request_id
        }
        if result and result.custom_outputs and "function_tokens" in result.custom_outputs:
            ans["function_predicted_token"] = result.custom_outputs["function_tokens"]
        return ans


    def to(self, device_or_dtype: Union[torch.device, torch.dtype]) -> 'VLLMModel':
        """
        Move model to specified device or convert to specified dtype.

        Note: vLLM manages device placement internally, this is for compatibility.
        """
        if isinstance(device_or_dtype, torch.device):
            self._device = device_or_dtype
        elif isinstance(device_or_dtype, torch.dtype):
            # dtype conversion not directly supported, update config
            pass
        return self

    def eval(self) -> 'VLLMModel':
        """Set model to evaluation mode (vLLM is always in eval mode)."""
        return self

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self._device

    def _run_async(self, coro):
        """Run a coroutine on the vLLM event loop, blocking until done.

        Serialized with a lock so that the main Triton thread and the FC-async
        background thread never call run_until_complete() simultaneously on the
        same loop (which would raise 'This event loop is already running').
        """
        with self._loop_lock:
            return self._loop.run_until_complete(coro)

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a specific generation request.

        Args:
            request_id: Request identifier to abort

        Returns:
            bool: True if abort was successful
        """
        return self._run_async(self.engine.abort_generation(request_id))

    def restart_request(self, request_id: str) -> bool:
        """
        Restart a finished or aborted generation request.

        Args:
            request_id: Request identifier to restart

        Returns:
            bool: True if restart was successful
        """
        # First abort if active
        if request_id in self.engine.requests:
            self.abort_request(request_id)

        # Start new generation
        return self._run_async(self.engine.start_generation(request_id=request_id))

    def get_request_status(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of a specific request or all requests.

        Args:
            request_id: Optional request ID. If None, returns all requests.

        Returns:
            Status dictionary
        """
        return self.engine.get_status(request_id)

    def shutdown(self):
        """Shutdown the vLLM engine and cleanup resources."""
        self._run_async(self.engine.shutdown())

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass

@dataclass
class TTSGenerationResult:
    codes: torch.Tensor  # Generated acoustic tokens
    past_key_values: Optional[Any]  # Updated cache (if applicable)

    def __getitem__(self, item: str | int):
        """Allows for accessing attributes by key or index."""
        if isinstance(item, str):
            return getattr(self, item)
        else:
            # Access fields in the order they are defined in the dataclass
            return getattr(self, fields(self)[item].name)


class VllmEARTTSModel(VllmLLMModel):
    """
    vLLM-based model interface specialized for EARTTS models.

    Inherits from VllmLLMModel and sets EARTTS-specific configurations.
    """

    def __init__(self, **kwargs):
        """
        Initialize vLLM EARTTS model interface.

        Args:
            **kwargs: Arguments passed to the VllmLLMModel constructor
        """
        super().__init__(**kwargs)
        logging.info("VllmEARTTSModel initialized with EARTTS-specific settings.")

    def _convert_ckpt(self, save_path: str):
        """Convert EARTTS checkpoint to vLLM format."""
        from nemo.collections.speechlm2.inference.vllm.scripts.convert_eartts_checkpoint import convert
        ckpt_dir = os.path.normpath(self.model_path)
        config_file = os.path.join(ckpt_dir, "config.json")
        model_ckpt = os.path.join(ckpt_dir, "model.safetensors")
        convert(save_path, config_file, model_ckpt)

    def __call__(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        request_id: Optional[str] = None,
        prompt_token_ids: Optional[list] = None,
        **kwargs
    ) -> TTSGenerationResult:
        """
        Perform TTS inference using vLLM streaming engine.

        Supports two calling conventions:
        1. model(inputs_dict, request_id="id")  - pass dict as first positional arg
        2. model(**inputs_dict)  - unpack dict as keyword arguments

        Args:
            inputs: Optional dict of model inputs (if None, uses **kwargs)
            request_id: Optional request identifier
            **kwargs: Model inputs as keyword arguments (used if inputs is None):
                - code: prev_audio_tokens
                - context_hidden_state: context_hidden_state (must be None)
                - subword_ids: current_subword_id
                - subword_mask: current_subword_mask
                - past_key_values: past_key_values
                - use_cache: True
                - guidance_enabled: guidance_enabled
                - generation_config: generation_config
                - ignore_eos_flag_stop: ignore_eos_flag_stop

        Returns:
            TTSGenerationResult containing generated acoustic tokens and cache
        """
        # Handle both calling conventions
        if inputs is not None:
            # Called as model(inputs_dict, request_id="id")
            input_dict = inputs
        else:
            # Called as model(**inputs_dict)
            # Extract request_id from kwargs if present
            if request_id is None:
                request_id = kwargs.pop('request_id', None)
            input_dict = kwargs

        # Use default request_id if still None
        if request_id is None:
            request_id = 'tts_request_id_1'

        # Run async inference
        result = self._run_async(
            self._async_inference(input_dict, request_id, prompt_token_ids=prompt_token_ids)
        )

        return result

    async def _process_inputs_to_outputs(
        self,
        inputs: Dict[str, torch.Tensor],
        request_id: str,
        prompt_token_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Process embeddings sequentially to generate text and ASR tokens.

        Args:
        inputs = {
            "code": prev_audio_tokens,
            "context_hidden_state": context_hidden_state,
            "subword_ids": current_subword_id,
            "subword_mask": current_subword_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "guidance_enabled": guidance_enabled,
            "generation_config": generation_config,
            "ignore_eos_flag_stop": ignore_eos_flag_stop,
            }
        Returns:
            step_acoustic_tokens: Generated acoustic tokens for the current step
            cache: None (vLLM manages cache internally)
        """

        assert inputs["context_hidden_state"] is None, "EARTTS vllm model does not support context_hidden_state input"

        codes = inputs["code"].squeeze(0)  # T x 31
        if codes.shape[0] > 1:
            # in prefill stage, we needto shift acoustic tokens for vllm,
            # replicating the NeMo logic from here:
            # https://github.com/erastorgueva-nv/NeMo/blob/duplex-realtime-inference/nemo/collections/speechlm2/modules/ear_tts_model.py#L1357
            codes = torch.nn.functional.pad(codes[:-1], [0, 0, 1, 0])
        input_tensors = [
            codes,
            inputs["subword_ids"].squeeze(0),
            inputs["subword_mask"].squeeze(0),
        ]

        if "non_prompt_mask" in inputs:
            # Apply edge detection to match native model's BOS placement logic:
            # BOS should only be applied at the FIRST position where non_prompt_mask is True
            non_prompt_mask = inputs["non_prompt_mask"].squeeze(0)  # T
            # Compute edge: positions where mask is True AND previous position is False
            padded_prev = torch.nn.functional.pad(non_prompt_mask[:-1], [1, 0], value=False)
            bos_mask = (non_prompt_mask & (~padded_prev)).to(dtype=getattr(torch, self._dtype))
            input_tensors.append(bos_mask)


        else:
            current_subword_id = input_tensors[1]
            # Use a tiny epsilon instead of exact 0 so the vLLM model's
            # (bos_mask == 0) check is False during decoding.  This prevents
            # use_audio_prompt_frozen_projection from incorrectly applying the
            # speaker-prompt projection to every decoding step.  The epsilon is
            # small enough that bos_mask * bos_emb remains negligible.
            bos_mask = torch.full_like(current_subword_id, 1e-20, dtype=getattr(torch, self._dtype))
            input_tensors.append(bos_mask)

        result = await self.engine.generate_next_token(input_tensors, prompt_token_ids=prompt_token_ids, request_id=request_id)
        acoustic_tokens = result.custom_outputs["acoustic_tokens"]  # T x 31
        step_acoustic_tokens = acoustic_tokens[-1:]  # 1 x 31
        return TTSGenerationResult(
            codes=step_acoustic_tokens.unsqueeze(0).cuda(),  # Add batch dim back: 1 x 1 x 31
            past_key_values=None  # vLLM manages cache internally
        )

class NativeModel(ModelInterface):
    """
    Native PyTorch model interface.

    This wraps the existing DuplexS2SExternalSpeechDecoderModel to conform
    to the ModelInterface contract. Supports top-k, top-p sampling and repetition penalty.
    """

    def __init__(
        self,
        model,
        special_token_ids: Optional[Set[int]] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize with an existing model.

        Args:
            model: The DuplexS2SExternalSpeechDecoderModel instance
            special_token_ids: Set of special token IDs (pad, eos, bos) that should bypass sampling.
                               These tokens will use greedy decoding and won't be penalized.
                               If None, will try to extract from model.tokenizer for tokens:
                               '<s>' (bos), '</s>' (eos), '<SPECIAL_12>' (pad).
                               You can also manually provide: {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
            top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
            repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
                               Recommended value when enabling: 1.2
            temperature: Temperature for sampling. 1.0 = no change, <1.0 = sharper, >1.0 = flatter.
                        0.0 = greedy (argmax). Default: 1.0
        """
        # Default special token IDs: bos=1, eos=2, pad=12
        DEFAULT_SPECIAL_TOKEN_IDS = {1, 2, 12}

        # Try to extract special token IDs from model if not provided
        if special_token_ids is None:
            special_token_ids = self._extract_special_token_ids_from_nemo(model)
        # Fallback to default if extraction failed
        if not special_token_ids:
            special_token_ids = DEFAULT_SPECIAL_TOKEN_IDS
        # Initialize base class with sampling parameters
        super().__init__(
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

        self.model = model

        logging.debug(f"Special token IDs: {self.special_token_ids}")

        # Validate: if sampling is enabled, special_token_ids should be set
        sampling_active = top_p < 1.0 or repetition_penalty != 1.0 or (temperature != 1.0 and temperature != 0.0)
        if sampling_active and not self.special_token_ids:
            import warnings
            warnings.warn(
                "Sampling is enabled but special_token_ids is empty. "
                "Could not auto-extract from model.tokenizer. "
                "Please provide special_token_ids manually to ensure special tokens use greedy decoding. "
                "Otherwise, EOS tokens may be randomly sampled and generation may not stop properly!"
            )

    def __call__(
        self,
        input_embeds: torch.Tensor,
        cache: Optional[Any] = None,
        generated_tokens: Optional[torch.Tensor] = None,
        current_step: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform inference using the native model.

        Args:
            input_embeds: Input embeddings [batch, seq_len, hidden_dim]
            cache: Optional DynamicCache for transformers
            generated_tokens: Previously generated tokens [batch, num_generated].
                             Required for repetition_penalty. If None, creates empty tensor.
            current_step: Current decoding step. Used for repetition penalty.
            **kwargs: Additional arguments passed to the model

        Returns:
            Dictionary with 'predicted_token', 'asr_predicted_token', and 'cache'
        """
        # Call the underlying model
        result = self.model.stt_model(input_embeds, cache=cache, **kwargs)

        # Ensure consistent return format
        if not isinstance(result, dict):
            raise TypeError(f"Model returned {type(result)}, expected dict")

        if 'text_logits' not in result:
            raise KeyError("Model output must contain 'text_logits' key")

        text_logits = result["text_logits"][:, -1]  # [batch, vocab_size]
        batch_size = text_logits.shape[0]

        # Use provided generated_tokens or create empty tensor
        if generated_tokens is None:
            gen_tokens = torch.empty(batch_size, 0, device=text_logits.device, dtype=torch.long)
        else:
            gen_tokens = generated_tokens

        # Apply sampling with top-p and repetition penalty
        predicted_token = self._sample_text_token(
            logits=text_logits,
            generated_tokens=gen_tokens,
            current_step=current_step,
        )

        # ASR tokens use greedy decoding (no sampling)
        if "asr_logits" in result:
            asr_predicted_token = result["asr_logits"][:, -1].argmax(dim=-1)
        else:
            asr_predicted_token = torch.full_like(predicted_token, fill_value=self.model.stt_model.text_pad_id)

        ans = {
            "predicted_token": predicted_token,
            "asr_predicted_token": asr_predicted_token,
            "cache": result.get("cache", None),
        }
        if "function_logits" in result:
            ans["function_predicted_token"] = result["function_logits"][:, -1].argmax(dim=-1)
        return ans

    @staticmethod
    def _extract_special_token_ids_from_nemo(model) -> Set[int]:
        """
        Extract special token IDs from NeMo model's tokenizer.

        NeMo tokenizer uses bos_token, eos_token, pad_token (not bos_token_id).
        Then converts token strings to IDs using token_to_id method.

        Args:
            model: The DuplexS2SExternalSpeechDecoderModel instance

        Returns:
            Set of special token IDs, or empty set if extraction fails
        """
        special_ids = set()
        try:
            tokenizer = model.stt_model.tokenizer

            # Get token strings (NeMo uses bos_token, not bos_token_id)
            bos_token = getattr(tokenizer, 'bos_token', None)
            eos_token = getattr(tokenizer, 'eos_token', None)
            pad_token = getattr(tokenizer, 'pad_token', None)

            # Convert token strings to IDs
            if hasattr(tokenizer, 'token_to_id'):
                for token in [bos_token, eos_token, pad_token]:
                    if token is not None:
                        tid = tokenizer.token_to_id(token)
                        if tid is not None and isinstance(tid, int):
                            special_ids.add(tid)
        except Exception as e:
            pass  # Return empty set on failure

        return special_ids

    def to(self, device_or_dtype: Union[torch.device, torch.dtype]) -> 'NativeModelInterface':
        """Move underlying model to device or convert dtype."""
        self.model = self.model.to(device_or_dtype)
        return self

    def eval(self) -> 'NativeModelInterface':
        """Set underlying model to eval mode."""
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        """Get device of the underlying model."""
        # Try to get device from model parameters
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # No parameters, return CPU
            return torch.device('cpu')

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying model.

        This allows transparent access to model attributes like
        perception, tokenizer, etc.ß
        """
        # Avoid infinite recursion for special attributes
        if name in ('model', '__dict__', '__class__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Delegate to wrapped model
        return getattr(self.model, name)


def create_model(
    model=None,
    engine_type: str = "native",
    vllm_config: Optional[Dict[str, Any]] = None,
    special_token_ids: Optional[Set[int]] = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    **kwargs
) -> ModelInterface:
    """
    Factory function to create appropriate model interface.

    This is the main entry point for creating model interfaces.

    Args:
        model: The base model to wrap (required for "native" engine, optional for "vllm")
        engine_type: Type of engine ("native", "vllm")
        vllm_config: Configuration dict for vLLM engines (required for "vllm")
        special_token_ids: Set of special token IDs (pad, eos, bos) that should bypass sampling.
                          If None (default), will auto-extract from model.tokenizer for tokens:
                          '<s>' (bos), '</s>' (eos), '<SPECIAL_12>' (pad).
                          You can manually provide: {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id}
        top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
        repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
        temperature: Temperature for sampling. 1.0 = no change, 0.0 = greedy. Default: 1.0
        **kwargs: Additional arguments passed to the interface constructor

    Returns:
        ModelInterface instance

    Example:
        >>> # Use native PyTorch model with greedy decoding (default)
        >>> interface = create_model(model, engine_type="native")
        >>>
        >>> # Use native with top-p sampling (special_token_ids auto-extracted from model.tokenizer)
        >>> # Auto-extracts IDs for: '<s>', '</s>', '<SPECIAL_12>'
        >>> interface = create_model(
        >>>     model,
        >>>     engine_type="native",
        >>>     top_p=0.9
        >>> )
        >>>
        >>> # Use native with top-p and repetition penalty (auto-extract special tokens)
        >>> interface = create_model(
        >>>     model,
        >>>     engine_type="native",
        >>>     top_p=0.9,
        >>>     repetition_penalty=1.2
        >>> )
        >>>
        >>> # Manually provide special_token_ids (if auto-extraction fails or you want custom tokens)
        >>> special_ids = {
        >>>     tokenizer.pad_token_id,
        >>>     tokenizer.eos_token_id,
        >>>     tokenizer.bos_token_id
        >>> }
        >>> interface = create_model(
        >>>     model,
        >>>     engine_type="native",
        >>>     special_token_ids=special_ids,
        >>>     top_p=0.9,
        >>>     repetition_penalty=1.2
        >>> )
        >>>
        >>> # Use vLLM with streaming engine
        >>> vllm_cfg = {
        >>>     "model_path": "/path/to/vllm/checkpoint",
        >>>     "max_model_len": 10240,
        >>>     "gpu_memory_utilization": 0.8,
        >>>     "dtype": "bfloat16"
        >>> }
        >>> interface = create_model(
        >>>     engine_type="vllm",
        >>>     vllm_config=vllm_cfg
        >>> )
        >>>
        >>> # Perform inference
        >>> result = interface(input_embeds, cache=cache)
        >>>
        >>> # For repetition penalty, pass generated_tokens and current_step
        >>> result = interface(input_embeds, cache=cache, generated_tokens=prev_tokens, current_step=step)
    """
    engine_type = engine_type.lower()

    if engine_type == "native":
        if model is None:
            raise ValueError("model must be provided for native engine")
        return NativeModel(
            model=model,
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

    elif engine_type == "vllm_eartts":
        if vllm_config is None:
            raise ValueError("vllm_config must be provided for vLLM EARTTS engine")
        # VllmEARTTSModel for TTS inference
        return VllmEARTTSModel(
            **vllm_config,
            model_type="eartts",
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            **kwargs
        )

    elif engine_type.startswith("vllm"):
        if vllm_config is None:
            raise ValueError("vllm_config must be provided for vLLM engine")
        # VllmLLMModel doesn't need the PyTorch model, only the config
        return VllmLLMModel(
            **vllm_config,
            model_type="llm",
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown engine_type: {engine_type}. "
            f"Supported types: 'native', 'vllm', 'vllm_llm', 'vllm_eartts'"
        )
