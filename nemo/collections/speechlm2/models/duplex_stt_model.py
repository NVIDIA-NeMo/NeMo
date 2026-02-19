# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import (
    get_pad_id, 
    create_one_second_silence_template,
    get_silence_embeddings_from_ratio,
)
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.augmentation import AudioAugmenter, DEFAULT_CODEC_SETTINGS
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.label_prep import prepare_labels
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.text_wer import TextWER
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TurnTakingMetrics
from nemo.collections.speechlm2.parts.metrics.empty_text import EmptyTextMetric
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

import glob
import soundfile as sf
import librosa
import numpy as np


class DuplexSTTModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()

        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)
        self.predict_user_text = self.cfg.get("predict_user_text", False)

        # Load LLM first
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()

        # Handle different model types with all their specific configurations
        if 'Nemotron' in self.cfg.pretrained_llm:
            # ====== NEMOTRON-SPECIFIC HANDLING ======
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token = '<SPECIAL_12>'
            # self.user_bos_id = self.tokenizer.text_to_ids('<SPECIAL_13>')[0]
            # self.user_eos_id = self.tokenizer.text_to_ids('<SPECIAL_14>')[0]
            self.user_bos_id = self.tokenizer.text_to_ids('^')[0]
            self.user_eos_id = self.tokenizer.text_to_ids('$')[0]

            self.llm = getattr(llm, self.cfg.get("base_model_name", "backbone"))
            self.lm_head = llm.lm_head
            embed_tokens_name = self.cfg.get("embed_tokens_name", "embeddings")
            self.embed_tokens = getattr(self.llm, embed_tokens_name)

            delattr(self.llm, embed_tokens_name)
        elif 'Qwen2.5' in self.cfg.pretrained_llm:
            # ====== QWEN2.5-SPECIFIC HANDLING ======
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            self.user_bos_id = self.tokenizer.text_to_ids('^')[0]
            self.user_eos_id = self.tokenizer.text_to_ids('$')[0]

            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

            self.llm = llm.model
            self.lm_head = llm.lm_head
            self.embed_tokens = self.llm.embed_tokens

            del self.llm.embed_tokens
        else:
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            self.llm = llm.model
            self.lm_head = llm.lm_head
            self.embed_tokens = self.llm.embed_tokens
            del self.llm.embed_tokens
            self.user_bos_id = self.tokenizer.text_to_ids('^')[0]
            self.user_eos_id = self.tokenizer.text_to_ids('$')[0]

        if self.predict_user_text:
            self.asr_head = copy.deepcopy(self.lm_head)
            self.embed_asr_tokens = copy.deepcopy(self.embed_tokens)
        
        # Initialize function calling head (separate from text head) - always enabled
        self.function_head = copy.deepcopy(self.lm_head)
        logging.info("[Function Calling] Initialized separate function_head (shared embeddings with text channel)")

        maybe_install_lora(self)

        # Load the pretrained streaming ASR model
        setup_speech_encoder(self)

        if self.cfg.get("pretrained_perception_from_s2s", None):
            self.init_perception_from_another_s2s_checkpoint(self.cfg.pretrained_perception_from_s2s)

        if self.cfg.get("pretrained_s2s_model", None):
            logging.info(f"Loading pretrained s2s model from {self.cfg.pretrained_s2s_model}")
            if os.path.isdir(self.cfg.pretrained_s2s_model) and self.cfg.get("incremental_loading", False):
                # Hugging Face format
                from safetensors import safe_open
                import gc
                
                # Load tensors incrementally to avoid OOM
                model_state_dict = self.state_dict()
                loaded_keys = []
                missing_keys = []
                
                with safe_open(os.path.join(self.cfg.pretrained_s2s_model, "model.safetensors"), framework="pt", device="cpu") as f:
                    available_keys = f.keys()
                    for key in available_keys:
                        if key in model_state_dict:
                            # Load tensor and copy to model parameter
                            tensor = f.get_tensor(key)
                            model_state_dict[key].copy_(tensor)
                            loaded_keys.append(key)
                            del tensor  # Free memory immediately
                        else:
                            missing_keys.append(key)
                        
                        # Periodic garbage collection for very large models
                        if len(loaded_keys) % 100 == 0:
                            gc.collect()
                
                logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
                if missing_keys:
                    logging.warning(f"Keys in checkpoint but not in model: {len(missing_keys)} keys")
                
                del model_state_dict
                gc.collect()
            else:
                self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        self._use_fsdp = False
        self._use_tp = False

        # Initialize audio augmenter if any augmentation is enabled
        if (self.cfg.get('use_old_noise_aug', None) or 
            self.cfg.get('use_room_ir_aug', None) or 
            self.cfg.get('use_mic_ir_aug', None) or 
            self.cfg.get('use_codec_aug', None)):
            self.audio_augmenter = AudioAugmenter(sample_rate=self.source_sample_rate)

        # Early interruption augmentation counters for cumulative logging
        self.early_interruption_total = 0
        self.early_interruption_attempted = 0
        self.early_interruption_successful = 0

        # Cache for backchannel file names to avoid repeated glob operations
        if self.cfg.get('backchannel_prob', None) and self.cfg.backchannel_prob > 0:
            self._backchannel_files_cache = {}
        
        # Lazy initialization of silence template
        # We defer creation until first use to avoid device/dtype issues during __init__
        # (model may not be fully moved to GPU yet during __init__)
        self._silence_template_embeddings = None
        self._silence_template_asr_embeddings = None
        self._silence_frames_per_second = None
        self._silence_template_initialized = False
        logging.info("Silence template will be created lazily on first use (after model is moved to device)")

    def init_perception_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            elif os.path.isdir(checkpoint_path):
                logging.info(f"Loading from HuggingFace format directory: {checkpoint_path}")
                pretrained_model = self.__class__.from_pretrained(checkpoint_path)
                checkpoint_state = pretrained_model.state_dict()
                del pretrained_model
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = {
                k.replace("perception.", ""): v for k, v in checkpoint_state.items() if "perception." in k
            }
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.perception.state_dict())
            self.perception.load_state_dict(checkpoint_state, strict=True)

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            elif os.path.isdir(checkpoint_path):
                logging.info(f"Loading from HuggingFace format directory: {checkpoint_path}")
                pretrained_model = self.__class__.from_pretrained(checkpoint_path)
                checkpoint_state = pretrained_model.state_dict()
                del pretrained_model
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
            self,
            input_embeds: Tensor,
            cache=None,
            input_audio_tokens=None,
            seq_mask=None,
            target_text_tokens=None,
    ) -> dict[str, Tensor]:
        """
        Text prediction only (audio_loss_weight=0).
        """
        # Handle different cache parameter names for different models
        if 'Nemotron' in self.cfg.pretrained_llm:
            kwargs = {
                "inputs_embeds": input_embeds,
                "return_dict": True,
                "use_cache": cache is not None,
            }
            if cache is not None:
                kwargs['use_cache'] = True
                kwargs[self.cfg.get("cache_key", "past_key_values")] = cache
            out = self.llm(**kwargs)
        else:
            out = self.llm(
                inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
            )

        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])

        if self.predict_user_text:
            asr_in = out['last_hidden_state']
            asr_logits = self.asr_head(asr_in)  # (B, T, asr_vocab_size)
        
        # Function calling: separate head for function channel, shared vocab with text
        function_in = out['last_hidden_state']
        function_logits = self.function_head(function_in)  # (B, T, vocab_size)

        if not self.training:
            if self.cfg.get("inference_pad_boost", None):
                text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
            if self.cfg.get("inference_bos_boost", None):
                text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost
            if self.cfg.get("inference_eos_boost", None):
                text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost
            
            if self.predict_user_text:
                if self.cfg.get("inference_user_pad_boost", None):
                    asr_logits[:, :, self.text_pad_id] += self.cfg.inference_user_pad_boost
                if self.cfg.get("inference_user_bos_boost", None):
                    asr_logits[:, :, self.user_bos_id] += self.cfg.inference_user_bos_boost
                if self.cfg.get("inference_user_eos_boost", None):
                    asr_logits[:, :, self.text_eos_id] += self.cfg.inference_user_eos_boost

        ans = {"text_logits": text_logits, "function_logits": function_logits}
        if self.predict_user_text:
            ans["asr_logits"] = asr_logits

        if cache is not None:
            if 'Nemotron' in self.cfg.pretrained_llm:
                cache_key = self.cfg.get("cache_key", "cache_params")
                ans["cache"] = getattr(out, cache_key, out.get(cache_key))
            else:
                ans["cache"] = out["past_key_values"]

        return ans

    def _is_noise_augmentation_dataset(self, formatter: str) -> bool:
        if self.cfg.get('force_use_noise_augmentation', False):
            return True
        return formatter != 's2s_duplex_overlap_as_s2s_duplex' and formatter != 'nemo_tarred_to_duplex'

    def _maybe_zero_out_scale_for_asr(self, loss_scale: torch.Tensor, text_labels: torch.Tensor,
                                      batch: dict) -> torch.Tensor:
        """
        Zero out the loss scale after text_bos_id token for ASR datasets.
        """
        if batch['formatter'][0] == 'nemo_tarred_to_duplex':
            for i in range(text_labels.shape[0]):
                bos_indices = (text_labels[i] == self.text_bos_id).nonzero(as_tuple=True)
                if bos_indices[0].numel() > 0:
                    bos_idx = bos_indices[0][0].item()
                    loss_scale[i, bos_idx + 1:, :] = 0
        return loss_scale

    def add_backchannel_to_batch(
        self,
        batch_audio,
        target_tokens,
        source_tokens,
        backchannel_folder,
        snr_db=15,
        backchannel_prob_scale=0.5,
        debug=False,
        debug_save_path=None,
        debug_max_files=16,
        target_audio=None,
    ):
        """
        Add backchannel audio segments to user audio during silence periods (when agent is talking).
        
        Args:
            batch_audio: User audio tensor (B, T_audio)
            target_tokens: Agent tokens (B, T_tokens) - used to identify when agent is talking
            backchannel_folder: Folder containing backchannel audio files
            snr_db: Signal-to-noise ratio for mixing backchannel
            backchannel_prob_scale: Probability of adding backchannel at each silence period
            debug: If True, save augmented audio samples for debugging
            debug_save_path: Path to save debug audio files
            debug_max_files: Maximum number of debug files to save
            target_audio: Agent audio tensor (B, T_audio) - optional, saved for debugging
        """
        # Check if we should save debug files
        should_save_debug = False
        if debug and debug_save_path is not None:
            os.makedirs(debug_save_path, exist_ok=True)
            existing_files = glob.glob(os.path.join(debug_save_path, "*.wav"))
            if len(existing_files) < debug_max_files:
                should_save_debug = True
        
        batch_size, audio_length = batch_audio.shape
        
        # Use cached backchannel file list to avoid repeated glob operations
        if backchannel_folder not in self._backchannel_files_cache:
            # backchannel_folder already contains the glob pattern (e.g., "/path/to/audio/*")
            # so we need to add the .wav extension to the pattern
            backchannel_files = [f for f in glob.glob(backchannel_folder + ".wav")]
            if not backchannel_files:
                raise ValueError(f"No backchannel files found matching pattern: {backchannel_folder}.wav")
            self._backchannel_files_cache[backchannel_folder] = backchannel_files
        else:
            backchannel_files = self._backchannel_files_cache[backchannel_folder]
        # Process each sample in the batch
        for i in range(batch_size):
            agent_tokens = target_tokens[i]
            user_tokens = source_tokens[i]
            # Find agent BOS and EOS positions
            agent_bos_positions = torch.where(agent_tokens == self.text_bos_id)[0]
            agent_eos_positions = torch.where(agent_tokens == self.text_eos_id)[0]
            if len(agent_bos_positions) == 0:
                continue  # No agent speech, skip
            
            # Convert token positions to audio sample positions
            # Each token represents frame_length from config (default ~80ms) of audio
            token_duration_seconds = 0.08  # Frame length
            samples_per_token = int(token_duration_seconds * self.source_sample_rate)
            
            # Build agent turns (BOS-EOS pairs) from agent tokens
            agent_turns = []  # List of (bos_token, eos_token) tuples
            
            for bos_idx in agent_bos_positions:
                bos_idx = bos_idx.item()
                # Find the first EOS that comes AFTER this BOS
                eos_candidates = agent_eos_positions[agent_eos_positions > bos_idx]
                if len(eos_candidates) > 0:
                    # Found a valid EOS after this BOS
                    eos_idx = eos_candidates[0].item()
                    agent_turns.append((bos_idx, eos_idx))
                else:
                    # No EOS found after this BOS (unpaired BOS at the end)
                    # Mark from BOS to end of sequence as agent speaking
                    agent_turns.append((bos_idx, len(agent_tokens) - 1))
            
            # Build user turns directly from source_tokens BOS/EOS positions
            user_bos_positions = torch.where(user_tokens == self.user_bos_id)[0]
            user_eos_positions = torch.where(user_tokens == self.user_eos_id)[0]
            # Format: [(start_sample, end_sample), ...]
            user_regions = []
            
            for bos_idx in user_bos_positions:
                bos_idx = bos_idx.item()
                # Find the first EOS that comes AFTER this BOS
                eos_candidates = user_eos_positions[user_eos_positions > bos_idx]
                if len(eos_candidates) > 0:
                    eos_idx = eos_candidates[0].item()
                    # Convert token positions to audio samples
                    user_start_sample = bos_idx * samples_per_token
                    user_end_sample = min((eos_idx + 1) * samples_per_token, audio_length)
                    user_regions.append((user_start_sample, user_end_sample))
                else:
                    # Unpaired BOS at the end
                    user_start_sample = bos_idx * samples_per_token
                    user_end_sample = audio_length
                    user_regions.append((user_start_sample, user_end_sample))
            
            # Add backchannel to each agent turn with some probability
            # Only 1 backchannel per agent turn, placed randomly within the turn
            for turn_idx, (bos_token, eos_token) in enumerate(agent_turns):
                # Convert token positions to audio samples
                start_sample = bos_token * samples_per_token
                end_sample = min((eos_token + 1) * samples_per_token, audio_length)
                
                if random.random() > backchannel_prob_scale:
                    continue
                
                region_length = end_sample - start_sample
                if region_length < self.source_sample_rate * 0.3:  # Skip very short regions (<0.3s)
                    continue
                
                # Load a random backchannel audio file
                backchannel_path = random.choice(backchannel_files)
                backchannel_audio, sr = sf.read(backchannel_path, dtype='float32')

                # Resample if needed
                if sr != self.source_sample_rate:
                    backchannel_audio = librosa.resample(
                        backchannel_audio, orig_sr=sr, target_sr=self.source_sample_rate
                    )

                # Convert to mono if stereo
                if len(backchannel_audio.shape) > 1:
                    backchannel_audio = np.mean(backchannel_audio, axis=1)
                
                backchannel_length = len(backchannel_audio)
                # Skip if backchannel is longer than the agent's turn
                if backchannel_length > region_length:
                    continue
                
                # Randomly place the backchannel within the agent's speaking segment
                # Ensure there's space for the full backchannel
                max_start_offset = region_length - backchannel_length
                if max_start_offset > 0:
                    # Bias placement toward the middle of the agent turn while keeping enough margin
                    margin = int(0.05 * region_length)
                    min_offset = max(0, margin)
                    max_offset = max_start_offset - margin
                    if max_offset <= min_offset:
                        min_offset = 0
                        max_offset = max_start_offset
                    if max_offset > min_offset:
                        peak = (min_offset + max_offset) / 2
                        sampled_offset = random.triangular(min_offset, max_offset, peak)
                        random_offset = int(round(sampled_offset))
                    else:
                        random_offset = min_offset
                    random_offset = min(max(random_offset, 0), max_start_offset)
                else:
                    random_offset = 0
                
                insertion_point = start_sample + random_offset
                
                # Convert to tensor
                backchannel_tensor = torch.tensor(
                    backchannel_audio, dtype=batch_audio.dtype, device=batch_audio.device
                )
                
                # Find the previous user turn to match its loudness
                # user_regions[turn_idx] corresponds to the user speech before agent_turns[turn_idx]
                user_rms = None
                if turn_idx < len(user_regions):
                    user_start, user_end = user_regions[turn_idx]
                    # Calculate RMS of the previous user turn
                    user_segment = batch_audio[i, user_start:user_end]
                    if len(user_segment) > 0:
                        user_rms = torch.sqrt(torch.mean(user_segment**2) + 1e-8)
                
                # Scale backchannel to match user's loudness (or use SNR-based if no user turn found)
                backchannel_rms = torch.sqrt(torch.mean(backchannel_tensor**2) + 1e-8)
                
                if user_rms is not None and user_rms > 1e-6:
                    # Match the user's speaking loudness
                    scaling_factor = user_rms / backchannel_rms
                else:
                    # Fallback: use SNR-based scaling with local signal
                    window_start = max(start_sample, insertion_point - int(0.5 * self.source_sample_rate))
                    window_end = min(end_sample, insertion_point + backchannel_length + int(0.5 * self.source_sample_rate))
                    signal_power = torch.mean(batch_audio[i, window_start:window_end]**2) + 1e-8
                    backchannel_power = backchannel_rms**2
                    target_backchannel_power = signal_power / (10 ** (snr_db / 10))
                    scaling_factor = torch.sqrt(target_backchannel_power / backchannel_power)
                
                backchannel_tensor = backchannel_tensor * scaling_factor
                
                # Add backchannel at the random position within the agent's turn
                batch_audio[i, insertion_point:insertion_point+backchannel_length] += backchannel_tensor
        
        # Save debug audio if enabled and we haven't reached the limit
        if should_save_debug:
            try:
                import time
                
                # Get GPU rank for multi-GPU training to avoid filename collisions
                try:
                    if torch.distributed.is_initialized():
                        rank = torch.distributed.get_rank()
                    else:
                        rank = 0
                except:
                    rank = 0
                
                # Save the first sample in the batch
                timestamp = int(time.time() * 1000000)  # Use microseconds for better uniqueness
                
                # Save user audio with backchannel
                user_debug_filename = f"backchannel_debug_rank{rank}_{timestamp}_user.wav"
                user_debug_filepath = os.path.join(debug_save_path, user_debug_filename)
                user_audio_to_save = batch_audio[0].detach().cpu().numpy()
                sf.write(user_debug_filepath, user_audio_to_save, self.source_sample_rate)
                logging.info(f"Saved user audio with backchannel to: {user_debug_filepath}")
                
                # Save agent audio if available
                if target_audio is not None:
                    agent_debug_filename = f"backchannel_debug_rank{rank}_{timestamp}_agent.wav"
                    agent_debug_filepath = os.path.join(debug_save_path, agent_debug_filename)
                    agent_audio_to_save = target_audio[0].detach().cpu().numpy()
                    sf.write(agent_debug_filepath, agent_audio_to_save, self.source_sample_rate)
                    logging.info(f"Saved agent audio to: {agent_debug_filepath}")
                
                should_save_debug = False  # Only save once per batch to avoid too many files
            except Exception as e:
                logging.warning(f"Failed to save debug audio: {e}")
        
        return batch_audio

    def _convert_pad_to_sil(self, target_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Convert pad tokens to sil tokens when agent is in listening state.
        """
        if 'Nemotron' in self.cfg.pretrained_llm:
            sil_id = self.tokenizer.tokenizer._tokenizer.token_to_id('<SPECIAL_11>')
        elif 'Qwen2.5' in self.cfg.pretrained_llm:
            sil_id = self.tokenizer.tokenizer._tokenizer.token_to_id('<|object_ref_start|>')
        else:
            logging.warning("Model type not supported for sil_token conversion, skipping conversion")
            return target_tokens, None

        if sil_id is None:
            logging.warning("sil_token not found in tokenizer vocabulary, skipping conversion")
            return target_tokens, None

        target_tokens = target_tokens.clone()
        B, T = target_tokens.shape

        for b in range(B):
            inside_speech = False

            for t in range(T):
                token = target_tokens[b, t].item()

                if token == self.text_bos_id:
                    inside_speech = True
                elif token == self.text_eos_id:
                    inside_speech = False
                elif token == self.text_pad_id and not inside_speech:
                    target_tokens[b, t] = sil_id

        return target_tokens, sil_id

    def _log_long_list(self, tag: str, values: list[int], chunk_size: int) -> None:
        if not values:
            logging.info(f"{tag}: []")
            return
        if chunk_size <= 0:
            logging.info(f"{tag}: {values}")
            return
        total = len(values)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            logging.info(f"{tag} [{start}:{end}]: {values[start:end]}")

    def _get_function_call_special_tokens(self):
        """
        Get function calling special tokens based on model type.
        Model-specific logic ensures portability across different LLMs.
        """
        if 'Nemotron' in self.cfg.pretrained_llm:
            # Nemotron uses <SPECIAL_XX> tokens
            sotc_token = '<SPECIAL_20>'  # Start Of Tool Call
            eotc_token = '<SPECIAL_21>'  # End Of Tool Call  
            eotr_token = '<SPECIAL_22>'  # End Of Tool Response
        elif 'Qwen' in self.cfg.pretrained_llm:
            # Qwen might use different tokens - configure as needed
            # For now, using same tokens if available in Qwen tokenizer
            sotc_token = '<SPECIAL_20>'
            eotc_token = '<SPECIAL_21>'
            eotr_token = '<SPECIAL_22>'
        else:
            # Default/fallback tokens
            sotc_token = '<SPECIAL_20>'
            eotc_token = '<SPECIAL_21>'
            eotr_token = '<SPECIAL_22>'
        
        # Get token IDs
        sotc_id = self.tokenizer.text_to_ids(sotc_token)[0]
        eotc_id = self.tokenizer.text_to_ids(eotc_token)[0]
        eotr_id = self.tokenizer.text_to_ids(eotr_token)[0]
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[FC Model] Using special tokens for {self.cfg.pretrained_llm}: "
                        f"SOTC={sotc_token}({sotc_id}), EOTC={eotc_token}({eotc_id}), EOTR={eotr_token}({eotr_id})")
        
        return sotc_id, eotc_id, eotr_id

    def _build_function_calling_channel(self, batch: dict, seq_length: int) -> tuple:
        """
        Build function calling channel using insertion approach to expand sequence length.
        
        According to the architecture diagram:
        - Function calls/responses are INSERTED at specific positions
        - Sequence length expands from L to L+F where F is total function token length
        - Agent text channel will have PAD at insertion positions
        - User audio channel will have silence (zeros) at insertion positions
        
        Args:
            batch: Batch dictionary containing function calling data
            seq_length: Current sequence length (AFTER system prompt prepending if applicable)
        
        Returns:
            function_channel: Tensor of shape [B, T_expanded] with function tokens
            function_loss_mask: Tensor of shape [B, T_expanded] - True for calls, False for responses/padding
            insertion_positions: List[List[Tuple[int, int]]] - (position, length) pairs per batch item
        """
        B = batch["function_calls"].shape[0] if batch.get("function_calls") is not None else len(batch["target_tokens"])
        device = batch["target_tokens"].device
        
        # Get model-specific special token IDs
        sotc_id, eotc_id, eotr_id = self._get_function_call_special_tokens()
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[FC Model] Building function calling channel with INSERTION approach")
            logging.info(f"[FC Model] Batch size: {B}, Sequence length (with prompt): {seq_length}")
            logging.info(f"[FC Model] Special tokens: SOTC={sotc_id}, EOTC={eotc_id}, EOTR={eotr_id}")
        
        # If no function calling data, return empty channel with no insertions
        if batch.get("function_calls") is None:
            if self.cfg.get("debug_fc", False):
                logging.info("[FC Model] No function calling data in batch")
            function_channel = torch.full((B, seq_length), self.text_pad_id, dtype=torch.long, device=device)
            function_loss_mask = torch.zeros((B, seq_length), dtype=torch.bool, device=device)
            insertion_positions = [[] for _ in range(B)]
            return function_channel, function_loss_mask, insertion_positions
        
        function_calls = batch["function_calls"]  # [B, num_turns, max_call_len]
        function_call_lengths = batch["function_call_lengths"]  # [B, num_turns]
        function_call_steps = batch["function_call_steps"]  # [B, num_turns]
        function_responses = batch["function_responses"]  # [B, num_turns, max_response_len]
        function_response_lengths = batch["function_response_lengths"]  # [B, num_turns]
        function_response_steps = batch["function_response_steps"]  # [B, num_turns]
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[FC Model] Function calls shape: {function_calls.shape}")
            logging.info(f"[FC Model] Function responses shape: {function_responses.shape}")
        
        # Build function channel per batch item using efficient insertion
        batch_channels = []
        batch_loss_masks = []
        batch_insertions = []
        num_turns = function_calls.shape[1]
        
        for b in range(B):
            # Calculate per-sample prompt offset if system prompt is present
            # function_call_steps/function_response_steps are in ORIGINAL coordinate space (without prompt)
            # We need to add prompt_offset to get positions in CURRENT coordinate space (with prompt)
            prompt_offset = 0
            if "prompt_token_lens" in batch and batch["prompt_token_lens"] is not None:
                prompt_offset = batch["prompt_token_lens"][b].item()
                if self.cfg.get("debug_fc", False):
                    logging.info(f"[FC Model] System prompt offset (batch {b}): {prompt_offset} frames")
            # Collect all function call/response events with their positions
            events = []  # List of (position, tokens, is_call) tuples
            
            for turn_idx in range(num_turns):
                # Process function call
                call_step_original = function_call_steps[b, turn_idx].item()
                call_length = function_call_lengths[b, turn_idx].item()
                
                if call_step_original >= 0 and call_length > 0:
                    # IMPORTANT: Add prompt offset to get position in current coordinate space
                    call_step_adjusted = call_step_original + prompt_offset
                    
                    # Extract call tokens and wrap with special tokens: <SOTC> tokens <EOTC>
                    call_tokens = function_calls[b, turn_idx, :call_length]
                    wrapped_call = torch.cat([
                        torch.tensor([sotc_id], device=device, dtype=torch.long),
                        call_tokens,
                        torch.tensor([eotc_id], device=device, dtype=torch.long)
                    ])
                    events.append((call_step_adjusted, wrapped_call, True))  # True = compute loss
                    if self.cfg.get("fc_log", False):
                        wrapped_call_text = self.tokenizer.ids_to_text(wrapped_call.tolist())
                        logging.info(f"[FC Model] Batch {b}, Turn {turn_idx}: Call at step {call_step_adjusted} (original={call_step_original}, offset={prompt_offset}), length {len(wrapped_call)}, wrapped_call: {wrapped_call}, wrapped_call_text: {wrapped_call_text}")
                
                # Process function response
                response_step_original = function_response_steps[b, turn_idx].item()
                response_length = function_response_lengths[b, turn_idx].item()
                
                if response_step_original >= 0 and response_length > 0:
                    # IMPORTANT: Add prompt offset to get position in current coordinate space
                    response_step_adjusted = response_step_original + prompt_offset
                    
                    # Insert response content first (without loss - from API)
                    # Sequence: <EOTC> <TOOLRESPONSE> response_content </TOOLRESPONSE> <EOTR>
                    response_tokens = function_responses[b, turn_idx, :response_length]
                    events.append((response_step_adjusted, response_tokens, False))  # False = no loss on response content
                    
                    # Then insert EOTR marker after response (with loss - model should learn this marker)
                    eotr_marker = torch.tensor([eotr_id], device=device, dtype=torch.long)
                    events.append((response_step_adjusted, eotr_marker, True))  # True = compute loss on EOTR
                    if self.cfg.get("fc_log", False):
                        response_text = self.tokenizer.ids_to_text(response_tokens.tolist())
                        logging.info(f"[FC Model] Batch {b}, Turn {turn_idx}: Response content at step {response_step_adjusted}, length {len(response_tokens)}, response_text: {response_text}")
                        logging.info(f"[FC Model] Batch {b}, Turn {turn_idx}: EOTR marker after response at step {response_step_adjusted} (original={response_step_original}, offset={prompt_offset})")
            
            # Build channel by inserting function tokens at specified positions
            channel_tokens = []
            loss_mask = []
            insertions = []
            current_pos = 0
            
            for insert_pos, tokens, compute_loss in events:
                # Add PAD tokens from current position to insertion point (both in original space)
                pad_length = insert_pos - current_pos
                if pad_length > 0:
                    channel_tokens.extend([self.text_pad_id] * pad_length)
                    loss_mask.extend([True] * pad_length)  # Enable PAD loss to prevent hallucination
                
                # Insert function tokens (this expands the sequence)
                channel_tokens.extend(tokens.tolist())
                loss_mask.extend([compute_loss] * len(tokens))
                
                # Track insertion for expanding other channels
                insertions.append((insert_pos, len(tokens)))
                
                # Update current position to AFTER insertion point in ORIGINAL space
                # (We stay in original coordinate space, not expanded space)
                current_pos = insert_pos
            
            # Add remaining PAD tokens from last insertion point to end of original sequence
            remaining = seq_length - current_pos
            if remaining > 0:
                channel_tokens.extend([self.text_pad_id] * remaining)
                loss_mask.extend([True] * remaining)  # Enable PAD loss to prevent hallucination
            
            # Convert to tensors
            batch_channels.append(torch.tensor(channel_tokens, dtype=torch.long, device=device))
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.bool, device=device))
            batch_insertions.append(insertions)
            
            if self.cfg.get("debug_fc", False):
                expanded_length = len(channel_tokens)
                total_inserted = sum(length for _, length in insertions)
                logging.info(f"[FC Model] Batch {b}: {len(events)} events, {len(insertions)} insertions")
                logging.info(f"[FC Model] Batch {b}: Original length {seq_length} → Expanded length {expanded_length} (inserted {total_inserted})")
        
        # Pad all batch items to maximum expanded length
        max_expanded_length = max(len(ch) for ch in batch_channels)
        function_channel = torch.full((B, max_expanded_length), self.text_pad_id, dtype=torch.long, device=device)
        function_loss_mask = torch.zeros((B, max_expanded_length), dtype=torch.bool, device=device)
        
        for b in range(B):
            length = len(batch_channels[b])
            function_channel[b, :length] = batch_channels[b]
            function_loss_mask[b, :length] = batch_loss_masks[b]
        
        if self.cfg.get("debug_fc", False):
            non_pad = (function_channel != self.text_pad_id).sum().item()
            loss_true = function_loss_mask.sum().item()
            logging.info(f"[FC Model] Final function channel shape: {function_channel.shape}")
            logging.info(f"[FC Model] Non-PAD tokens: {non_pad}, Loss computed on: {loss_true} tokens")
            
            # Print first batch item's function channel in detail
            logging.info(f"[FC Model] ============ FUNCTION CHANNEL VERIFICATION (Batch 0) ============")
            fc_sample = function_channel[0]
            fc_mask_sample = function_loss_mask[0]
            
            # Find non-PAD positions
            non_pad_positions = (fc_sample != self.text_pad_id).nonzero(as_tuple=True)[0]
            if len(non_pad_positions) > 0:
                logging.info(f"[FC Model] Non-PAD positions: {non_pad_positions.tolist()}")
                logging.info(f"[FC Model] Function tokens at those positions:")
                for pos in non_pad_positions[:20]:  # Show first 20
                    pos_val = pos.item()
                    token_id = fc_sample[pos_val].item()
                    compute_loss = fc_mask_sample[pos_val].item()
                    # Decode token
                    try:
                        token_text = self.tokenizer.ids_to_text([token_id])
                    except:
                        token_text = f"<ID:{token_id}>"
                    loss_str = "LOSS=YES" if compute_loss else "LOSS=NO"
                    logging.info(f"[FC Model]   Pos {pos_val}: Token={token_text} (id={token_id}) {loss_str}")
            else:
                logging.info(f"[FC Model] No function calls/responses in this batch")
            logging.info(f"[FC Model] ================================================================")
        return function_channel, function_loss_mask, batch_insertions

    def _compute_audio_duration_from_frames(self, num_frames: int, subsampling_factor: float, sample_rate: int) -> float:
        """
        Reverse calculation: Convert number of embedding frames back to audio duration.
        
        This reverses the typical compute_num_frames calculation:
            Forward:  num_frames = floor(audio_samples / subsampling_factor)
            Reverse:  audio_samples = num_frames * subsampling_factor
            Duration: duration_seconds = audio_samples / sample_rate
        
        Args:
            num_frames: Number of embedding frames
            subsampling_factor: The ratio of audio samples to embedding frames
            sample_rate: Audio sampling rate (e.g., 16000 Hz)
            
        Returns:
            duration_seconds: Duration in seconds
        """
        audio_samples = num_frames * subsampling_factor
        duration_seconds = audio_samples / sample_rate
        return duration_seconds
    
    def _ensure_silence_fps_initialized(self):
        """
        Ensure the silence frames-per-second ratio is initialized.
        
        This should be called from on_train_start() for eager initialization,
        or as a fallback during inference if not yet initialized.
        """
        if self._silence_frames_per_second is not None:
            return  # Already initialized
            
        if not hasattr(self, 'perception') or self.perception is None:
            logging.warning("Perception module not available, skipping silence FPS initialization")
            return
        
        # Create 1-second template to get frames-per-second ratio
        logging.info("[Silence Init] Creating 1-second silence template to compute frames-per-second ratio...")
        try:
            device = next(self.perception.parameters()).device
            _, training_fps = create_one_second_silence_template(
                perception_module=self.perception,
                sample_rate=self.source_sample_rate,
                device=device,
            )
            self._silence_frames_per_second = training_fps
            
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                logging.info(f"[Silence Init] Rank {rank}: Silence ratio = {training_fps:.2f} frames/sec")
            else:
                logging.info(f"[Silence Init] Silence ratio = {training_fps:.2f} frames/sec")
        except Exception as e:
            logging.error(f"[Silence Init] Failed to create silence FPS template: {e}")
            raise
    
    def _ensure_silence_template_initialized(self):
        """
        Lazily initialize the full 60-second silence template for INFERENCE.
        
        This is only needed during inference when function responses are injected.
        Typically only called during inference/validation, not training.
        """
        if self._silence_template_initialized:
            return  # Already initialized
            
        if not hasattr(self, 'perception') or self.perception is None:
            logging.warning("Perception module not available, skipping silence template initialization")
            return
        
        # Create silence template (60 seconds for INFERENCE)
        silence_template_seconds = 60
        logging.info(f"[Silence Template Init] Creating {silence_template_seconds}-second silence template for inference...")
        
        try:
            silence_embeddings, silence_asr_embeddings, silence_fps = self._create_silence_template(
                duration_seconds=silence_template_seconds,
            )
            
            # Move to CPU for storage to save GPU memory
            self._silence_template_embeddings = silence_embeddings.cpu()
            self._silence_template_asr_embeddings = silence_asr_embeddings.cpu()
            self._silence_frames_per_second = silence_fps
            self._silence_template_initialized = True
            
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                logging.info(f"[Rank {rank}] 60s silence template created: {self._silence_template_embeddings.shape[0]} frames "
                           f"({silence_template_seconds}s @ {self._silence_frames_per_second:.2f} frames/sec)")
            else:
                logging.info(f"[Silence Template Init] Silence template created: {self._silence_template_embeddings.shape[0]} frames "
                           f"({silence_template_seconds}s @ {self._silence_frames_per_second:.2f} frames/sec)")
        except Exception as e:
            logging.error(f"[Silence Template Init] Failed to create 60s silence template: {e}")
            raise
    
    def on_train_start(self) -> None:
        """
        PyTorch Lightning hook called when training starts.
        
        Initialize silence templates here to ensure:
        - Model is fully on GPU and distributed setup is complete
        - All ranks participate together (no deadlocks)
        - Happens before any batches are processed
        """
        super().on_train_start()
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            logging.info(f"[Rank {rank}] on_train_start: Initializing silence templates for training...")
        else:
            logging.info("[on_train_start] Initializing silence templates for training...")
        
        # Initialize the 1-second template for training (all ranks participate)
        self._ensure_silence_fps_initialized()
        
        if torch.distributed.is_initialized():
            # Synchronize all ranks before proceeding
            torch.distributed.barrier()
            logging.info(f"[Rank {rank}] Silence FPS initialization complete, training ready to start.")
        else:
            logging.info("Silence FPS initialization complete, training ready to start.")
    
    def _create_silence_template(
        self,
        duration_seconds: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Create a silence embedding template of specified duration.
        
        This template will be used to efficiently insert silence during function calling
        by slicing frame-by-frame instead of regenerating silence each time.
        
        Args:
            duration_seconds: Duration of silence template in seconds (e.g., 60)
            
        Returns:
            silence_embeddings: Tensor of shape [num_frames, hidden_size] 
            frames_per_second: Number of embedding frames per second
        """
        # Create silence audio
        num_samples = int(duration_seconds * self.source_sample_rate)
        
        # Use the same device and dtype as regular user audio
        # Audio is always float32 by default, same as user audio from dataset
        perception_device = next(self.perception.parameters()).device
        perception_dtype = next(self.perception.parameters()).dtype
        audio_dtype = torch.float32  # Standard dtype for audio, same as user audio
        
        logging.info(f"[Silence Template] Creating with device={perception_device}, audio_dtype={audio_dtype} (matching user audio)")
        logging.info(f"[Silence Template] Perception module parameters dtype: {perception_dtype}")
        
        silence_audio = torch.zeros(1, num_samples, device=perception_device, dtype=audio_dtype)
        audio_length = torch.tensor([num_samples], device=perception_device, dtype=torch.long)
        
        logging.info(f"[Silence Template] Created silence_audio with dtype={silence_audio.dtype}, device={silence_audio.device}")
        
        # Encode through perception module with autocast if needed
        # During training, autocast is enabled which handles float32->bfloat16 conversion
        # During init, we need to explicitly enable it to match training behavior
        with torch.no_grad():
            # Enable autocast if perception module uses bfloat16 (to match training behavior)
            if perception_dtype == torch.bfloat16 and perception_device.type == 'cuda':
                logging.info(f"[Silence Template] Enabling autocast for bfloat16 perception module")
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    silence_encoded, encoded_length, silence_asr = self.perception(
                        input_signal=silence_audio,
                        input_signal_length=audio_length,
                        return_encoder_emb=True
                    )
            else:
                silence_encoded, encoded_length, silence_asr = self.perception(
                    input_signal=silence_audio,
                    input_signal_length=audio_length,
                    return_encoder_emb=True
                )
        
        # Extract embeddings (remove batch dimension)
        num_frames = encoded_length[0].item()
        silence_embeddings = silence_encoded[0, :num_frames, :].clone()  # [num_frames, H]
        silence_asr_embeddings = silence_asr[0, :num_frames, :].clone()  # [num_frames, H]
        
        # Calculate frames per second ratio
        frames_per_second = num_frames / duration_seconds
        
        # Log statistics to check if silence is uniform across time
        if self.cfg.get("debug_fc", False) or True:  # Always log this for now
            logging.info(f"[Silence Template] Statistics for {duration_seconds}s template ({num_frames} frames):")
            
            # Overall statistics
            silence_mean = silence_embeddings.mean().item()
            silence_std = silence_embeddings.std().item()
            silence_min = silence_embeddings.min().item()
            silence_max = silence_embeddings.max().item()
            logging.info(f"[Silence Template]   Overall: mean={silence_mean:.6f}, std={silence_std:.6f}, min={silence_min:.6f}, max={silence_max:.6f}")
            
            # Compare different time positions to check uniformity
            if num_frames >= 10:
                # Sample frames at 0%, 25%, 50%, 75%, 100% positions
                positions = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
                logging.info(f"[Silence Template]   Comparing frames at different positions:")
                
                for i, pos in enumerate(positions):
                    frame = silence_embeddings[pos]  # [H]
                    frame_mean = frame.mean().item()
                    frame_std = frame.std().item()
                    frame_norm = torch.norm(frame).item()
                    logging.info(f"[Silence Template]     Frame {pos} ({pos/num_frames*100:.0f}%): mean={frame_mean:.6f}, std={frame_std:.6f}, L2norm={frame_norm:.6f}")
                
                # Check similarity between first and last frame
                first_frame = silence_embeddings[0]
                last_frame = silence_embeddings[-1]
                cosine_sim = torch.nn.functional.cosine_similarity(first_frame.unsqueeze(0), last_frame.unsqueeze(0)).item()
                l2_distance = torch.norm(first_frame - last_frame).item()
                logging.info(f"[Silence Template]   First vs Last frame: cosine_similarity={cosine_sim:.6f}, L2_distance={l2_distance:.6f}")
                
                # Check frame-to-frame variance
                frame_diffs = torch.diff(silence_embeddings, dim=0)  # [num_frames-1, H]
                avg_frame_diff = torch.norm(frame_diffs, dim=1).mean().item()
                max_frame_diff = torch.norm(frame_diffs, dim=1).max().item()
                logging.info(f"[Silence Template]   Frame-to-frame variation: avg_L2_diff={avg_frame_diff:.6f}, max_L2_diff={max_frame_diff:.6f}")
                
                # Recommendation
                if cosine_sim > 0.99 and avg_frame_diff < 0.01:
                    logging.info(f"[Silence Template]   ✓ Silence is HIGHLY UNIFORM - can use single-frame repetition for efficiency")
                elif cosine_sim > 0.95 and avg_frame_diff < 0.1:
                    logging.info(f"[Silence Template]   ~ Silence is MOSTLY UNIFORM - single-frame repetition likely acceptable")
                else:
                    logging.info(f"[Silence Template]   ✗ Silence varies across time - full template needed")
        
        return silence_embeddings, silence_asr_embeddings, frames_per_second
    
    def _get_silence_embeddings(
        self, 
        length: int, 
        subsampling_factor: float = None
    ) -> torch.Tensor:
        """
        Generate proper silence embeddings for TRAINING by encoding actual silence audio.
        
        Uses pre-computed frames-per-second ratio to calculate exact audio duration needed.
        Device and dtype are auto-detected from the perception module.
        
        Args:
            length: Number of embedding frames needed
            subsampling_factor: Ratio of audio samples to embedding frames (optional, for compatibility)
            
        Returns:
            silence_embeddings: [length, hidden_size] - proper silence embeddings
        """
        # Ensure FPS ratio is initialized (lazy initialization on first use during training)
        # This only creates a 1-second template to compute the ratio (much faster than 60s)
        self._ensure_silence_fps_initialized()
        
        # Use the pre-computed ratio to generate silence for training
        # Device and dtype are auto-detected from perception module
        return get_silence_embeddings_from_ratio(
            perception_module=self.perception,
            frames_per_second=self._silence_frames_per_second,
            sample_rate=self.source_sample_rate,
            length=length,
        )
    
    def _get_silence_embeddings_from_template(
        self,
        length: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Generate silence embeddings for INFERENCE by slicing from pre-computed template.
        
        This is used during inference when function responses are injected to extend
        the user audio channel, maintaining consistency with training's expanded sequences.
        
        Example:
            - Template has 3000 frames (60 seconds @ 50 fps)
            - Need 20 frames → slice template[0:20]
            - Need 3500 frames → slice template[0:3000] + repeat template[-1] 500 times
        
        Args:
            length: Number of embedding frames needed
            device: Device to create tensors on (optional, defaults to template's device)
            dtype: Data type for the embeddings (optional, defaults to template's dtype)
            
        Returns:
            silence_embeddings: [length, hidden_size] - silence embeddings of exact length
        """
        # Ensure template is initialized (lazy initialization on first use)
        self._ensure_silence_template_initialized()
        
        # Use template's device and dtype if not specified
        if device is None:
            device = self._silence_template_embeddings.device
        if dtype is None:
            dtype = self._silence_template_embeddings.dtype
            
        # Move template to target device if needed
        template = self._silence_template_embeddings.to(device=device, dtype=dtype)
        template_length = template.shape[0]
        
        if length <= template_length:
            # Simple case: slice from template
            result = template[:length].clone()
        else:
            # Requested length exceeds template: slice all + repeat last frame
            num_repeats = length - template_length
            last_frame = template[-1:].expand(num_repeats, -1)  # [num_repeats, H]
            result = torch.cat([template, last_frame], dim=0)  # [length, H]
        
        # Log statistics about the sliced silence (only occasionally to avoid spam)
        if self.cfg.get("debug_fc", False) and random.random() < 0.05:  # 5% sampling
            result_mean = result.mean().item()
            result_std = result.std().item()
            result_norm = torch.norm(result).item()
            logging.info(f"[Silence Slice] Requested {length} frames: mean={result_mean:.6f}, std={result_std:.6f}, L2norm={result_norm:.6f}")
            
            if length > template_length:
                logging.info(f"[Silence Slice]   Extended beyond template: used {template_length} real + {num_repeats} repeated frames")
        
        return result

    def _get_silence_asr_embeddings_from_template(
        self,
        length: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Generate silence ASR embeddings for INFERENCE by slicing from pre-computed template.
        """
        self._ensure_silence_template_initialized()

        if self._silence_template_asr_embeddings is None:
            raise RuntimeError("Silence ASR template embeddings are not initialized.")

        if device is None:
            device = self._silence_template_asr_embeddings.device
        if dtype is None:
            dtype = self._silence_template_asr_embeddings.dtype

        template = self._silence_template_asr_embeddings.to(device=device, dtype=dtype)
        template_length = template.shape[0]

        if length <= template_length:
            result = template[:length].clone()
        else:
            num_repeats = length - template_length
            last_frame = template[-1:].expand(num_repeats, -1)
            result = torch.cat([template, last_frame], dim=0)

        return result

    def _sync_fc_insertions_collective(self, total_insertions_this_rank: int, subsampling_factor: float = 8.0):
        """Participate in the same NCCL collectives as _expand_channels_with_insertions without modifying tensors.
        Used for minimal/dropped batch so this rank does not desync (avoids timeout).
        """
        if not torch.distributed.is_initialized():
            return
        device = next(self.parameters()).device
        max_insertions_tensor = torch.tensor([total_insertions_this_rank], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(max_insertions_tensor, op=torch.distributed.ReduceOp.MAX)
        max_insertions_across_ranks = max_insertions_tensor.item()
        dummy_calls_needed = max_insertions_across_ranks - total_insertions_this_rank
        if dummy_calls_needed > 0:
            for _ in range(dummy_calls_needed):
                _ = self._get_silence_embeddings(1, subsampling_factor)

    def _expand_channels_with_insertions(
        self,
        target_tokens: torch.Tensor,
        source_encoded: torch.Tensor,
        insertion_positions: list,
        subsampling_factor: float = 8.0,
        source_tokens: torch.Tensor = None,
    ) -> tuple:
        """
        Expand agent text, user audio, and optionally user text channels by inserting PAD/silence at function call positions.
        
        According to the architecture diagram:
        - Agent text channel gets PAD tokens at function call positions
        - User audio channel gets silence (proper encoded silence) at function call positions
        - User text channel (if provided) gets PAD tokens at function call positions
        - Original content is preserved by shifting right
        
        Args:
            target_tokens: [B, L] - original agent text tokens
            source_encoded: [B, L, H] - original user audio encoding  
            insertion_positions: List[List[Tuple[int, int]]] - (position, length) pairs per batch item
            subsampling_factor: Ratio of audio samples to embedding frames (computed from batch)
            source_tokens: [B, L] - optional user text tokens (ASR transcription)
        
        Returns:
            target_tokens_expanded: [B, L+F] - agent text with PAD insertions
            source_encoded_expanded: [B, L+F, H] - user audio with silence insertions
            expanded_lengths: [B] - actual lengths after expansion
            source_tokens_expanded: [B, L+F] - user text with PAD insertions (only if source_tokens provided)
        """
        B, L = target_tokens.shape
        H = source_encoded.shape[2]
        device = target_tokens.device
        dtype = source_encoded.dtype
        
        expanded_target_list = []
        expanded_source_list = []
        expanded_source_tokens_list = [] if source_tokens is not None else None
        
        # IMPORTANT: In distributed training, all ranks must call perception.forward() the SAME NUMBER OF TIMES
        # to avoid NCCL deadlocks. We need to find the maximum number of insertions across all ranks
        # and ensure every rank participates in that many perception calls.
        
        # Count total insertions per rank
        total_insertions_this_rank = sum(len(pos_list) for pos_list in insertion_positions)
        
        if torch.distributed.is_initialized():
            # Find the maximum number of insertions across all ranks
            max_insertions_tensor = torch.tensor([total_insertions_this_rank], device=device, dtype=torch.int32)
            torch.distributed.all_reduce(max_insertions_tensor, op=torch.distributed.ReduceOp.MAX)
            max_insertions_across_ranks = max_insertions_tensor.item()
            
            # Calculate how many dummy calls this rank needs to make
            dummy_calls_needed = max_insertions_across_ranks - total_insertions_this_rank
            # ALL ranks must participate in collective operations the same number of times.
            if dummy_calls_needed > 0:
                rank = torch.distributed.get_rank()
                logging.debug(f"[Rank {rank}] Making {dummy_calls_needed} dummy silence calls for synchronization (has {total_insertions_this_rank}, max is {max_insertions_across_ranks})")
                # Make dummy calls to participate in collective ops
                for _ in range(dummy_calls_needed):
                    dummy_silence = self._get_silence_embeddings(1, subsampling_factor)
        
        for b in range(B):
            # Work with individual sequences
            tokens = target_tokens[b]  # [L]
            encoded = source_encoded[b]  # [L, H]
            src_tokens = source_tokens[b] if source_tokens is not None else None  # [L]
            
            # Apply insertions sequentially (already sorted by position in _build_function_calling_channel)
            offset = 0  # Track cumulative shift
            for insert_pos, insert_length in insertion_positions[b]:
                adjusted_pos = insert_pos + offset
                
                # Insert PAD tokens in agent text channel (silence during function calling)
                pad_tokens = torch.full((insert_length,), self.text_pad_id, device=device, dtype=tokens.dtype)
                tokens = torch.cat([tokens[:adjusted_pos], pad_tokens, tokens[adjusted_pos:]], dim=0)
                
                # Insert proper silence embeddings in user audio channel (encoded from actual silence audio)
                # Device and dtype are auto-detected from perception module
                silence = self._get_silence_embeddings(insert_length, subsampling_factor)
                encoded = torch.cat([encoded[:adjusted_pos], silence, encoded[adjusted_pos:]], dim=0)
                
                # Insert PAD tokens in user text channel if present (ASR transcription)
                if src_tokens is not None:
                    src_tokens = torch.cat([src_tokens[:adjusted_pos], pad_tokens, src_tokens[adjusted_pos:]], dim=0)
                
                offset += insert_length
            
            expanded_target_list.append(tokens)
            expanded_source_list.append(encoded)
            if src_tokens is not None:
                expanded_source_tokens_list.append(src_tokens)
        
        # Pad to maximum expanded length across batch
        max_expanded_length = max(len(tokens) for tokens in expanded_target_list)
        target_tokens_expanded = torch.full((B, max_expanded_length), self.text_pad_id, dtype=torch.long, device=device)
        source_encoded_expanded = torch.zeros((B, max_expanded_length, H), device=device, dtype=dtype)
        source_tokens_expanded = torch.full((B, max_expanded_length), self.text_pad_id, dtype=torch.long, device=device) if source_tokens is not None else None
        
        # Track actual lengths for each batch item (before padding to max)
        expanded_lengths = torch.zeros(B, dtype=torch.long, device=device)
        
        for b in range(B):
            length = len(expanded_target_list[b])
            target_tokens_expanded[b, :length] = expanded_target_list[b]
            source_encoded_expanded[b, :length] = expanded_source_list[b]
            if source_tokens_expanded is not None:
                source_tokens_expanded[b, :length] = expanded_source_tokens_list[b]
            expanded_lengths[b] = length
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[FC Model] Expanded channels: {L} → {max_expanded_length} (+{max_expanded_length - L})")
            
            # Detailed verification for first batch item
            logging.info(f"[FC Model] ============ CHANNEL EXPANSION VERIFICATION (Batch 0) ============")
            logging.info(f"[FC Model] Insertion positions: {insertion_positions[0]}")
            
            # Show agent text tokens at insertion positions
            expanded_tokens = expanded_target_list[0]
            expanded_audio = expanded_source_list[0]
            
            for insert_idx, (pos, length) in enumerate(insertion_positions[0]):
                # Calculate actual position after previous insertions
                actual_pos = pos + sum(l for p, l in insertion_positions[0][:insert_idx])
                
                logging.info(f"[FC Model] Insertion {insert_idx+1}: Original pos={pos}, Actual pos={actual_pos}, Length={length}")
                
                # Show agent text tokens at insertion (should be all PAD)
                agent_tokens_at_insertion = expanded_tokens[actual_pos:actual_pos+length]
                num_pads = (agent_tokens_at_insertion == self.text_pad_id).sum().item()
                logging.info(f"[FC Model]   Agent text at insertion: {num_pads}/{length} are PAD tokens ✓" if num_pads == length else f"[FC Model]   Agent text at insertion: {num_pads}/{length} are PAD tokens ✗")
                
                # Show user-audio channel at insertion.
                # Note: this channel contains encoded silence embeddings, which are non-zero.
                audio_at_insertion = expanded_audio[actual_pos:actual_pos+length]
                audio_norm = torch.norm(audio_at_insertion).item()
                audio_rms = torch.sqrt(torch.mean(audio_at_insertion.float().pow(2))).item()
                logging.info(
                    f"[FC Model]   User audio at insertion: L2 norm={audio_norm:.6f}, "
                    f"RMS={audio_rms:.6f} (encoded silence is expected to be non-zero)"
                )
                
                # Show what comes before and after
                if actual_pos > 0:
                    before_tokens = expanded_tokens[max(0, actual_pos-3):actual_pos]
                    try:
                        before_text = self.tokenizer.ids_to_text(before_tokens.tolist())
                    except:
                        before_text = str(before_tokens.tolist())
                    logging.info(f"[FC Model]   Agent text BEFORE insertion: '{before_text}'")
                
                if actual_pos + length < len(expanded_tokens):
                    after_tokens = expanded_tokens[actual_pos+length:min(len(expanded_tokens), actual_pos+length+3)]
                    try:
                        after_text = self.tokenizer.ids_to_text(after_tokens.tolist())
                    except:
                        after_text = str(after_tokens.tolist())
                    logging.info(f"[FC Model]   Agent text AFTER insertion: '{after_text}'")
            
            logging.info(f"[FC Model] ==================================================================")
        if source_tokens_expanded is not None:
            return target_tokens_expanded, source_encoded_expanded, expanded_lengths, source_tokens_expanded
        else:
            return target_tokens_expanded, source_encoded_expanded, expanded_lengths

    def prepare_inputs(self, batch: dict):

        if self.cfg.get('debug', False):
            import soundfile as sf
            output_dir = "/lustre/fsw/portfolios/llmservice/users/kevinhu/debug"
            os.makedirs(output_dir, exist_ok=True)
            wav_path = os.path.join(output_dir, f"{batch['sample_id'][0]}_clean.wav")
            # Try best to select a valid sampling rate from config or fallback
            sample_rate = self.cfg.get('source_sample_rate', 16000)
            src_audio_np = batch["source_audio"][0].detach().cpu().numpy()
            sf.write(wav_path, src_audio_np, sample_rate)
            print(f"Wrote batch 0 source_audio to {wav_path}")

        # Apply augmentations in order: noise -> room IR -> mic IR -> codec
        # Each augmentation has its own independent condition and flag
        
        # 1. Noise augmentation (controlled by use_old_noise_aug flag)
        if self.cfg.get('use_old_noise_aug', None) and self.training and self._is_noise_augmentation_dataset(batch["formatter"][0]):
            noise_prob = self.cfg.get('old_noise_prob', 0.99)
            noise_min_snr = self.cfg.get('old_noise_min_snr', 20)
            noise_max_snr = self.cfg.get('old_noise_max_snr', 50)
            noise_path = self.cfg.get('old_noise_aug_path', None)
            noise_path_name = "*"
            
            if noise_prob and random.random() < noise_prob and noise_path:
                batch["source_audio"] = self.audio_augmenter.add_noise_to_batch(
                    batch["source_audio"],
                    os.path.join(noise_path, noise_path_name),
                    snr_db=random.randint(noise_min_snr, noise_max_snr),
                    noise_prob_scale_user=self.cfg.get('noise_prob_scale_user', 0.3),
                    noise_prob_scale_user_min_snr=self.cfg.get('noise_prob_scale_user_min_snr', -15),
                    noise_prob_scale_user_max_snr=self.cfg.get('noise_prob_scale_user_max_snr', 24),
                    snr_measure_dur=self.cfg.get('snr_measure_dur', 0.0),
                    noise_resample=self.cfg.get('noise_resample', True),
                    noise_prob_low_pass=self.cfg.get('noise_prob_low_pass', 0.1),
                )
        
        # 2. Room impulse response augmentation
        if self.cfg.get('use_room_ir_aug', None) and self.training and self._is_noise_augmentation_dataset(batch["formatter"][0]):
            roomir_prob = self.cfg.get('roomir_prob', 0.0)
            roomir_path = self.cfg.get('roomir_aug_path', None)
            
            if roomir_prob > 0 and roomir_path and random.random() < roomir_prob:
                batch["source_audio"] = self.audio_augmenter.add_room_ir_to_batch(
                    batch["source_audio"],
                    batch["source_audio_lens"],
                    roomir_path,
                    use_loudness_norm=self.cfg.get('roomir_use_loudness_norm', True),
                )
        
        # 3. Microphone impulse response augmentation
        if self.cfg.get('use_mic_ir_aug', None) and self.training and self._is_noise_augmentation_dataset(batch["formatter"][0]):
            micir_prob = self.cfg.get('micir_prob', 0.0)
            micir_path = self.cfg.get('micir_aug_path', None)
            
            if micir_prob > 0 and micir_path and random.random() < micir_prob:
                batch["source_audio"] = self.audio_augmenter.add_mic_ir_to_batch(
                    batch["source_audio"],
                    batch["source_audio_lens"],
                    micir_path,
                    use_loudness_norm=self.cfg.get('micir_use_loudness_norm', True),
                )
        
        # 4. Codec augmentation
        if self.cfg.get('use_codec_aug', None) and self.training and self._is_noise_augmentation_dataset(batch["formatter"][0]):
            codec_prob = self.cfg.get('codec_prob', 0.0)
            codec_settings = self.cfg.get('codec_settings', None)
            
            if codec_prob > 0 and random.random() < codec_prob:
                # Use custom codec settings if provided, otherwise use defaults
                if codec_settings is None:
                    codec_settings = DEFAULT_CODEC_SETTINGS
                batch["source_audio"] = self.audio_augmenter.add_codec_to_batch(
                    batch["source_audio"],
                    batch["source_audio_lens"],
                    codec_settings,
                )

        if self.cfg.get('debug', False):
            import soundfile as sf
            output_dir = "/lustre/fsw/portfolios/llmservice/users/kevinhu/debug"
            os.makedirs(output_dir, exist_ok=True)
            wav_path = os.path.join(output_dir, f"{batch['sample_id'][0]}.wav")
            sample_rate = self.cfg.get('source_sample_rate', 16000)
            src_audio_np = batch["source_audio"][0].detach().cpu().numpy()
            sf.write(wav_path, src_audio_np, sample_rate)
            print(f"Wrote batch 0 source_audio to {wav_path}")

        # Add backchannel augmentation (only during training, when agent is talking / user is silent)
        if self.cfg.get('backchannel_prob', None) and self.cfg.backchannel_prob > 0:
            if (
                self.training
                and not batch.get("is_minimal_batch", False)  # Skip placeholder minimal batches
                and "source_tokens" in batch  # Ensure source_tokens exists
                and batch["formatter"][0] != 's2s_duplex_overlap_as_s2s_duplex'  # Skip overlap data (already has real backchannels)
                and batch["formatter"][0] != 'nemo_tarred_to_duplex'  # Skip ASR datasets
                and batch["formatter"][0] != 'lhotse_tts_as_repeat_after_me'  # Skip TTS repeat-after-me (synthetic data)
                and random.random() < self.cfg.backchannel_prob
            ):
                batch["source_audio"] = self.add_backchannel_to_batch(
                    batch["source_audio"],
                    batch["target_tokens"],
                    batch["source_tokens"],
                    os.path.join(self.cfg.backchannel_file_path, "*"),
                    snr_db=self.cfg.get('backchannel_snr_db', 15),
                    backchannel_prob_scale=self.cfg.get('backchannel_prob_scale', 0.5),
                    debug=self.cfg.get('backchannel_debug', False),
                    debug_save_path=self.cfg.get('backchannel_debug_path', None),
                    debug_max_files=self.cfg.get('backchannel_debug_max_files', 16),
                    target_audio=batch.get("target_audio", None),
                )

        
        if self.cfg.get("asr_log", False):
            logging.info(f"User audio dtype: {batch['source_audio'].dtype}, device: {batch['source_audio'].device}, shape: {batch['source_audio'].shape}")
        
        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        target_tokens = batch["target_tokens"]
        batch_size = source_encoded.shape[0]

        # Distributed safety: keep prompt branch identical across ranks.
        # Some ranks may miss prompt keys (e.g., non-FC/minimal mixes); normalize to empty prompt tensors.
        if "prompt_tokens" not in batch or batch["prompt_tokens"] is None:
            batch["prompt_tokens"] = torch.empty((batch_size, 0), dtype=torch.long, device=target_tokens.device)
        if "prompt_token_lens" not in batch or batch["prompt_token_lens"] is None:
            batch["prompt_token_lens"] = torch.zeros((batch_size,), dtype=torch.long, device=target_tokens.device)

        if batch["prompt_tokens"] is not None:
            prompt_embedded = self.embed_tokens(batch["prompt_tokens"])
            B, max_prompt_len, H = prompt_embedded.shape
            T_src = source_encoded.shape[1]
            T_tgt = target_tokens.shape[1]
            
            if self.cfg.get("fc_log", False):
                logging.info(f"[Training] System prompt detected: batch_size={B}, max_prompt_len={max_prompt_len}")
                for i in range(min(B, 2)):  # Show first 2 samples
                    prompt_len = batch["prompt_token_lens"][i].item()
                    if prompt_len > 0:
                        prompt_tokens_sample = batch["prompt_tokens"][i, :prompt_len].tolist()
                        prompt_text = self.tokenizer.ids_to_text(prompt_tokens_sample)
                        logging.info(f"[Training] System Prompt (Sample {i}, {prompt_len} tokens):")
                        logging.info(f"[Training]   Full text: {prompt_text}")
                        if len(prompt_text) > 200:
                            logging.info(f"[Training]   (truncated preview): {prompt_text[:200]}...")

            new_source_encoded = torch.zeros(B, max_prompt_len + T_src, H,
                                             dtype=source_encoded.dtype, device=source_encoded.device)
            new_target_tokens = torch.full((B, max_prompt_len + T_tgt), self.text_pad_id, dtype=target_tokens.dtype, device=target_tokens.device)
            # If source_tokens are present (used by ASR head for user text prediction),
            # prepend PADs to align ASR labels with the prompt span as well.
            if "source_tokens" in batch:
                source_tokens = batch["source_tokens"]
                T_src_tok = source_tokens.shape[1]
                new_source_tokens = torch.full(
                    (B, max_prompt_len + T_src_tok),
                    self.text_pad_id,
                    dtype=source_tokens.dtype,
                    device=source_tokens.device,
                )

            # For each item, insert prompt and original data at correct offsets
            for i, prompt_len in enumerate(batch["prompt_token_lens"]):
                prompt_len = prompt_len.item()

                if prompt_len > 0:
                    new_source_encoded[i, :prompt_len, :] = prompt_embedded[i, :prompt_len, :]

                src_len = source_encoded_lens[i].item()
                new_source_encoded[i, prompt_len:prompt_len + src_len, :] = source_encoded[i, :src_len, :]

                tgt_len = batch["target_token_lens"][i].item()
                new_target_tokens[i, prompt_len:prompt_len + tgt_len] = target_tokens[i, :tgt_len]

                source_encoded_lens[i] = prompt_len + src_len
                batch["target_token_lens"][i] = prompt_len + tgt_len
                
                # If source_tokens exist, copy them after the prompt and update lengths
                if "source_tokens" in batch:
                    src_len = batch["source_token_lens"][i].item()
                    new_source_tokens[i, prompt_len:prompt_len + src_len] = source_tokens[i, :src_len]
                    batch["source_token_lens"][i] = prompt_len + src_len
            
            if self.cfg.get("debug_fc", False):
                logging.info(f"[Training] After prompt prepending: source_encoded shape={new_source_encoded.shape}, "
                           f"target_tokens shape={new_target_tokens.shape}")
                # Verify PAD region
                for i in range(min(B, 2)):
                    prompt_len = batch["prompt_token_lens"][i].item()
                    if prompt_len > 0:
                        pad_count = (new_target_tokens[i, :prompt_len] == self.text_pad_id).sum().item()
                        logging.info(f"[Training] Sample {i}: prompt_len={prompt_len}, PAD tokens in prompt region={pad_count}")
            
            source_encoded = new_source_encoded
            target_tokens = new_target_tokens
            if "source_tokens" in batch:
                batch["source_tokens"] = new_source_tokens

        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat([
                target_tokens,
                (torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id).to(
                    torch.long),
            ], dim=-1)
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]


        # Align source_tokens (user text) with source_encoded (user audio) if present
        # Semantically correct: user text aligns with user audio (both are source/user content)
        if "source_tokens" in batch and self.predict_user_text:
            source_tokens = batch["source_tokens"]
            if (diff := source_tokens.shape[1] - source_encoded.shape[1]) < 0:
                source_tokens = torch.cat([
                    source_tokens,
                    (torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id).to(
                        torch.long),
                ], dim=-1)
                batch["source_token_lens"] = batch["source_token_lens"] + abs(diff)
            elif diff > 0:
                source_tokens = source_tokens[:, : source_encoded.shape[1]]
                batch["source_token_lens"] = batch["source_token_lens"] - diff
            batch["source_tokens"] = source_tokens

        # Optional: convert pad tokens to sil tokens
        sil_id = None
        if self.cfg.get("use_sil_token", False):
            target_tokens, sil_id = self._convert_pad_to_sil(target_tokens)

        # Build function calling channel and expand sequences BEFORE prepare_labels
        # IMPORTANT: Using insertion approach - sequence length will expand from L to L+F
        # Also enter this block for minimal/dropped batch so this rank participates in the same
        # collectives (all_reduce in _expand_channels_with_insertions) and avoids NCCL timeout.
        function_channel = None
        function_channel_loss_mask = None
        insertion_positions = None
        has_fc = "function_calls" in batch and batch["function_calls"] is not None
        is_minimal_batch = batch.get("is_minimal_batch", False)
        is_minimal_batch_fc = batch.get("minimal_batch_fc", False)  # True iff minimal batch was dropped due to FC (e.g. over max_fc_total_tokens)
        is_minimal_batch_non_fc = is_minimal_batch and not is_minimal_batch_fc  # e.g. all-cuts-filtered, force-align failed
        
        # Enter for FC batches, any minimal batch, or (when distributed) normal non-FC so all ranks participate
        # in the same collectives (avoid NCCL timeout). With mixed batch types across ranks (e.g. one rank normal
        # non-FC, another FC or minimal), every rank must run the same all_reduce/dummy path; when this rank has
        # normal non-FC we go through the expansion path with empty insertion_positions (sync only, tensors unchanged).
        if has_fc or is_minimal_batch or (torch.distributed.is_initialized() and not has_fc):
            # Compute subsampling factor (needed for sync and for expansion)
            audio_lens = batch["source_audio_lens"]
            subsampling_factors = []
            for i in range(min(len(audio_lens), len(source_encoded_lens))):
                audio_len = audio_lens[i].item()
                encoded_len = source_encoded_lens[i].item()
                if encoded_len > 0:
                    subsampling_factors.append(audio_len / encoded_len)
            subsampling_factor = sum(subsampling_factors) / len(subsampling_factors) if subsampling_factors else 8.0

            # Minimal batches should take sync-only path on all setups.
            # This avoids unnecessary FC/data-dependent branching while still participating
            # in the same collectives via _sync_fc_insertions_collective.
            if is_minimal_batch:
                if is_minimal_batch_fc:
                    fc_drop = batch.get("fc_drop_info")
                    if fc_drop:
                        logging.info(
                            "[FC Model] is_minimal_batch_fc=True | cut_id=%s total_prompt_tokens=%s max_fc_total_tokens=%s reason=%s; using sync-only path.",
                            fc_drop.get("cut_id", "?"),
                            fc_drop.get("total_prompt_tokens", "?"),
                            fc_drop.get("max_fc_total_tokens", fc_drop.get("max_system_fc_tokens", "?")),
                            fc_drop.get("reason", "?"),
                        )
                    else:
                        logging.info("[FC Model] is_minimal_batch_fc=True (dropped due to FC); using sync-only path.")
                elif is_minimal_batch_non_fc:
                    logging.info("[FC Model] is_minimal_batch_non_fc=True (e.g. all-cuts-filtered, force-align failed); using sync-only path.")
                self._sync_fc_insertions_collective(0, subsampling_factor)
                # Keep schema consistent with FC path: build a neutral function channel tensor
                # (all PAD, no loss) so downstream/debug logic does not see None.
                B, T = target_tokens.shape
                function_channel = torch.full((B, T), self.text_pad_id, dtype=torch.long, device=target_tokens.device)
                function_channel_loss_mask = torch.zeros((B, T), dtype=torch.bool, device=target_tokens.device)
                insertion_positions = [[] for _ in range(B)]
            else:
                if has_fc and self.cfg.get("debug_fc", False):
                    logging.info(f"[FC Model] ============================================================")
                    logging.info(f"[FC Model] FUNCTION CALLING INSERTION - BEFORE EXPANSION")
                    logging.info(f"[FC Model] ============================================================")
                    logging.info(f"[FC Model] Original target_tokens shape: {target_tokens.shape}")
                    logging.info(f"[FC Model] Original source_encoded shape: {source_encoded.shape}")
                    
                    sample_len = min(50, target_tokens.shape[1])
                    sample_tokens = target_tokens[0, :sample_len]
                    try:
                        sample_text = self.tokenizer.ids_to_text(sample_tokens.tolist())
                        logging.info(f"[FC Model] Original agent text (first 50 tokens): '{sample_text}'")
                    except:
                        logging.info(f"[FC Model] Original agent text (first 50 tokens): {sample_tokens.tolist()}")
                
                if has_fc:
                    function_channel, function_channel_loss_mask, insertion_positions = self._build_function_calling_channel(
                        batch, target_tokens.shape[1]
                    )
                else:
                    B = target_tokens.shape[0]
                    T = target_tokens.shape[1]
                    # Non-FC batch: supervise function channel to stay PAD.
                    function_channel = torch.full((B, T), self.text_pad_id, dtype=torch.long, device=target_tokens.device)
                    function_channel_loss_mask = torch.ones((B, T), dtype=torch.bool, device=target_tokens.device)
                    insertion_positions = [[] for _ in range(B)]
                
                if self.cfg.get("debug_fc", False):
                    logging.info(f"[FC Model] Computed subsampling factor: {subsampling_factor:.2f} (avg of {len(subsampling_factors)} samples)")
                
                source_tokens_to_expand = batch.get("source_tokens") if self.predict_user_text else None
                if source_tokens_to_expand is not None:
                    target_tokens, source_encoded, expanded_lengths, source_tokens_expanded = self._expand_channels_with_insertions(
                        target_tokens, source_encoded, insertion_positions, subsampling_factor, source_tokens_to_expand
                    )
                    batch["source_tokens"] = source_tokens_expanded
                    if self.cfg.get("debug_fc", False):
                        logging.info(f"[FC Model] Expanded source_tokens (ASR): {source_tokens_expanded.shape}")
                else:
                    target_tokens, source_encoded, expanded_lengths = self._expand_channels_with_insertions(
                        target_tokens, source_encoded, insertion_positions, subsampling_factor
                    )
                
                batch["target_token_lens"] = expanded_lengths
                source_encoded_lens = expanded_lengths.clone()
            
            if has_fc and self.cfg.get("debug_fc", False):
                logging.info(f"[FC Model] ============================================================")
                logging.info(f"[FC Model] FUNCTION CALLING INSERTION - AFTER EXPANSION")
                logging.info(f"[FC Model] ============================================================")
                logging.info(f"[FC Model] Expanded target_tokens: {target_tokens.shape}")
                logging.info(f"[FC Model] Expanded source_encoded: {source_encoded.shape}")
                if function_channel is None:
                    logging.info("[FC Model] Function channel: None (sync-only/minimal batch path)")
                else:
                    logging.info(f"[FC Model] Function channel: {function_channel.shape}")
                    logging.info(
                        f"[FC Model] Expansion result length: target={target_tokens.shape[1]}, function={function_channel.shape[1]}"
                    )
                
                # Show sample of expanded agent text to verify PAD insertions
                sample_len = min(50, target_tokens.shape[1])
                sample_tokens = target_tokens[0, :sample_len]
                try:
                    sample_text = self.tokenizer.ids_to_text(sample_tokens.tolist())
                    logging.info(f"[FC Model] Expanded agent text (first 50 tokens): '{sample_text}'")
                except:
                    logging.info(f"[FC Model] Expanded agent text (first 50 tokens): {sample_tokens.tolist()}")
                
                # Count PAD tokens in expanded sequence
                num_pads = (target_tokens[0] == self.text_pad_id).sum().item()
                logging.info(f"[FC Model] Number of PAD tokens in expanded agent text: {num_pads}")
                
                # Verify channel lengths only when function channel exists (non-minimal FC path).
                if function_channel is not None:
                    assert target_tokens.shape[1] == source_encoded.shape[1] == function_channel.shape[1], \
                        f"Channel length mismatch: target={target_tokens.shape[1]}, source={source_encoded.shape[1]}, function={function_channel.shape[1]}"
                    logging.info(f"[FC Model] ✓ All channels have same length: {target_tokens.shape[1]}")
                logging.info(f"[FC Model] ============================================================")
        else:
            if self.cfg.get("debug_fc", False):
                logging.debug(f"[FC Model] No function calling data in this batch")

        # Single-device/non-distributed non-FC path may skip the FC/sync branch above.
        # Still train function channel to PAD on those batches.
        if function_channel is None and not has_fc and not is_minimal_batch:
            B, T = target_tokens.shape
            function_channel = torch.full((B, T), self.text_pad_id, dtype=torch.long, device=target_tokens.device)
            function_channel_loss_mask = torch.ones((B, T), dtype=torch.bool, device=target_tokens.device)
            if insertion_positions is None:
                insertion_positions = [[] for _ in range(B)]
        
        # Now call prepare_labels with EXPANDED sequences
        if function_channel is not None and self.cfg.get("debug_fc", False):
            chunk_size = int(self.cfg.get("debug_fc", 200))
            for b in range(function_channel.shape[0]):
                self._log_long_list(
                    f"[FC Model] FULL function_channel[{b}] len={function_channel.shape[1]}",
                    function_channel[b].tolist(),
                    chunk_size,
                )
                self._log_long_list(
                    f"[FC Model] FULL function_loss_mask[{b}] len={function_channel_loss_mask.shape[1]}",
                    function_channel_loss_mask[b].int().tolist(),
                    chunk_size,
                )
                logging.info(f"[FC Model] Insertion positions (original space) batch {b}: {insertion_positions[b]}")

        inputs = prepare_labels(
            batch=batch,
            target_tokens=target_tokens,
            source_encoded=source_encoded,
            cfg=self.cfg,
            predict_user_text=self.predict_user_text,
            user_bos_id=self.user_bos_id,
            user_eos_id=self.user_eos_id,
            text_pad_id=self.text_pad_id,
            text_bos_id=self.text_bos_id,
            text_eos_id=self.text_eos_id,
            advance_text_channel_by=self.advance_text_channel_by,
            use_tp=self._use_tp,
            device_mesh=self.device_mesh if self._use_tp else None,
            function_channel=function_channel,
            function_channel_loss_mask=function_channel_loss_mask,
            prompt_token_lens=batch.get("prompt_token_lens", None),
        )

        source_encoded = inputs["source_encoded"]
        text_inputs = inputs["text_inputs"]
        text_labels = inputs["text_labels"]
        if self.predict_user_text:
            asr_inputs = inputs["asr_inputs"]
            asr_labels = inputs["asr_labels"]
        
        # Extract function calling channel if present 
        function_inputs = None
        function_labels = None
        function_loss_mask = None
        if function_channel is not None:
            function_inputs = inputs["function_inputs"]
            function_labels = inputs["function_labels"]
            function_loss_mask = inputs["function_loss_mask"]
            
            if self.cfg.get("debug_fc", False):
                logging.info(f"[FC Model] ============================================================")
                logging.info(f"[FC Model] AFTER TEMPORAL SHIFTS (advance_text_channel_by={self.advance_text_channel_by}, "
                           f"delay_text_channel_by={self.cfg.get('delay_text_channel_by', 0)})")
                logging.info(f"[FC Model] text_inputs shape: {text_inputs.shape}")
                logging.info(f"[FC Model] function_inputs shape: {function_inputs.shape}")
                logging.info(f"[FC Model] source_encoded shape (after [:, :-1]): {source_encoded[:, :-1].shape}")
                
                # Show first 50 tokens from each channel to verify alignment
                sample_len = min(50, text_inputs.shape[1])
                
                # Agent text channel
                text_sample = text_inputs[0, :sample_len]
                try:
                    text_decoded = self.tokenizer.ids_to_text(text_sample.tolist())
                    logging.info(f"[FC Model] Agent text (first {sample_len}): '{text_decoded}'")
                except:
                    logging.info(f"[FC Model] Agent text (first {sample_len}): {text_sample.tolist()}")
                
                # Function calling channel
                func_sample = function_inputs[0, :sample_len]
                non_pad_func = func_sample[func_sample != self.text_pad_id]
                if len(non_pad_func) > 0:
                    try:
                        func_decoded = self.tokenizer.ids_to_text(func_sample.tolist())
                        logging.info(f"[FC Model] Function channel (first {sample_len}): '{func_decoded}'")
                        logging.info(f"[FC Model] Function channel non-PAD tokens: {non_pad_func.tolist()[:20]}")
                    except:
                        logging.info(f"[FC Model] Function channel (first {sample_len}): {func_sample.tolist()}")
                else:
                    logging.info(f"[FC Model] Function channel (first {sample_len}): all PAD tokens")
                
                # Check alignment at positions where function tokens exist
                func_positions = (function_inputs[0] != self.text_pad_id).nonzero(as_tuple=True)[0]
                if len(func_positions) > 0:
                    logging.info(f"[FC Model] Function tokens found at {len(func_positions)} positions")
                    logging.info(f"[FC Model] First 10 function token positions: {func_positions[:10].tolist()}")
                    
                    # Verify that agent text has PAD at these positions
                    text_at_func_pos = text_inputs[0, func_positions[:10]]
                    num_text_pads = (text_at_func_pos == self.text_pad_id).sum().item()
                    logging.info(f"[FC Model] At first 10 function positions, agent text has {num_text_pads}/10 PAD tokens (should be high)")
                
                # Show loss mask statistics
                num_loss_on = function_loss_mask[0].sum().item()
                total_positions = function_loss_mask[0].numel()
                logging.info(f"[FC Model] Function loss mask: {num_loss_on}/{total_positions} positions enabled")
                logging.info(f"[FC Model] ============================================================")
        
        # Build input embeddings from all channels
        input_embeds = self.embed_tokens(text_inputs) * self.cfg.get("duplex_text_channel_weight", 1.0)
        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))
        
        if self.predict_user_text:
            # ASR inputs are text tokens IDs for the user asr channel
            asr_inputs_embeds = self.embed_asr_tokens(asr_inputs) * self.cfg.get("duplex_asr_text_weight", 1.0)
            input_embeds.add_(asr_inputs_embeds)
        
        # Add function calling channel embedding if present
        if function_inputs is not None:
            # Verify shape consistency after expansion and temporal shifts
            if function_inputs.shape != text_inputs.shape:
                raise ValueError(
                    f"Shape mismatch after insertion and temporal shifts: function_inputs {function_inputs.shape} "
                    f"vs text_inputs {text_inputs.shape}. This indicates a bug in expansion/shift logic."
                )
            
            if self.cfg.get("debug_fc", False):
                logging.info(f"[FC Model] ============================================================")
                logging.info(f"[FC Model] FINAL INPUT EMBEDDINGS - ALL CHANNELS COMBINED")
                logging.info(f"[FC Model] ============================================================")
                logging.info(f"[FC Model] Adding function calling channel to input_embeds")
                logging.info(f"[FC Model] text_inputs shape: {text_inputs.shape}")
                logging.info(f"[FC Model] function_inputs shape: {function_inputs.shape}")
                logging.info(f"[FC Model] ✓ Shapes match - channels are properly aligned")
                
                # Show where each channel contributes to the input
                # Sample first 30 positions
                sample_len = min(30, text_inputs.shape[1])
                logging.info(f"[FC Model] Channel contribution (first {sample_len} positions, batch 0):")
                logging.info(f"[FC Model] {'Pos':<5} {'Agent':<15} {'Function':<15} {'Combined'}")
                logging.info(f"[FC Model] {'-'*5} {'-'*15} {'-'*15} {'-'*15}")
                
                for pos in range(sample_len):
                    text_token = text_inputs[0, pos].item()
                    func_token = function_inputs[0, pos].item()
                    
                    # Decode tokens
                    try:
                        text_str = self.tokenizer.ids_to_text([text_token])[:10]
                    except:
                        text_str = f"<{text_token}>"
                    
                    try:
                        func_str = self.tokenizer.ids_to_text([func_token])[:10]
                    except:
                        func_str = f"<{func_token}>"
                    
                    # Determine what's active
                    text_active = "✓" if text_token != self.text_pad_id else " "
                    func_active = "✓" if func_token != self.text_pad_id else " "
                    
                    combined = []
                    if text_token != self.text_pad_id:
                        combined.append("Agent")
                    if func_token != self.text_pad_id:
                        combined.append("Function")
                    if not combined:
                        combined.append("Silence")
                    
                    logging.info(f"[FC Model] {pos:<5} {text_active} {text_str:<13} {func_active} {func_str:<13} {'+'.join(combined)}")
                
                logging.info(f"[FC Model] ============================================================")
            
            # Use shared embedding for function channel (same vocab as text)
            function_channel_weight = self.cfg.get("duplex_function_channel_weight", 1.0)
            function_inputs_embeds = self.embed_tokens(function_inputs) * function_channel_weight
            input_embeds.add_(function_inputs_embeds)
        else:
            function_labels = None
            function_channel_loss_mask = None
        
        seq_mask = torch.ones_like(text_labels.unsqueeze(-1), device=self.device, dtype=torch.bool)
        
        if self.cfg.get("mask_sequence_loss", True):
            for i in range(batch["target_token_lens"].size(0)):
                # If function calling is present, batch["target_token_lens"] contains expanded_lengths
                # Need to subtract 1 to account for temporal shift in prepare_labels ([:, :-1])
                if function_channel is not None:
                    speech_end_idx = batch["target_token_lens"][i] - 1
                else:
                    speech_end_idx = batch["target_token_lens"][i]
                seq_mask[i, speech_end_idx:, :] = 0
        
        # Explicitly mask system prompt region to prevent loss computation
        # This ensures no loss is computed on prompt regardless of pad_weight setting
        # Account for temporal shift: prepare_labels does [:, 1:] which removes first token
        if "prompt_token_lens" in batch:
            for i, prompt_len in enumerate(batch["prompt_token_lens"]):
                prompt_len_val = prompt_len.item()
                if prompt_len_val > 0:
                    # Subtract 1 to account for temporal shift
                    shifted_prompt_len = prompt_len_val - 1
                    seq_mask[i, :shifted_prompt_len, :] = 0
                    if self.cfg.get("debug_fc", False) and i == 0:
                        logging.info(f"[Training] Masked system prompt region [0:{shifted_prompt_len}] from loss computation (original: {prompt_len_val})")

        loss_scale = seq_mask.clone().float()
        asr_loss_scale = seq_mask.clone().float()
        if self.cfg.get("token_loss_weight"):
            token_weights = self.cfg.token_loss_weight
            pad_weight = token_weights.get("pad", 1.0)
            bos_weight = token_weights.get("bos", 1.0)
            eos_weight = token_weights.get("eos", 1.0)
            text_weight = token_weights.get("text", 1.0)
            sil_weight = token_weights.get("sil", 1.0)

            if sil_id is not None:
                loss_scale = torch.where(
                    text_labels.unsqueeze(-1) == self.text_pad_id, pad_weight,
                    torch.where(
                        text_labels.unsqueeze(-1) == self.text_bos_id, bos_weight,
                        torch.where(
                            text_labels.unsqueeze(-1) == self.text_eos_id, eos_weight,
                            torch.where(
                                text_labels.unsqueeze(-1) == sil_id, sil_weight,
                                text_weight
                            )
                        )
                    )
                )
            else:
                loss_scale = torch.where(
                    text_labels.unsqueeze(-1) == self.text_pad_id, pad_weight,
                    torch.where(
                        text_labels.unsqueeze(-1) == self.text_bos_id, bos_weight,
                        torch.where(
                            text_labels.unsqueeze(-1) == self.text_eos_id, eos_weight,
                            text_weight
                        )
                    )
                )
            loss_scale = self._maybe_zero_out_scale_for_asr(loss_scale, text_labels, batch)
            
            # Re-apply seq_mask to preserve prompt masking and sequence boundaries
            loss_scale = loss_scale * seq_mask
            
            if self.predict_user_text:
                asr_loss_scale = torch.where(
                    asr_labels.unsqueeze(-1) == self.text_pad_id, pad_weight,
                    torch.where(
                        asr_labels.unsqueeze(-1) == self.user_bos_id, bos_weight,
                        torch.where(
                            asr_labels.unsqueeze(-1) == self.user_eos_id, eos_weight,
                            text_weight
                        )
                    )
                )
                # Re-apply seq_mask for ASR loss scale too
                asr_loss_scale = asr_loss_scale * seq_mask

        # Function calling loss scale
        function_loss_scale = None
        if function_loss_mask is not None:
            # Use the already-shifted mask from prepare_labels
            function_loss_scale = function_loss_mask.unsqueeze(-1).float()
            # Apply sequence mask
            function_loss_scale = function_loss_scale * seq_mask
            
            if self.cfg.get("debug_fc", False):
                num_loss_positions = (function_loss_scale > 0).sum().item()
                num_no_loss_positions = (function_loss_scale == 0).sum().item() - (seq_mask == 0).sum().item()
                logging.info(f"[FC Model] Function loss scale: loss_on={num_loss_positions}, loss_off={num_no_loss_positions}")

        ans = {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": source_encoded_lens - 1,
            "text_labels": text_labels,
            "loss_scale": loss_scale,
            "seq_mask": seq_mask,
        }
        if self.predict_user_text:
            ans["asr_labels"] = asr_labels
            ans["asr_loss_scale"] = asr_loss_scale
        if function_labels is not None:
            ans["function_labels"] = function_labels
            ans["function_loss_scale"] = function_loss_scale
        return ans

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        res = {"learning_rate": torch.as_tensor(
            self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)}

        if batch["audio_data"] is not None:
            inputs = self.prepare_inputs(batch["audio_data"])
            is_minimal_batch = batch["audio_data"].get("is_minimal_batch", False)
            
            forward_outputs = self(inputs["input_embeds"])

            num_frames = inputs["input_lens"].sum()

            with loss_parallel():
                text_logits = forward_outputs["text_logits"]
                if self.predict_user_text:
                    asr_logits = forward_outputs["asr_logits"]

                if self.cfg.get("mask_sequence_loss", True):
                    text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

                text_loss = (torch.nn.functional.cross_entropy(
                                        text_logits.flatten(0, 1),
                                        inputs["text_labels"].flatten(0, 1),
                                        reduction="none",
                                    )
                                    * inputs["loss_scale"][:, :, 0].flatten(0, 1)
                            ).sum(-1) / num_frames

                if self.predict_user_text:
                    asr_loss = (
                        torch.nn.functional.cross_entropy(
                            asr_logits.flatten(0, 1),
                            inputs["asr_labels"].flatten(0, 1),
                            reduction="none",
                        )
                        * inputs["asr_loss_scale"][:, :, 0].flatten(0, 1)
                    ).sum(-1) / num_frames
                    if self.cfg.get("debug", False):
                        batch_idx = 0
                        stacked = torch.stack([inputs["asr_labels"][batch_idx], inputs["asr_loss_scale"][batch_idx, :, 0].int()], dim=1)
                        stacked = stacked * (stacked != self.text_pad_id)
                        print("Stacked asr_labels and asr_loss_scale for first batch (up to 500 steps):")
                        print(stacked[:500].int())
                    if self.cfg.get("asr_log", False):
                        print(f'asr_loss: {asr_loss}')

                # Function calling loss (use separate head if available)
                function_loss = torch.tensor(0.0, device=text_loss.device)
                # Always read function logits so function_head is part of the graph on all ranks.
                # This avoids per-rank autograd divergence when some ranks have FC labels and others do not.
                function_logits = forward_outputs["function_logits"]
                if "function_labels" in inputs and inputs["function_labels"] is not None:
                    # Get special token IDs
                    sotc_id, eotc_id, eotr_id = self._get_function_call_special_tokens()
                    
                    if self.cfg.get("mask_sequence_loss", True):
                        function_logits = function_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)
                    
                    # Log function channel predictions
                    with torch.no_grad():
                        function_predicted_tokens = torch.argmax(function_logits, dim=-1)  # (B, T)
                        function_target_tokens = inputs["function_labels"]  # (B, T)
                        
                        # Log first sample's function channel
                        pred_tokens = function_predicted_tokens[0].cpu().tolist()
                        target_tokens = function_target_tokens[0].cpu().tolist()
                        
                        # Only log non-PAD tokens
                        pred_non_pad = [t for t in pred_tokens if t != self.text_pad_id]
                        target_non_pad = [t for t in target_tokens if t != self.text_pad_id]
                        
                        if len(pred_non_pad) > 0 or len(target_non_pad) > 0:
                            pred_text = self.tokenizer.ids_to_text(pred_non_pad[:100]) if len(pred_non_pad) > 0 else ""
                            target_text = self.tokenizer.ids_to_text(target_non_pad[:100]) if len(target_non_pad) > 0 else ""
                            
                            logging.info(f"[FC Channel] Predicted ({len(pred_non_pad)} tokens): {pred_text[:150]}")
                            logging.info(f"[FC Channel] Target ({len(target_non_pad)} tokens): {target_text[:150]}")
                    
                    # Start with base function loss scale (now includes PAD=True, responses=False)
                    function_loss_scale = inputs["function_loss_scale"][:, :, 0].float()  # [B, T]
                    
                    # Apply fine-grained token-specific weights
                    if self.cfg.get("function_token_loss_weight"):
                        func_weights = self.cfg.function_token_loss_weight
                        pad_weight = func_weights.get("pad", 0.1)        # PAD tokens
                        sotc_weight = func_weights.get("sotc", 10.0)    # Start of Tool Call marker
                        eotc_weight = func_weights.get("eotc", 10.0)    # End of Tool Call marker
                        eotr_weight = func_weights.get("eotr", 1.0)     # End of Tool Response marker (won't have loss anyway)
                        call_weight = func_weights.get("call", 5.0)     # Actual call content tokens
                        
                        function_labels_2d = inputs["function_labels"]  # [B, T]
                        
                        # Build weight mask - start with call_weight as default
                        weight_mask = torch.full_like(function_labels_2d, call_weight, dtype=torch.float32)
                        
                        # Override with specific weights for each token type
                        weight_mask = torch.where(function_labels_2d == self.text_pad_id, pad_weight, weight_mask)
                        weight_mask = torch.where(function_labels_2d == sotc_id, sotc_weight, weight_mask)
                        weight_mask = torch.where(function_labels_2d == eotc_id, eotc_weight, weight_mask)
                        weight_mask = torch.where(function_labels_2d == eotr_id, eotr_weight, weight_mask)
                        
                        # Apply weights
                        function_loss_scale = function_loss_scale * weight_mask
                    
                    # Apply sequence mask
                    function_loss_scale = function_loss_scale.unsqueeze(-1) * inputs["seq_mask"]
                    
                    # Calculate loss (normalized by num_frames, consistent with text loss)
                    function_loss = (
                        torch.nn.functional.cross_entropy(
                            function_logits.flatten(0, 1),
                            inputs["function_labels"].flatten(0, 1),
                            reduction="none",
                        )
                        * function_loss_scale[:, :, 0].flatten(0, 1)
                    ).sum(-1) / num_frames
                    
                    if self.cfg.get("debug_fc", False):
                        # Log detailed loss statistics
                        num_loss_tokens = (function_loss_scale[:, :, 0] > 0).sum().item()
                        num_pad = ((inputs["function_labels"] == self.text_pad_id) & (function_loss_scale[:, :, 0] > 0)).sum().item()
                        num_sotc = ((inputs["function_labels"] == sotc_id) & (function_loss_scale[:, :, 0] > 0)).sum().item()
                        num_eotc = ((inputs["function_labels"] == eotc_id) & (function_loss_scale[:, :, 0] > 0)).sum().item()
                        num_eotr = ((inputs["function_labels"] == eotr_id) & (function_loss_scale[:, :, 0] > 0)).sum().item()
                        num_call = num_loss_tokens - num_pad - num_sotc - num_eotc - num_eotr
                        
                        logging.info(f"[FC Model Training] Function loss: {function_loss.item():.6f}")
                        logging.info(f"[FC Model Training] Loss breakdown: PAD={num_pad}, SOTC={num_sotc}, EOTC={num_eotc}, EOTR={num_eotr}, CALL={num_call}")
                        
                        # Log function calling predictions vs labels
                        with torch.no_grad():
                            function_predicted_tokens = torch.argmax(function_logits, dim=-1)  # (B, T)
                            function_target_tokens = inputs["function_labels"]  # (B, T)
                            logging.info(f"[FC Model Training] Function calling predictions:")
                            # Show first sample's predictions vs labels
                            fc_pred_tokens_sample = function_predicted_tokens[0, :min(100, function_predicted_tokens.shape[1])].cpu().tolist()
                            fc_label_tokens_sample = function_target_tokens[0, :min(100, function_target_tokens.shape[1])].cpu().tolist()
                            fc_loss_mask_sample = inputs["function_loss_scale"][0, :min(100, inputs["function_loss_scale"].shape[1]), 0].cpu().tolist()
                            fc_pred_text = self.tokenizer.ids_to_text([t for t, m in zip(fc_pred_tokens_sample, fc_loss_mask_sample) if m > 0 and t != self.text_pad_id])
                            fc_label_text = self.tokenizer.ids_to_text([t for t, m in zip(fc_label_tokens_sample, fc_loss_mask_sample) if m > 0 and t != self.text_pad_id])
                            logging.info(f"[FC Model Training]   Predicted text (calls only): '{fc_pred_text[:200]}'")
                            logging.info(f"[FC Model Training]   Label text (calls only): '{fc_label_text[:200]}'")
                            
                            # Calculate function call token accuracy (only on tokens with loss)
                            fc_valid_mask = (inputs["function_loss_scale"][:, :, 0] > 0)
                            if fc_valid_mask.sum() > 0:
                                fc_correct = (function_predicted_tokens == function_target_tokens) & fc_valid_mask
                                fc_accuracy = fc_correct.sum().float() / fc_valid_mask.sum().float()
                                logging.info(f"[FC Model Training]   Function call token accuracy: {fc_accuracy.item():.4f}")
                else:
                    if is_minimal_batch:
                        # Minimal/sync-only placeholder batches should not contribute FC loss,
                        # but keep function_head in the graph for distributed consistency.
                        function_loss = function_logits[..., :1].sum() * 0.0
                    else:
                        # Non-minimal batches (including regular non-FC) are expected to provide
                        # function_labels so function channel is trained (PAD for non-FC data).
                        raise RuntimeError(
                            "Missing function_labels on a non-minimal batch. "
                            "Expected PAD-supervised function channel for non-FC batches."
                        )

                with torch.no_grad():
                    predicted_tokens = torch.argmax(text_logits, dim=-1)  # (B, T)
                    target_tokens = inputs["text_labels"]  # (B, T)
                    valid_mask = (target_tokens != self.text_pad_id)

                    correct_predictions = (predicted_tokens == target_tokens) & valid_mask

                    if valid_mask.sum() > 0:
                        token_accuracy = correct_predictions.sum().float() / valid_mask.sum().float()
                    else:
                        token_accuracy = torch.tensor(0.0, device=text_logits.device)
                    
                    # Log predictions vs labels if debug_fc is enabled
                    if self.cfg.get("debug_fc", False):
                        logging.info(f"[FC Model Training] Agent text predictions:")
                        # Show first sample's predictions vs labels
                        pred_tokens_sample = predicted_tokens[0, :min(100, predicted_tokens.shape[1])].cpu().tolist()
                        label_tokens_sample = target_tokens[0, :min(100, target_tokens.shape[1])].cpu().tolist()
                        pred_text = self.tokenizer.ids_to_text(pred_tokens_sample)
                        label_text = self.tokenizer.ids_to_text(label_tokens_sample)
                        logging.info(f"[FC Model Training]   Predicted tokens[:100]: {pred_tokens_sample}")
                        logging.info(f"[FC Model Training]   Predicted text[:100]: '{pred_text[:200]}'")
                        logging.info(f"[FC Model Training]   Label tokens[:100]: {label_tokens_sample}")
                        logging.info(f"[FC Model Training]   Label text[:100]: '{label_text[:200]}'")
                        logging.info(f"[FC Model Training]   Token accuracy: {token_accuracy.item():.4f}")
                        
                        # Log user text predictions if enabled
                        if self.predict_user_text:
                            asr_predicted_tokens = torch.argmax(asr_logits, dim=-1)  # (B, T)
                            asr_target_tokens = inputs["asr_labels"]  # (B, T)
                            logging.info(f"[FC Model Training] User text predictions:")
                            asr_pred_tokens_sample = asr_predicted_tokens[0, :min(100, asr_predicted_tokens.shape[1])].cpu().tolist()
                            asr_label_tokens_sample = asr_target_tokens[0, :min(100, asr_target_tokens.shape[1])].cpu().tolist()
                            asr_pred_text = self.tokenizer.ids_to_text(asr_pred_tokens_sample)
                            asr_label_text = self.tokenizer.ids_to_text(asr_label_tokens_sample)
                            logging.info(f"[FC Model Training]   Predicted tokens[:100]: {asr_pred_tokens_sample}")
                            logging.info(f"[FC Model Training]   Predicted text[:100]: '{asr_pred_text[:200]}'")
                            logging.info(f"[FC Model Training]   Label tokens[:100]: {asr_label_tokens_sample}")
                            logging.info(f"[FC Model Training]   Label text[:100]: '{asr_label_text[:200]}'")

                # For placeholder minimal batches, keep distributed control-flow but do not optimize on fake labels.
                if is_minimal_batch:
                    # Keep autograd graph valid across ranks while making this batch contribute 0 loss.
                    text_loss = text_loss * 0.0
                    token_accuracy = token_accuracy * 0.0
                    if self.predict_user_text:
                        asr_loss = asr_loss * 0.0
                    function_loss = function_loss * 0.0
                    if self.cfg.get("debug_fc", False):
                        logging.info("[Training] Minimal batch detected: zeroing text/asr/function losses.")

                loss = self.cfg.text_loss_weight * text_loss
    
                if self.predict_user_text:
                    loss = loss + self.cfg.get('asr_loss_weight', 1.0) * asr_loss
                
                # Always connect function branch into the final loss when labels exist.
                # For minimal batches, function_loss is already zeroed above, so this keeps
                # autograd/FSDP graph parity across ranks without updating on fake labels.
                if "function_labels" in inputs and inputs["function_labels"] is not None:
                    loss = loss + self.cfg.get('function_loss_weight', 1.0) * function_loss

                B, T = inputs["input_embeds"].shape[:2]
                ans = {
                    "audio_loss": loss,
                    "audio_to_text_loss": text_loss,
                    "batch": B,
                    "length": T,
                    "token_accuracy": token_accuracy,
                    # Distributed safety: keep logged metric keys identical across ranks
                    # when batches are mixed (FC/minimal/non-FC) to avoid logger sync desync.
                    "function_loss": function_loss,
                }
                if self.predict_user_text:
                    ans["asr_loss"] = asr_loss

                res.update(ans)

        if batch["text_data"] is not None:
            text_input_ids = batch["text_data"]["text_tokens"][:, :-1]
            text_target = batch["text_data"]["text_tokens"][:, 1:]

            text_out = self.llm(
                inputs_embeds=self.embed_tokens(text_input_ids),
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            text_logits = self.lm_head(text_out['last_hidden_state'])  # (B, T, Vt)

            text_loss = torch.nn.functional.cross_entropy(
                text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                text_target.flatten(0, 1),
                ignore_index=self.text_pad_id,
            )
            res.update(
                {
                    "text_to_text_loss": text_loss,
                }
            )

        res["loss"] = (1. - self.cfg.get('text_to_text_loss_weight', 0.0)) * res.get("audio_loss", 0.0) + \
                      self.cfg.get('text_to_text_loss_weight', 0.0) * res.get("text_to_text_loss", 0.0)

        # Track early interruption augmentation stats
        early_interruption_stats = batch.get("early_interruption_stats")
        if early_interruption_stats is not None:
            self.early_interruption_total += early_interruption_stats["batch_total"]
            self.early_interruption_attempted += early_interruption_stats["batch_attempted"]
            self.early_interruption_successful += early_interruption_stats["batch_successful"]

        # Always execute this sync_dist log so all ranks participate in the same collectives,
        # even when some ranks only see FC/minimal batches and do not update EI counters.
        early_interruption_successful_ratio = (
            self.early_interruption_successful / self.early_interruption_total
            if self.early_interruption_total > 0
            else 0.0
        )
        self.log(
            "early_interruption_successful_ratio",
            early_interruption_successful_ratio,
            on_step=True,
            sync_dist=True,
        )

        self.log_dict(res, on_step=True)

        return res

    def on_train_epoch_start(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.bleu = BLEU().reset()
        # BLEU for "assistant response after TOOLRESPONSE": ref = extracted GT segment, hyp = full agent prediction
        self.bleu_after_tool = BLEU().reset()

        self.turn_taking_metrics = TurnTakingMetrics(
            eos_token_id=self.tokenizer.text_to_ids('$')[0],
            bos_token_id=self.text_bos_id,
            tolerance=13,
            latency_multiplier=0.08
        ).reset()

        if self.predict_user_text:
            self.src_bleu = BLEU().reset()
            self.src_wer = TextWER().reset()
            self.empty_user_text = EmptyTextMetric().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            if "qa" not in k and "mmsu" not in k:
                self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        # BLEU: ref = ground-truth "assistant response after TOOLRESPONSE" only, hyp = full agent prediction.
        # Log key is val_txt_bleu_after_tool (distinct from main BLEU's val_txt_bleu / val_txt_bleu_{name}).
        # Always log the same key so all ranks participate in sync_dist.
        bleu_after_tool = self.bleu_after_tool.compute()
        after_tool_val = bleu_after_tool["txt_bleu"].to(self.device) if bleu_after_tool and "txt_bleu" in bleu_after_tool else torch.tensor(0.0, device=self.device)
        self.log(f"{prefix}_txt_bleu_after_tool", after_tool_val, on_epoch=True, sync_dist=True)

        acc_metrics = self.results_logger.compute_and_save()

        for name, result_dict in acc_metrics.items():
            if 'acc' in result_dict:
                self.log(f"{prefix}_{name}_acc", result_dict['acc'].to(self.device), on_epoch=True, sync_dist=True)

            if 'mcq_acc' in result_dict:
                self.log(f"{prefix}_{name}_mcq_acc", result_dict['mcq_acc'].to(self.device), on_epoch=True,
                         sync_dist=True)

        turn_taking_metrics = self.turn_taking_metrics.compute()
        for k, m in turn_taking_metrics.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        if self.predict_user_text:
            src_bleu = self.src_bleu.compute()
            for k, m in src_bleu.items():
                self.log(f"{prefix}_src_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
            src_wer = self.src_wer.compute()
            for k, m in src_wer.items():
                self.log(f"{prefix}_src_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
            empty_user_text = self.empty_user_text.compute()
            for k, m in empty_user_text.items():
                self.log(f"{prefix}_src_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue

            dataset_batch = dataset_batch["audio_data"]
            if dataset_batch.get("is_minimal_batch", False):
                continue  # Skip placeholder minimal batches in validation

            prompt_tokens = dataset_batch.get("prompt_tokens", None)
            prompt_token_lens = dataset_batch.get("prompt_token_lens", None)
            
            # Get function calling data if present in test set
            # function_calls: ground truth calls (for logging/comparison)
            # function_responses: API responses (for inference injection)
            function_calls = dataset_batch.get("function_calls", None)
            function_call_lengths = dataset_batch.get("function_call_lengths", None)
            function_call_steps = dataset_batch.get("function_call_steps", None)
            function_responses = dataset_batch.get("function_responses", None)
            function_response_lengths = dataset_batch.get("function_response_lengths", None)
            function_response_steps = dataset_batch.get("function_response_steps", None)

            # Choose between online and offline inference based on config
            use_online_inference = self.cfg.get("use_online_inference", False)
            
            # Use fixed-length inference (pre-allocated, faster)
            extra_decoding_seconds = self.cfg.get("extra_decoding_seconds", 0.0)
            input_pad_len = int(extra_decoding_seconds * self.source_sample_rate)
            logging.info(f"Padding input audio by {extra_decoding_seconds} seconds")

            if use_online_inference:
                logging.info(f"Using ONLINE inference for validation (window_size={self.cfg.get('online_window_size', 70)})")
                results = self.online_inference(
                    dataset_batch["source_audio"],
                    dataset_batch["source_audio_lens"],
                    prompt_tokens=prompt_tokens,
                    prompt_token_lens=prompt_token_lens,
                )
            else:
                # Use fixed-length inference (pre-allocated, faster)
                results = self.offline_inference(
                    dataset_batch["source_audio"],
                    dataset_batch["source_audio_lens"],
                    input_pad_len=input_pad_len,
                    prompt_tokens=prompt_tokens,
                    prompt_token_lens=prompt_token_lens,
                    sample_id=dataset_batch.get("sample_id", None),
                    function_calls=function_calls,
                    function_call_lengths=function_call_lengths,
                    function_responses=function_responses,
                    function_response_lengths=function_response_lengths,
                    function_response_steps=function_response_steps,
                    function_call_steps=function_call_steps,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

            # BLEU for assistant response after TOOLRESPONSE: refs = segments after each TOOLRESPONSE (dataset), hyp = full agent prediction per sample.
            # Skip only empty references (no GT to compare to); empty hypotheses are included and score 0.
            if function_calls is not None:
                n_samples = len(results["text"])
                raw_refs = dataset_batch.get("target_text_after_tool_response")
                if raw_refs is None or len(raw_refs) != n_samples:
                    raw_refs = [[] for _ in range(n_samples)]
                full_hyps = results["text"]
                refs_filt, hyps_filt = [], []
                for sample_refs, h in zip(raw_refs, full_hyps):
                    if not isinstance(sample_refs, (list, tuple)):
                        sample_refs = [sample_refs] if (sample_refs is not None and str(sample_refs).strip()) else []
                    h = h if h is not None else ""
                    for r in sample_refs:
                        if r is None or not str(r).strip():
                            continue
                        refs_filt.append(r)
                        hyps_filt.append(h)
                if refs_filt and hyps_filt:
                    self.bleu_after_tool.update(name=name, refs=refs_filt, hyps=hyps_filt)

            if "source_tokens" in dataset_batch and results["tokens_text"] is not None:
                self.turn_taking_metrics.update(
                    name=name,
                    source_tokens=dataset_batch["source_tokens"],
                    pred_tokens=results["tokens_text"]
                )

            fake_pred_audio, fake_audio_len = self._generate_fake_audio_from_tokens(results["tokens_text"])

            pred_turns_list = self._split_agent_tokens_into_turns(results["tokens_text"])

            # Decode function channel tokens to text
            function_channel_text = None
            function_call_positions = None
            func_tokens_for_pred = results.get("tokens_function_pred", results.get("tokens_function", None))
            if func_tokens_for_pred is not None:
                function_channel_text = tokens_to_str(
                    func_tokens_for_pred,
                    results["tokens_len"], 
                    tokenizer=self.tokenizer, 
                    pad_id=self.text_pad_id, 
                    user_bos_id=self.user_bos_id, 
                    eval_text_turn_taking=False,
                    sil_id=None
                )

                # Also decode full function channel (includes prefilled responses) for debugging
                function_channel_with_inserted_response = None
                if results.get("tokens_function") is not None:
                    function_channel_with_inserted_response = tokens_to_str(
                        results["tokens_function"],
                        results["tokens_len"],
                        tokenizer=self.tokenizer,
                        pad_id=self.text_pad_id,
                        user_bos_id=self.user_bos_id,
                        eval_text_turn_taking=False,
                        sil_id=None
                    )
                
                # Extract function call positions/timing
                function_call_positions = self._extract_function_call_positions(
                    func_tokens_for_pred,
                    results["tokens_len"],
                    results["tokens_text"]
                )
                
                if self.cfg.get("fc_log", False):
                    logging.info(f"[Function Channel Predictions - {name}]:")
                    for idx, fc_text in enumerate(function_channel_text):
                        sample_id = dataset_batch['sample_id'][idx]
                        logging.info(f"  Sample {sample_id}: {fc_text}")
                        try:
                            pred_tokens = func_tokens_for_pred[idx, :results["tokens_len"][idx]].tolist()
                            logging.info(f"    tokens_function_pred[:40]: {pred_tokens[:40]}")
                        except Exception as e:
                            logging.info(f"    tokens_function_pred: <unavailable> ({e})")
                        if function_call_positions is not None and idx < len(function_call_positions):
                            pos_info = function_call_positions[idx]
                            logging.info(f"    Timeline for sample {sample_id}:")
                            if pos_info.get("user_speech_segments"):
                                for i, seg in enumerate(pos_info["user_speech_segments"]):
                                    logging.info(f"      User Speech {i+1}: pos [{seg['start_pos']}:{seg['end_pos']}]")
                            if pos_info.get("function_calls"):
                                for i, call in enumerate(pos_info["function_calls"]):
                                    logging.info(f"      Function Call {i+1}: pos [{call['start_pos']}:{call['end_pos']}]")
                            if pos_info.get("agent_text_segments"):
                                for i, seg in enumerate(pos_info["agent_text_segments"]):
                                    logging.info(f"      Agent Response {i+1}: pos [{seg['start_pos']}:{seg['end_pos']}] - '{seg['text_preview']}'")

            # Decode ground truth function channel tokens (target)
            # Note: target_function_channel contains ground truth CALLS (what model should predict)
            # NOT responses (which come from external APIs and are used for inference injection)
            target_function_channel = None
            if function_calls is not None:
                # function_calls shape: [B, T, L] where T is num calls, L is max length
                B = function_calls.shape[0]
                target_function_channel_list = []
                
                for b in range(B):
                    # Flatten all function calls for this batch item into one sequence
                    calls_for_batch = []
                    if function_call_lengths is not None:
                        num_calls = (function_call_lengths[b] > 0).sum().item()
                        for t in range(num_calls):
                            call_length = function_call_lengths[b, t].item()
                            if call_length > 0:
                                call_tokens = function_calls[b, t, :call_length]
                                call_text = self.tokenizer.ids_to_text(call_tokens.tolist())
                                calls_for_batch.append(call_text)
                    
                    # Join all calls into one string
                    target_text = "".join(calls_for_batch) if calls_for_batch else ""
                    target_function_channel_list.append(target_text)
                
                target_function_channel = target_function_channel_list
                if self.cfg.get("fc_log", False):
                    logging.info(f"[Target Function Channel - {name}]:")
                    for idx, target_fc_text in enumerate(target_function_channel):
                        sample_id = dataset_batch['sample_id'][idx]
                        logging.info(f"  Sample {sample_id}: {target_fc_text}")

            self.results_logger.update(
                name=name,
                refs=dataset_batch["target_texts"],
                hyps=results["text"],
                asr_hyps=None,
                samples_id=dataset_batch['sample_id'],
                pred_audio=fake_pred_audio,
                pred_audio_sr=self.target_sample_rate,
                # Use padded input audio from inference so saved WAV includes extra decoding tail.
                user_audio=results["source_audio"],
                user_audio_sr=self.source_sample_rate,
                src_refs=dataset_batch["source_texts"],
                src_hyps=results["src_text"],
                system_prompt=dataset_batch.get("system_prompt", None),
                system_prompt_supervision_0=dataset_batch.get("system_prompt_supervision_0", None),
                source_turns=dataset_batch.get("source_turn_texts"),
                target_turns=dataset_batch.get("target_turn_texts"),
                pred_turns=pred_turns_list,
                function_channel_text=function_channel_text,
                function_channel_with_inserted_response=function_channel_with_inserted_response,
                target_function_channel=target_function_channel,
                function_call_positions=function_call_positions,
                target_text_after_tool_response=dataset_batch.get("target_text_after_tool_response"),
            )

            if self.cfg.get("eval_text_turn_taking", False):
                import re
                results["text"] = [re.sub(r"<\|.*?\|>", "", s).strip() for s in results["text"]]

            if self.predict_user_text:
                src_text_clean = [s.replace("^", " ").replace("$", " ") for s in results["src_text"]]
                self.src_bleu.update(name=name, refs=dataset_batch["source_texts"], hyps=src_text_clean)
                self.src_wer.update(name=name, refs=dataset_batch["source_texts"], hyps=src_text_clean)
                self.empty_user_text.update(name=name, hyps=results["src_text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def on_predict_epoch_start(self) -> None:
        return self.on_train_epoch_start()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        batch = batch["audio_data"]
        if batch.get("is_minimal_batch", False):
            # Return minimal prediction for placeholder batches (e.g. dropped FC)
            B = 1
            return {
                "text": [""],
                "src_text": [""] if self.predict_user_text else None,
                "tokens_text_src": torch.full((B, 0), self.text_pad_id, device=self.device, dtype=torch.long) if self.predict_user_text else None,
                "tokens_text": torch.full((B, 0), self.text_pad_id, device=self.device, dtype=torch.long),
                "tokens_function": torch.full((B, 0), self.text_pad_id, device=self.device, dtype=torch.long),
                "tokens_function_pred": torch.full((B, 0), self.text_pad_id, device=self.device, dtype=torch.long),
                "tokens_audio": None,
                "tokens_len": torch.tensor([0], device=self.device, dtype=torch.long),
                "source_audio": batch["source_audio"],
                "source_audio_len": batch["source_audio_lens"],
                "sample_id": batch.get("sample_id", ["empty_batch"]),
            }

        force_bos_positions = None
        force_bos_num_tokens_after_user_eos = self.cfg.prediction.get("force_bos_num_tokens_after_user_eos", None)
        if force_bos_num_tokens_after_user_eos is not None:
            force_bos_positions = []
            for cur_source_tokens in batch["source_tokens"]:
                tmp = torch.where(cur_source_tokens == self.text_eos_id)[0]
                if len(tmp) > 0:
                    force_bos_positions.append(tmp[0].item() + force_bos_num_tokens_after_user_eos)
                else:
                    force_bos_positions.append(None)

        prompt_tokens = batch.get("prompt_tokens", None)
        prompt_token_lens = batch.get("prompt_token_lens", None)
        
        # Get function calling data if present
        function_calls = batch.get("function_calls", None)
        function_call_lengths = batch.get("function_call_lengths", None)
        function_call_steps = batch.get("function_call_steps", None)
        function_responses = batch.get("function_responses", None)
        function_response_lengths = batch.get("function_response_lengths", None)
        function_response_steps = batch.get("function_response_steps", None)

        # Use fixed-length inference (pre-allocated, faster)
        prediction = self.offline_inference(
            batch["source_audio"],
            batch["source_audio_lens"],
            decode_audio=self.cfg.prediction.decode_audio,
            input_pad_len=self.cfg.prediction.max_new_seconds * self.cfg.prediction.input_sample_rate,
            force_bos_positions=force_bos_positions,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
            sample_id=batch.get("sample_id", None),
            function_calls=function_calls,
            function_call_lengths=function_call_lengths,
            function_responses=function_responses,
            function_response_lengths=function_response_lengths,
            function_response_steps=function_response_steps,
            function_call_steps=function_call_steps,
        )
        prediction["sample_id"] = batch["sample_id"]
        return prediction

    def _get_bos_embedding(self) -> torch.Tensor:
        """Get BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    def _get_asr_bos_embedding(self) -> torch.Tensor:
        """Get ASR BOS embedding for AR decoding."""
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_asr_tokens(text_bos)
        return input_embeds

    def _remove_continuous_agent_bos_id(self, gen_text: torch.Tensor, bos_id: int,
                                        is_asr: bool = False) -> torch.Tensor:
        """Remove continuous appearance of bos_id."""
        if is_asr:
            cleaned_gen_text = gen_text.clone()
            for b in range(cleaned_gen_text.size(0)):
                in_bos = False
                for t in range(cleaned_gen_text.size(1)):
                    token = cleaned_gen_text[b, t]
                    if token == bos_id:
                        if in_bos:
                            cleaned_gen_text[b, t] = self.text_pad_id
                        else:
                            in_bos = True
                    elif token == self.text_pad_id:
                        continue
                    else:
                        in_bos = False
            gen_text = cleaned_gen_text
        return gen_text

    def _remove_last_turn_if_short(self, gen_text: torch.Tensor, bos_id: int, is_asr: bool = False) -> torch.Tensor:
        """If the last turn contains less than 5 non-pad tokens, set the last turn all to pad."""
        if is_asr:
            fixed_gen_text = gen_text.clone()

            for b in range(gen_text.size(0)):
                bos_indices = (gen_text[b] == bos_id).nonzero(as_tuple=True)[0]

                if len(bos_indices) > 0:
                    last_bos_idx = bos_indices[-1].item()
                    last_turn_tokens = gen_text[b, last_bos_idx:]
                    non_pad_count = (last_turn_tokens != self.text_pad_id).sum().item()

                    if non_pad_count < 5:
                        fixed_gen_text[b, last_bos_idx + 1:] = self.text_pad_id
            return fixed_gen_text
        else:
            return gen_text

    def _find_agent_bos(self, gen_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        agent_bos_id = self.text_bos_id
        agent_bos_indices = (gen_text == agent_bos_id).nonzero(as_tuple=True)
        return agent_bos_indices

    def _segment_alternating_user_agent_text(self, gen_text: torch.Tensor, is_asr: bool = False, user_eos_id=None) -> \
    tuple[torch.Tensor, torch.Tensor]:
        """Segment text into alternating user and agent text segments."""
        user_bos_id = self.user_bos_id
        agent_bos_id = self.text_bos_id

        if is_asr:
            gen_text_src = torch.where(gen_text == agent_bos_id, user_eos_id, gen_text)
            gen_text_tgt = gen_text.clone()
            return gen_text_src, gen_text_tgt

        user_mask = torch.zeros_like(gen_text, dtype=torch.bool)
        agent_mask = torch.zeros_like(gen_text, dtype=torch.bool)

        for b in range(gen_text.size(0)):
            user_bos_indices = (gen_text[b] == user_bos_id).nonzero(as_tuple=True)[0]
            agent_bos_indices = (gen_text[b] == agent_bos_id).nonzero(as_tuple=True)[0]

            all_bos_positions = []
            for idx in user_bos_indices:
                all_bos_positions.append((idx.item(), 'user'))
            for idx in agent_bos_indices:
                all_bos_positions.append((idx.item(), 'agent'))

            all_bos_positions.sort(key=lambda x: x[0])

            current_type = None
            segment_start = 0

            for pos, bos_type in all_bos_positions:
                if current_type is not None:
                    if current_type == 'user':
                        user_mask[b, segment_start:pos] = True
                    else:
                        agent_mask[b, segment_start:pos] = True

                current_type = bos_type
                segment_start = pos

            if current_type is not None:
                if current_type == 'user':
                    user_mask[b, segment_start:] = True
                else:
                    agent_mask[b, segment_start:] = True

        gen_text_src = gen_text.clone()
        gen_text_src[~user_mask] = self.text_pad_id

        gen_text_tgt = gen_text.clone()
        gen_text_tgt[~agent_mask] = self.text_pad_id

        return gen_text_src, gen_text_tgt

    def _extract_function_call_positions(self, tokens_function: torch.Tensor, tokens_len: torch.Tensor, tokens_text: torch.Tensor) -> list:
        """
        Extract position/timing information for function calls and agent text predictions.
        
        Returns:
            List of dicts for each batch item, each containing:
            - function_calls: list of {"start_pos", "end_pos", "call_text"}
            - agent_text_segments: list of {"start_pos", "end_pos", "text_preview"}
            - user_speech_segments: list of {"start_pos", "end_pos"}
            - total_length: total sequence length
        """
        sotc_id, eotc_id, eotr_id = self._get_function_call_special_tokens()
        B = tokens_function.shape[0]
        
        positions_list = []
        
        for b in range(B):
            length = tokens_len[b].item()
            func_tokens = tokens_function[b, :length].cpu().tolist()
            text_tokens = tokens_text[b, :length].cpu().tolist()
            
            # Find function call boundaries
            function_calls = []
            current_call_start = None
            
            for pos, token_id in enumerate(func_tokens):
                if token_id == sotc_id and current_call_start is None:
                    # Start of function call
                    current_call_start = pos
                elif token_id == eotc_id and current_call_start is not None:
                    # End of function call
                    call_tokens = func_tokens[current_call_start:pos+1]
                    call_text = self.tokenizer.ids_to_text(call_tokens)
                    function_calls.append({
                        "start_pos": current_call_start,
                        "end_pos": pos,
                        "call_text": call_text
                    })
                    current_call_start = None
            
            # Find all user speech segments (marked by user BOS)
            user_speech_segments = []
            if self.user_bos_id is not None:
                current_user_start = None
                for pos, token_id in enumerate(text_tokens):
                    if token_id == self.user_bos_id:
                        if current_user_start is not None:
                            # Previous user segment ended
                            user_speech_segments.append({
                                "start_pos": current_user_start,
                                "end_pos": pos - 1
                            })
                        current_user_start = pos
                
                # Close last user segment if any
                if current_user_start is not None:
                    # Find where user speech likely ends (next agent BOS or EOS)
                    user_end = current_user_start
                    for pos in range(current_user_start + 1, length):
                        if text_tokens[pos] == self.text_bos_id or text_tokens[pos] == self.text_eos_id:
                            user_end = pos - 1
                            break
                        user_end = pos
                    
                    user_speech_segments.append({
                        "start_pos": current_user_start,
                        "end_pos": user_end
                    })
            
            # Find all agent text segments (marked by agent BOS)
            agent_text_segments = []
            if self.text_bos_id is not None:
                current_agent_start = None
                for pos, token_id in enumerate(text_tokens):
                    if token_id == self.text_bos_id:
                        if current_agent_start is not None:
                            # Previous agent segment ended, extract text
                            segment_tokens = text_tokens[current_agent_start:pos]
                            # Get preview (first 50 chars)
                            segment_text = self.tokenizer.ids_to_text(segment_tokens)
                            text_preview = segment_text[:50] + "..." if len(segment_text) > 50 else segment_text
                            agent_text_segments.append({
                                "start_pos": current_agent_start,
                                "end_pos": pos - 1,
                                "text_preview": text_preview
                            })
                        current_agent_start = pos
                
                # Close last agent segment if any
                if current_agent_start is not None:
                    # Find where agent speech likely ends (EOS or end of sequence)
                    agent_end = length - 1
                    for pos in range(current_agent_start + 1, length):
                        if text_tokens[pos] == self.text_eos_id:
                            agent_end = pos
                            break
                    
                    segment_tokens = text_tokens[current_agent_start:agent_end+1]
                    segment_text = self.tokenizer.ids_to_text(segment_tokens)
                    text_preview = segment_text[:50] + "..." if len(segment_text) > 50 else segment_text
                    agent_text_segments.append({
                        "start_pos": current_agent_start,
                        "end_pos": agent_end,
                        "text_preview": text_preview
                    })
            
            positions_list.append({
                "function_calls": function_calls,
                "agent_text_segments": agent_text_segments,
                "user_speech_segments": user_speech_segments,
                "total_length": length
            })
        
        return positions_list

    def _split_agent_tokens_into_turns(self, tokens_text: torch.Tensor):
        """Split sequence of agent_tokens into turns as detected by text_bos_id and text_eos_id."""
        batch_size, seq_len = tokens_text.shape
        token_duration = 0.08

        turns_list = []

        for b in range(batch_size):
            current_tokens = tokens_text[b].cpu().numpy()

            in_turn = False
            current_turn_start = None
            current_turn_tokens = []
            batch_turns = []

            def _save_current_turn(turn_start, turn_tokens, end_token_idx, is_complete=True):
                if turn_start is None:
                    return

                start_time = turn_start * token_duration
                end_time = (end_token_idx + 1) * token_duration
                duration = end_time - start_time

                if len(turn_tokens) > 0:
                    turn_tokens_filtered = [t for t in turn_tokens if t != self.text_pad_id]
                    text = self.tokenizer.ids_to_text(turn_tokens_filtered)
                else:
                    text = ""

                turn = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "text": text,
                    "token_ids": turn_tokens.copy(),
                    "start_token_idx": turn_start,
                    "end_token_idx": end_token_idx,
                    "num_tokens": len(turn_tokens),
                    "is_complete": is_complete,
                }
                batch_turns.append(turn)

            for t in range(seq_len):
                token_id = current_tokens[t]

                if token_id == self.text_bos_id:
                    if in_turn and current_turn_start is not None:
                        logging.debug(
                            f"Batch {b}: Found BOS at position {t} while already in a turn "
                            f"(started at {current_turn_start}). Saving incomplete turn."
                        )
                        _save_current_turn(current_turn_start, current_turn_tokens, end_token_idx=t - 1,
                                           is_complete=False)

                    in_turn = True
                    current_turn_start = t
                    current_turn_tokens = []

                elif token_id == self.text_eos_id:
                    if in_turn:
                        _save_current_turn(current_turn_start, current_turn_tokens, end_token_idx=t, is_complete=True)

                        current_turn_start = None
                        current_turn_tokens = []
                        in_turn = False

                elif token_id == self.text_pad_id:
                    if in_turn:
                        current_turn_tokens.append(token_id)
                else:
                    if in_turn:
                        current_turn_tokens.append(token_id)

            if in_turn and current_turn_start is not None:
                logging.debug(
                    f"Batch {b}: Sequence ended while in a turn (started at {current_turn_start}). "
                    f"Saving incomplete turn."
                )
                _save_current_turn(current_turn_start, current_turn_tokens, end_token_idx=seq_len - 1,
                                   is_complete=False)

            turns_list.append(batch_turns)

        return turns_list

    def _generate_fake_audio_from_tokens(self, tokens_text: torch.Tensor):
        """Generate fake audio based on text tokens for analysis."""
        batch_size, seq_len = tokens_text.shape
        token_duration = 0.08
        samples_per_token = int(token_duration * self.target_sample_rate)
        audio_len = seq_len * samples_per_token

        sil_id = None
        if self.cfg.get("use_sil_token", False):
            if 'Nemotron' in self.cfg.pretrained_llm:
                sil_id = self.tokenizer.tokenizer._tokenizer.token_to_id('<SPECIAL_11>')
            elif 'Qwen2.5' in self.cfg.pretrained_llm:
                sil_id = self.tokenizer.tokenizer._tokenizer.token_to_id('<|object_ref_start|>')

        fake_audio = torch.zeros(batch_size, audio_len, device=tokens_text.device, dtype=torch.float32)
        audio_lengths = torch.full((batch_size,), audio_len, device=tokens_text.device, dtype=torch.long)

        for b in range(batch_size):
            current_tokens = tokens_text[b].cpu().numpy()
            audio_values = torch.zeros(seq_len, device=tokens_text.device, dtype=torch.float32)

            in_speech = False

            for t in range(seq_len):
                token_id = int(current_tokens[t])

                if token_id == self.text_bos_id:
                    in_speech = True
                    audio_values[t] = 1.0
                elif token_id == self.text_eos_id:
                    in_speech = False
                    audio_values[t] = 0.0
                elif sil_id is not None and token_id == sil_id:
                    audio_values[t] = 0.2
                elif token_id == self.text_pad_id:
                    if in_speech:
                        audio_values[t] = 0.5
                    else:
                        audio_values[t] = 0.0
                else:
                    if in_speech:
                        audio_values[t] = 1.0
                    else:
                        audio_values[t] = 0.0

            for t in range(seq_len):
                start_sample = t * samples_per_token
                end_sample = min((t + 1) * samples_per_token, audio_len)
                fake_audio[b, start_sample:end_sample] = audio_values[t]

        return fake_audio, audio_lengths

    def _write_debug_info(self, input_signal: torch.Tensor, source_encoded: torch.Tensor, sample_id=None):
        """Write debug information for input_signal and source_encoded to file."""
        
        debug_dir = "/lustre/fsw/portfolios/convai/users/kevinhu/debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Extract filename from sample_id or use a default
        if sample_id is not None:
            if isinstance(sample_id, (list, tuple)):
                filename_base = str(sample_id[0]) if len(sample_id) > 0 else "unknown"
            else:
                filename_base = str(sample_id)
            # Remove path separators and file extensions
            filename_base = os.path.basename(filename_base).replace('.', '_')
        else:
            filename_base = "unknown"
        
        # Save both tensors to a single .pt file
        # input_signal shape: [B, T] where T is time
        # source_encoded shape: [B, T, H] where T is time, H is hidden dim
        offline_file = os.path.join(debug_dir, f"offline_{filename_base}.pt")
        
        torch.save({
            'input_signal': input_signal.detach().cpu(),
            'source_encoded': source_encoded.detach().cpu()
        }, offline_file)
        
        print(f"Debug info written to {debug_dir}:")
        print(f"  - offline_{filename_base}.pt")
        print(f"    input_signal shape: {input_signal.shape}")
        print(f"    source_encoded shape: {source_encoded.shape}")

    def _init_inference(
            self,
            input_signal: torch.Tensor,
            input_signal_lens: torch.Tensor,
            input_pad_len: int,
            force_bos_positions,
            prompt_tokens: torch.Tensor,
            prompt_token_lens: torch.Tensor,
            sample_id=None,
    ):
        """Initialize inference resources and prepare inputs."""
        sil_id = None
        if 'Nemotron' in self.cfg.pretrained_llm:
            sil_id = self.tokenizer.tokenizer._tokenizer.token_to_id('<SPECIAL_11>')
        elif 'Qwen2.5' in self.cfg.pretrained_llm:
            sil_id = self.tokenizer.tokenizer._tokenizer.token_to_id('<|object_ref_start|>')

        if self.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            input_signal, sr = torchaudio.load(self.cfg.custom_sample_inference)
            input_signal = input_signal.to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        if force_bos_positions is not None:
            assert input_signal.shape[0] == len(
                force_bos_positions), "force_bos_positions must have the same length as batch size"

        if input_pad_len > 0:
            input_signal = torch.nn.functional.pad(input_signal, (0, input_pad_len), mode='constant', value=0)
            input_signal_lens = input_signal_lens + input_pad_len

        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True, sample_id=sample_id
        )
        
        # Write debug information
        # self._write_debug_info(input_signal, source_encoded, sample_id)

        B, T_local, H = source_encoded.shape

        if prompt_tokens is not None and prompt_token_lens is not None:
            prompt_embedded = self.embed_tokens(prompt_tokens)
            B_prompt, max_prompt_len, H_prompt = prompt_embedded.shape

            if self.cfg.get("fc_log", False):
                logging.info(f"[Inference Init] System prompt detected: batch_size={B_prompt}, max_prompt_len={max_prompt_len}")
                for i in range(min(B_prompt, 2)):  # Show first 2 samples
                    prompt_len = prompt_token_lens[i].item()
                    if prompt_len > 0:
                        prompt_tokens_sample = prompt_tokens[i, :prompt_len].tolist()
                        prompt_text = self.tokenizer.ids_to_text(prompt_tokens_sample)
                        logging.info(f"[Inference Init] Sample {i} system prompt ({prompt_len} tokens):")
                        logging.info(f"[Inference Init]   Full text: {prompt_text}")
                        if len(prompt_text) > 200:
                            logging.info(f"[Inference Init]   (truncated preview): {prompt_text[:200]}...")

            assert B == B_prompt, f"Batch size mismatch: source={B}, prompt={B_prompt}"
            assert H == H_prompt, f"Hidden size mismatch: source={H}, prompt={H_prompt}"

            new_source_encoded = torch.zeros(B, max_prompt_len + T_local, H,
                                             dtype=source_encoded.dtype, device=source_encoded.device)

            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len = prompt_len.item()

                if prompt_len > 0:
                    new_source_encoded[i, :prompt_len, :] = prompt_embedded[i, :prompt_len, :]

                src_len = lengths[i].item()
                new_source_encoded[i, prompt_len:prompt_len + src_len, :] = source_encoded[i, :src_len, :]

                lengths[i] = prompt_len + src_len

            if self.cfg.get("debug_fc", False):
                logging.info(f"[Inference Init] After prompt prepending: source_encoded shape={new_source_encoded.shape}")
                for i in range(min(B, 2)):
                    prompt_len = prompt_token_lens[i].item()
                    total_len = lengths[i].item()
                    logging.info(f"[Inference Init] Sample {i}: prompt_len={prompt_len}, total_len={total_len}, audio_len={total_len - prompt_len}")

            source_encoded = new_source_encoded
            T_local = source_encoded.shape[1]

        B, T_local, H = source_encoded.shape

        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame_source = source_encoded[:, T_local - 1: T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_encoded = torch.cat([source_encoded, pad_source], dim=1)
                last_frame_asr = asr_emb[:, T_local - 1: T_local, :]
                pad_asr = last_frame_asr.repeat(1, T - T_local, 1)
                asr_emb = torch.cat([asr_emb, pad_asr], dim=1)
        else:
            T = T_local

        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        use_cache = True
        if 'Nemotron' in self.cfg.pretrained_llm:
            cache = None
            use_cache = False
            logging.info("Using no-cache mode for Nemotron (full history each step)")
        else:
            cache = DynamicCache()
            use_cache = True

        use_pad_init = self.cfg.get("inference_init_with_pad", True)
        if use_pad_init:
            gen_text = torch.full((B, T), self.text_pad_id, device=self.device, dtype=torch.long)
        else:
            gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        if self.predict_user_text:
            if use_pad_init:
                gen_asr = torch.full((B, T), self.text_pad_id, device=self.device, dtype=torch.long)
            else:
                gen_asr = torch.empty(B, T, device=self.device, dtype=torch.long)
        else:
            gen_asr = None
        
        # Initialize function calling channel (all PAD tokens initially, shared vocab with text)
        gen_function = torch.full((B, T), self.text_pad_id, device=self.device, dtype=torch.long)

        if prompt_tokens is not None and prompt_token_lens is not None:
            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len = prompt_len.item()
                if prompt_len > 0:
                    gen_text[i, :prompt_len] = self.text_pad_id
                    if self.predict_user_text:
                        gen_asr[i, :prompt_len] = self.text_pad_id
                    # Function channel also starts with PAD in prompt region
                    gen_function[i, :prompt_len] = self.text_pad_id
            
            if self.cfg.get("debug_fc", False):
                logging.info(f"[Inference Init] Initialized generation tensors with PAD in prompt region")
                for i in range(min(B, 2)):
                    prompt_len = prompt_token_lens[i].item()
                    if prompt_len > 0:
                        pad_count_text = (gen_text[i, :prompt_len] == self.text_pad_id).sum().item()
                        pad_count_func = (gen_function[i, :prompt_len] == self.text_pad_id).sum().item()
                        logging.info(f"[Inference Init] Sample {i}: prompt_len={prompt_len}, "
                                   f"gen_text PAD={pad_count_text}, gen_function PAD={pad_count_func}")

        input_embeds[:, 0] += self._get_bos_embedding() * self.cfg.get("duplex_text_channel_weight", 1.0)
        if self.predict_user_text:
            input_embeds[:, 0] += self._get_asr_bos_embedding() * self.cfg.get("duplex_asr_text_weight", 1.0)
        # Function channel starts with PAD (no embedding contribution for PAD)

        start_gen_pos = 0
        if prompt_token_lens is not None:
            max_prompt_len = prompt_token_lens.max().item()
            start_gen_pos = max_prompt_len
            
            if self.cfg.get("debug_fc", False):
                logging.info(f"[Inference Init] Generation will start at position {start_gen_pos} (after prompt)")

        is_prompt_position_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        if prompt_token_lens is not None:
            for i, prompt_len in enumerate(prompt_token_lens):
                prompt_len_val = prompt_len.item()
                if prompt_len_val > 0:
                    is_prompt_position_mask[i, :prompt_len_val] = True
            
            if self.cfg.get("debug_fc", False):
                prompt_mask_counts = is_prompt_position_mask.sum(dim=1)
                logging.info(f"[Inference Init] Prompt position mask created: {prompt_mask_counts.tolist()}")

        # [TESTING] Silence template initialization DISABLED - using zeros for silence
        # This is a temporary test to isolate the NCCL timeout issue
        # if hasattr(self, '_ensure_silence_template_initialized'):
        #     if self.cfg.get("debug_fc", False):
        #         logging.info("[Inference Init] Initializing silence template for function calling...")
        #     self._ensure_silence_template_initialized()
        #     if self.cfg.get("debug_fc", False):
        #         logging.info("[Inference Init] Silence template ready")
        if self.cfg.get("debug_fc", False):
            logging.info("[Inference Init] Using ZERO embeddings for silence (testing mode)")
        
        return {
            "sil_id": sil_id,
            "input_signal": input_signal,
            "input_signal_lens": input_signal_lens,
            "asr_emb": asr_emb,
            "lengths": lengths,
            "B": B,
            "T": T,
            "T_local": T_local,
            "input_embeds": input_embeds,
            "cache": cache,
            "use_cache": use_cache,
            "gen_text": gen_text,
            "gen_asr": gen_asr,
            "gen_function": gen_function,
            "start_gen_pos": start_gen_pos,
            "is_prompt_position_mask": is_prompt_position_mask,
            "sample_id": sample_id,
        }

    def _step_zero(self, inference_state):
        """Perform inference for the first step (position 0)."""
        ans = self(
            inference_state["input_embeds"][:, :1],
            cache=inference_state["cache"],
            input_audio_tokens=None,
            seq_mask=None,
            target_text_tokens=None,
        )

        if inference_state["start_gen_pos"] > 0:
            pass
        else:
            inference_state["gen_text"][:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
            if self.predict_user_text:
                inference_state["gen_asr"][:, 0] = ans["asr_logits"][:, -1].argmax(dim=-1)
            # Function channel prediction (same logits, different role)
            inference_state["gen_function"][:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)

        return ans, inference_state

    def _maybe_apply_forced_turn_taking(self, t, inference_state, is_prompt_position):
        """Apply forced turn-taking rules based on ASR channel tokens."""
        if not self.cfg.get("force_turn_taking", False):
            return
        
        threshold = self.cfg.get("force_turn_taking_threshold", 40)
        pad_window_steps = self.cfg.get("force_turn_taking_pad_window", 25)
        
        for batch_idx in range(inference_state["B"]):
            if is_prompt_position[batch_idx]:
                continue
            
            lookback_start = max(0, t - threshold)
            agent_text_window = inference_state["gen_text"][batch_idx, lookback_start:t]
            current_asr_token = inference_state["gen_asr"][batch_idx, t]
            
            # ASR EOS or ~1 sec of pad tokens → insert agent BOS if not present in window
            # Skip if we don't have enough tokens at the beginning
            if t < pad_window_steps:
                continue
            
            pad_lookback_start = t - pad_window_steps
            asr_recent_tokens = inference_state["gen_asr"][batch_idx, pad_lookback_start:t]
            has_pad_window = (asr_recent_tokens == self.text_pad_id).all() if len(asr_recent_tokens) > 0 else False
            
            # Require that the pad window starts after a non-pad token
            if has_pad_window and pad_lookback_start > 0:
                token_before_window = inference_state["gen_asr"][batch_idx, pad_lookback_start - 1]
                has_pad_window = (token_before_window != self.text_pad_id) and (token_before_window != self.user_bos_id)
            elif has_pad_window and pad_lookback_start == 0:
                # If the pad window starts at position 0, it doesn't meet the requirement
                has_pad_window = False
            
            if has_pad_window:
                if not (agent_text_window == self.text_bos_id).any():
                    inference_state["gen_text"][batch_idx, t] = self.text_bos_id
            
            # ASR BOS → insert agent EOS if not present in window
            elif current_asr_token == self.user_bos_id:
                if not (agent_text_window == self.text_eos_id).any():
                    inference_state["gen_text"][batch_idx, t] = self.text_eos_id

    def _step_inference(self, t, inference_state, ans, force_bos_positions):
        """Perform inference for one step t in the autoregressive loop."""
        last_emb = self.embed_tokens(inference_state["gen_text"][:, t - 1]) * self.cfg.get("duplex_text_channel_weight", 1.0)
        if self.predict_user_text:
            last_asr_emb = self.embed_asr_tokens(inference_state["gen_asr"][:, t - 1]) * self.cfg.get("duplex_asr_text_weight", 1.0)
            last_emb += last_asr_emb
        
        # Add function calling channel embedding (shared embeddings with text channel)
        last_function_emb = self.embed_tokens(inference_state["gen_function"][:, t - 1]) * self.cfg.get("duplex_function_channel_weight", 1.0)
        last_emb += last_function_emb
        
        if force_bos_positions is not None:
            for batch_idx in range(last_emb.shape[0]):
                if force_bos_positions[batch_idx] == t and not (inference_state["gen_text"][batch_idx, :t] == self.text_bos_id).any():
                    last_emb[batch_idx] = self.embed_tokens(
                        torch.full((1,), fill_value=self.text_bos_id, device=self.device)) * self.cfg.get(
                        "duplex_text_channel_weight", 1.0)

        inference_state["input_embeds"][:, t] += last_emb

        is_prompt_position = inference_state["is_prompt_position_mask"][:, t]

        if inference_state["use_cache"]:
            ans = self(
                inference_state["input_embeds"][:, t: t + 1],
                cache=ans["cache"],
                input_audio_tokens=None,
                seq_mask=None,
                target_text_tokens=None,
            )
            if not is_prompt_position.all():
                generated_tokens = ans["text_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_text"][:, t] = torch.where(is_prompt_position, inference_state["gen_text"][:, t], generated_tokens)
                
                # Function channel: always use separate head
                generated_function_tokens = ans["function_logits"][:, -1].argmax(dim=-1)
                
                # Check if already set (from response injection) - only predict if not already injected
                already_injected = (inference_state["gen_function"][:, t] != self.text_pad_id)
                should_predict = ~is_prompt_position & ~already_injected
                inference_state["gen_function"][:, t] = torch.where(
                    should_predict, 
                    generated_function_tokens,
                    inference_state["gen_function"][:, t]
                )
        else:
            ans = self(
                inference_state["input_embeds"][:, :t + 1],
                cache=None,
                input_audio_tokens=None,
                seq_mask=None,
                target_text_tokens=None,
            )
            if not is_prompt_position.all():
                generated_tokens = ans["text_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_text"][:, t] = torch.where(is_prompt_position, inference_state["gen_text"][:, t], generated_tokens)
                
                # Function channel: always use separate head
                generated_function_tokens = ans["function_logits"][:, -1].argmax(dim=-1)
                
                # Check if already set (from response injection) - only predict if not already injected
                already_injected = (inference_state["gen_function"][:, t] != self.text_pad_id)
                should_predict = ~is_prompt_position & ~already_injected
                inference_state["gen_function"][:, t] = torch.where(
                    should_predict,
                    generated_function_tokens,
                    inference_state["gen_function"][:, t]
                )

        if self.predict_user_text:
            if not is_prompt_position.all():
                generated_asr = ans["asr_logits"][:, -1].argmax(dim=-1)
                inference_state["gen_asr"][:, t] = torch.where(is_prompt_position, inference_state["gen_asr"][:, t], generated_asr)
                self._maybe_apply_forced_turn_taking(t, inference_state, is_prompt_position)

        return ans

    def _prepare_function_responses_for_detection(
        self,
        function_responses: torch.Tensor = None,
        function_response_lengths: torch.Tensor = None,
        function_response_steps: torch.Tensor = None,
    ) -> dict:
        """
        Prepare function responses to be inserted when EOTC is detected.
        
        Note: function_response_steps from ground truth is used to know WHICH response
        to inject (for multiple function calls), but NOT the exact position (model decides).
        
        Returns:
        - response_queue: List of (turn_idx, response_tokens) for each batch item
        """
        if function_responses is None:
            return {"response_queue": []}
        
        B = function_responses.shape[0]
        num_responses = function_responses.shape[1]
        
        sotc_id, eotc_id, eotr_id = self._get_function_call_special_tokens()
        
        # For each batch, prepare queue of responses to inject
        response_queue = []
        for b in range(B):
            batch_queue = []
            for turn_idx in range(num_responses):
                response_length = function_response_lengths[b, turn_idx].item()
                if response_length > 0:
                    # Extract response tokens and prepend EOTR marker
                    response_tokens = function_responses[b, turn_idx, :response_length]
                    response_with_marker = torch.cat([
                        torch.tensor([eotr_id], device=self.device, dtype=torch.long),
                        response_tokens
                    ])
                    batch_queue.append((turn_idx, response_with_marker))
            response_queue.append(batch_queue)
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[Function Response Preparation] Prepared {sum(len(q) for q in response_queue)} responses across {B} batches")
        
        return {
            "response_queue": response_queue,
            "sotc_id": sotc_id,
            "eotc_id": eotc_id,
            "eotr_id": eotr_id,
        }

    def _post_inference(self, inference_state, prompt_token_lens):
        """Post-process inference results and prepare output."""
        gen_text = inference_state["gen_text"]
        gen_asr = inference_state["gen_asr"]
        gen_function = inference_state["gen_function"]
        response_spans = inference_state.get("function_response_spans", None)
        lengths = inference_state["lengths"]
        T_local = inference_state["T_local"]
        T = inference_state["T"]
        B = inference_state["B"]

        # Trim to local expanded length first (mirrors old FSDP padding behavior)
        if self._use_fsdp and inference_state.get("T_expanded_local") is not None:
            local_len = inference_state["T_expanded_local"]
            gen_text = gen_text[:, :local_len]
            gen_function = gen_function[:, :local_len]
            if self.predict_user_text:
                gen_asr = gen_asr[:, :local_len]
            # Keep lengths within local_len for decoding
            lengths = torch.minimum(lengths, torch.tensor(local_len, device=lengths.device))

        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_function = gen_function[:, :T_local]
            if self.predict_user_text:
                gen_asr = gen_asr[:, :T_local]

        if self.predict_user_text:
            gen_text_src = gen_asr
            src_text_cleaned = tokens_to_str(gen_text_src, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id, user_bos_id=self.user_bos_id, eval_text_turn_taking=self.cfg.get("eval_text_turn_taking", True), sil_id=inference_state["sil_id"])
        else:
            gen_text_src = None
            src_text_cleaned = None
        
        if prompt_token_lens is not None:
            max_prompt_len = prompt_token_lens.max().item()
            if max_prompt_len > 0:
                current_T = gen_text.shape[1]
                gen_text_trimmed = torch.zeros(B, current_T - max_prompt_len, device=self.device, dtype=torch.long)
                gen_function_trimmed = torch.zeros(B, current_T - max_prompt_len, device=self.device, dtype=torch.long)
                if self.predict_user_text:
                    gen_asr_trimmed = torch.zeros(B, current_T - max_prompt_len, device=self.device, dtype=torch.long)
                lengths_trimmed = lengths.clone()

                for i, prompt_len in enumerate(prompt_token_lens):
                    prompt_len_val = prompt_len.item()
                    actual_len = lengths[i].item() - prompt_len_val
                    if actual_len > 0:
                        gen_text_trimmed[i, :actual_len] = gen_text[i, prompt_len_val:prompt_len_val + actual_len]
                        gen_function_trimmed[i, :actual_len] = gen_function[i, prompt_len_val:prompt_len_val + actual_len]
                        if self.predict_user_text:
                            gen_asr_trimmed[i, :actual_len] = gen_asr[i, prompt_len_val:prompt_len_val + actual_len]
                    lengths_trimmed[i] = actual_len
                
                gen_text = gen_text_trimmed
                gen_function = gen_function_trimmed
                if self.predict_user_text:
                    gen_asr = gen_asr_trimmed
                    gen_text_src = gen_asr
                lengths = lengths_trimmed
                if response_spans is not None:
                    # Shift response spans to match prompt-trimmed coordinates
                    shifted_spans = [[] for _ in range(B)]
                    for i, spans in enumerate(response_spans):
                        prompt_len_val = prompt_token_lens[i].item()
                        for start, end in spans:
                            start_shifted = start - prompt_len_val
                            end_shifted = end - prompt_len_val
                            if end_shifted <= 0:
                                continue
                            if start_shifted < 0:
                                start_shifted = 0
                            shifted_spans[i].append((start_shifted, end_shifted))
                    response_spans = shifted_spans

        # Build function-channel predictions with prefilled responses masked out
        gen_function_pred = gen_function
        if response_spans is not None:
            gen_function_pred = gen_function.clone()
            for b in range(B):
                for start, end in response_spans[b]:
                    start_idx = max(0, start)
                    end_idx = min(gen_function_pred.shape[1], end)
                    if end_idx > start_idx:
                        gen_function_pred[b, start_idx:end_idx] = self.text_pad_id

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id, user_bos_id=self.user_bos_id, eval_text_turn_taking=self.cfg.get("eval_text_turn_taking", True), sil_id=inference_state["sil_id"]),
            "src_text": src_text_cleaned,
            "tokens_text_src": gen_text_src,
            "tokens_text": gen_text,
            "tokens_function": gen_function,
            "tokens_function_pred": gen_function_pred,
            "tokens_audio": None,
            "tokens_len": lengths,
            "source_audio": inference_state["input_signal"],
            "source_audio_len": inference_state["input_signal_lens"],
        }

        return ans

    def _expand_for_function_calling(
            self,
            inference_state: dict,
            function_call_lengths: torch.Tensor = None,
            function_responses: torch.Tensor = None,
            function_response_lengths: torch.Tensor = None,
            function_response_steps: torch.Tensor = None,
            function_call_steps: torch.Tensor = None,
            prompt_token_lens: torch.Tensor = None,
    ) -> dict:
        """
        Expand and pre-fill sequence for function calling inference.
        
        This helper handles all the complex expansion logic:
        1. Calculate total expansion needed (calls + responses + EOTR)
        2. Pre-allocate expanded tensors
        3. Shift original content and insert:
           - PADDING for call positions (model predicts)
           - PRE-FILLED response tokens (from API)
           - PADDING for EOTR positions (model predicts)
        
        Args:
            inference_state: State from _init_inference
            function_call_lengths: Ground truth call lengths [B, max_calls]
            function_responses: Response tokens [B, max_calls, max_resp_len]
            function_response_lengths: Response lengths [B, max_calls]
            function_response_steps: Insertion positions [B, max_calls]
            prompt_token_lens: System prompt lengths [B]
            
        Returns:
            Updated inference_state with expanded sequences
        """
        
        if self.cfg.get("fc_log", False):
            logging.info(f"╔═══ [FC EXPAND CALLED] ═══╗")
            logging.info(f"║ function_call_lengths: {function_call_lengths.shape if function_call_lengths is not None else 'None'}")
            logging.info(f"║ function_responses: {function_responses.shape if function_responses is not None else 'None'}")
            logging.info(f"║ fc_log: {self.cfg.get('fc_log', 'NOT_SET')}")
            logging.info(f"╚═════════════════════════╝")
        
        B = inference_state["B"]
        T_original = inference_state["T"]
        
        # Calculate total expansion from ground truth
        response_expansion_per_batch = function_response_lengths.sum(dim=1)  # [B]
        num_responses_per_batch = (function_response_lengths > 0).sum(dim=1)  # [B]
        response_expansion_per_batch += num_responses_per_batch  # Add EOTR tokens
        
        if function_call_lengths is not None:
            call_expansion_per_batch = function_call_lengths.sum(dim=1)  # [B]
            num_calls_per_batch = (function_call_lengths > 0).sum(dim=1)  # [B]
            call_expansion_per_batch += num_calls_per_batch * 2  # Add SOTC + EOTC tokens
        else:
            logging.warning("[Expand FC] function_call_lengths not provided, using conservative estimate")
            call_expansion_per_batch = num_responses_per_batch * 50
        
        total_expansion_per_batch = response_expansion_per_batch + call_expansion_per_batch
        total_expansion = total_expansion_per_batch.max().item()
        T_expanded = T_original + total_expansion
        T_expanded_local = T_expanded

        # IMPORTANT: when using FSDP/DDP, all ranks must run the SAME number of steps.
        # If T_expanded differs across ranks, some ranks will finish early and others will hang
        # during FSDP all_gather. Synchronize to the global max T_expanded.
        if dist.is_initialized():
            T_tensor = torch.tensor(
                [T_expanded],
                device=device if "device" in locals() else inference_state["gen_text"].device,
                dtype=torch.int32,
            )
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T_expanded = int(T_tensor.item())
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[Expand FC] T_original={T_original}, expansion={total_expansion}, T_expanded={T_expanded}")
        
        # Pre-allocate expanded tensors
        device = inference_state["gen_text"].device
        dtype = inference_state["gen_text"].dtype
        H = inference_state["input_embeds"].shape[2]
        
        gen_text_expanded = torch.full((B, T_expanded), self.text_pad_id, dtype=dtype, device=device)
        gen_function_expanded = torch.full((B, T_expanded), self.text_pad_id, dtype=dtype, device=device)
        input_embeds_expanded = torch.zeros(B, T_expanded, H, dtype=inference_state["input_embeds"].dtype, device=device)
        is_prompt_expanded = torch.zeros(B, T_expanded, dtype=torch.bool, device=device)
        # [TODO] Is there any need for asr_emb_expanded?
        if inference_state.get("asr_emb") is not None:
            asr_emb = inference_state["asr_emb"]
            asr_emb_expanded = torch.zeros(B, T_expanded, asr_emb.shape[2], dtype=asr_emb.dtype, device=asr_emb.device)
        else:
            asr_emb_expanded = None
        
        if self.predict_user_text and inference_state["gen_asr"] is not None:
            gen_asr_expanded = torch.full((B, T_expanded), self.text_pad_id, dtype=dtype, device=device)
        else:
            gen_asr_expanded = None
        
        # Build insertion plan for each batch
        insertion_events = []
        response_spans = [[] for _ in range(B)]  # expanded-space spans for response tokens
        for b in range(B):
            expansion_plan = []
            if function_call_lengths is not None:
                num_calls = function_call_lengths.shape[1] if len(function_call_lengths.shape) > 1 else 1
            else:
                num_calls = function_response_lengths.shape[1] if len(function_response_lengths.shape) > 1 else 1
            
            prompt_offset = 0
            if prompt_token_lens is not None:
                prompt_offset = prompt_token_lens[b].item()
            
            if b == 0 and self.cfg.get("fc_log", False):
                logging.info(f"[FC Expand] Building expansion plan for batch {b}")
                logging.info(f"[FC Expand] Prompt offset: {prompt_offset}, Num calls: {num_calls}")
                if function_call_lengths is not None:
                    logging.info(f"[FC Expand] function_call_lengths provided: shape={function_call_lengths.shape}")
                else:
                    logging.info(f"[FC Expand] WARNING: function_call_lengths is None - call regions won't be padded!")
            
            for call_idx in range(num_calls):
                # Note: in the dataset both function call and function responses has the same insertion step because it use the same start time in seconds
                response_len = function_response_lengths[b, call_idx].item() if len(function_response_lengths.shape) > 1 else function_response_lengths[b].item()
                insertion_step = function_call_steps[b, call_idx].item() if len(function_call_steps.shape) > 1 else function_call_steps[b].item()
                
                if insertion_step < 0:
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] Skipping call {call_idx}: insertion_step={insertion_step} < 0")
                    continue
                
                insertion_step_with_prompt = insertion_step + prompt_offset
                
                # Track position as we add events (call -> response -> eotr)
                current_insertion_pos = insertion_step_with_prompt
                
                call_len = function_call_lengths[b, call_idx].item() if len(function_call_lengths.shape) > 1 else 0
                if self.cfg.get("fc_log", False):
                    logging.info(f"[FC Expand DEBUG] Batch {b}, Call {call_idx}: call_len={call_len}, function_call_lengths shape={function_call_lengths.shape if function_call_lengths is not None else 'None'}")
                
                # Reserve PADDING for call
                if function_call_lengths is not None and call_len > 0:
                    call_space = call_len + 2  # SOTC + call + EOTC
                    expansion_plan.append((current_insertion_pos, call_space, None, 'call'))
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] Call {call_idx}: position={current_insertion_pos}, space={call_space} (call_len={call_len})")
                else:
                    if self.cfg.get("fc_log", False):
                        logging.error(f"[FC Expand BUG] Batch {b}, Call {call_idx}: NO SPACE RESERVED FOR CALL! call_len={call_len}, function_call_lengths={'None' if function_call_lengths is None else 'provided'}")
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.error(f"[FC Expand BUG] This means response tokens will be placed at position {current_insertion_pos} with NO call tokens before them!")
                    call_space = 0
                
                # Pre-fill response
                if response_len > 0:
                    # Remove padding tokens
                    if len(function_responses.shape) == 3:
                        response_tokens = function_responses[b, call_idx, :response_len]
                    else:
                        response_tokens = function_responses[b, :response_len]
                    
                    expansion_plan.append((current_insertion_pos, None, response_tokens, 'response'))
                    if b == 0 and self.cfg.get("fc_log", False):
                        response_text = self.tokenizer.ids_to_text(response_tokens.tolist())
                        logging.info(f"[FC Expand] Response {call_idx}: position={current_insertion_pos}, length={response_len}")
                        logging.info(f"[FC Expand] Response text: {response_text[:100]}")
                    
                    # Reserve PADDING for EOTR
                    expansion_plan.append((current_insertion_pos, 1, None, 'eotr'))
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] EOTR {call_idx}: position={current_insertion_pos}, space=1")
            
            insertion_events.append(expansion_plan)
            
            if b == 0 and self.cfg.get("fc_log", False):
                logging.info(f"[FC Expand] Batch {b} expansion plan: {len(expansion_plan)} events")
        
        # Expand each batch item by shifting and inserting
        for b in range(B):
            original_text = inference_state["gen_text"][b]
            original_function = inference_state["gen_function"][b]
            original_embeds = inference_state["input_embeds"][b]
            original_prompt_mask = inference_state["is_prompt_position_mask"][b]
            if gen_asr_expanded is not None:
                original_asr = inference_state["gen_asr"][b]
            if asr_emb_expanded is not None:
                original_asr_emb = inference_state["asr_emb"][b]
                if prompt_token_lens is not None:
                    prompt_len = prompt_token_lens[b].item()
                    if prompt_len > 0:
                        aligned_asr = torch.zeros(
                            T_original,
                            original_asr_emb.shape[1],
                            device=original_asr_emb.device,
                            dtype=original_asr_emb.dtype,
                        )
                        copy_len = min(original_asr_emb.shape[0], T_original - prompt_len)
                        if copy_len > 0:
                            aligned_asr[prompt_len:prompt_len + copy_len] = original_asr_emb[:copy_len]
                        original_asr_emb = aligned_asr
            
            current_pos = 0
            offset = 0
            
            if b == 0 and self.cfg.get("fc_log", False):
                logging.info(f"[FC Expand] Batch {b}: Processing {len(insertion_events[b])} insertion events")
            
            for event_idx, (insert_pos, space, tokens, event_type) in enumerate(insertion_events[b]):
                # Copy segment before insertion
                segment_length = insert_pos - current_pos
                if segment_length > 0:
                    new_start = current_pos + offset
                    new_end = new_start + segment_length
                    gen_text_expanded[b, new_start:new_end] = original_text[current_pos:insert_pos]
                    gen_function_expanded[b, new_start:new_end] = original_function[current_pos:insert_pos]
                    input_embeds_expanded[b, new_start:new_end] = original_embeds[current_pos:insert_pos]
                    is_prompt_expanded[b, new_start:new_end] = original_prompt_mask[current_pos:insert_pos]
                    if gen_asr_expanded is not None:
                        gen_asr_expanded[b, new_start:new_end] = original_asr[current_pos:insert_pos]
                    if asr_emb_expanded is not None:
                        asr_emb_expanded[b, new_start:new_end] = original_asr_emb[current_pos:insert_pos]
                    
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] Event {event_idx} ({event_type}): Copied original[{current_pos}:{insert_pos}] → expanded[{new_start}:{new_end}]")
                
                # Insert based on event type
                insertion_start = insert_pos + offset  # Position in expanded space
                if event_type == 'call':
                    # gen_function_expanded already initialized to PAD (for model to predict)
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] Event {event_idx} (call): Reserved PAD region expanded[{insertion_start}:{insertion_start + space}] for model to predict call")
                    try:
                        silence_embeds = self._get_silence_embeddings_from_template(
                            space,
                            device=input_embeds_expanded.device,
                            dtype=input_embeds_expanded.dtype,
                        )
                    except Exception:
                        silence_embeds = torch.zeros(
                            space,
                            input_embeds_expanded.shape[2],
                            device=input_embeds_expanded.device,
                            dtype=input_embeds_expanded.dtype,
                        )
                    silence_embeds = silence_embeds * self.cfg.get("duplex_user_channel_weight", 1.0)
                    input_embeds_expanded[b, insertion_start:insertion_start + space] = silence_embeds
                    if asr_emb_expanded is not None:
                        try:
                            silence_asr = self._get_silence_asr_embeddings_from_template(
                                space,
                                device=asr_emb_expanded.device,
                                dtype=asr_emb_expanded.dtype,
                            )
                        except Exception:
                            silence_asr = torch.zeros(
                                space,
                                asr_emb_expanded.shape[2],
                                device=asr_emb_expanded.device,
                                dtype=asr_emb_expanded.dtype,
                            )
                        asr_emb_expanded[b, insertion_start:insertion_start + space] = silence_asr
                    offset += space
                elif event_type == 'response':
                    response_len = len(tokens)
                    response_start = insertion_start
                    response_end = response_start + response_len
                    gen_function_expanded[b, response_start:response_end] = tokens  # Pre-fill response
                    try:
                        silence_embeds = self._get_silence_embeddings_from_template(
                            response_len,
                            device=input_embeds_expanded.device,
                            dtype=input_embeds_expanded.dtype,
                        )
                    except Exception:
                        silence_embeds = torch.zeros(
                            response_len,
                            input_embeds_expanded.shape[2],
                            device=input_embeds_expanded.device,
                            dtype=input_embeds_expanded.dtype,
                        )
                    silence_embeds = silence_embeds * self.cfg.get("duplex_user_channel_weight", 1.0)
                    input_embeds_expanded[b, response_start:response_end] = silence_embeds
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] Event {event_idx} (response): Pre-filled response expanded[{response_start}:{response_end}] with {response_len} tokens")
                        response_text = self.tokenizer.ids_to_text(tokens.tolist())
                        logging.info(f"[FC Expand] Response text: {response_text[:100]}")
                    response_spans[b].append((response_start, response_end))
                    if asr_emb_expanded is not None:
                        try:
                            silence_asr = self._get_silence_asr_embeddings_from_template(
                                response_len,
                                device=asr_emb_expanded.device,
                                dtype=asr_emb_expanded.dtype,
                            )
                        except Exception:
                            silence_asr = torch.zeros(
                                response_len,
                                asr_emb_expanded.shape[2],
                                device=asr_emb_expanded.device,
                                dtype=asr_emb_expanded.dtype,
                            )
                        asr_emb_expanded[b, response_start:response_end] = silence_asr
                    offset += response_len
                elif event_type == 'eotr':
                    # gen_function_expanded already initialized to PAD (for model to predict)
                    if b == 0 and self.cfg.get("fc_log", False):
                        logging.info(f"[FC Expand] Event {event_idx} (eotr): Reserved PAD region expanded[{insertion_start}:{insertion_start + space}] for model to predict EOTR")
                    try:
                        silence_embeds = self._get_silence_embeddings_from_template(
                            space,
                            device=input_embeds_expanded.device,
                            dtype=input_embeds_expanded.dtype,
                        )
                    except Exception:
                        silence_embeds = torch.zeros(
                            space,
                            input_embeds_expanded.shape[2],
                            device=input_embeds_expanded.device,
                            dtype=input_embeds_expanded.dtype,
                        )
                    silence_embeds = silence_embeds * self.cfg.get("duplex_user_channel_weight", 1.0)
                    input_embeds_expanded[b, insertion_start:insertion_start + space] = silence_embeds
                    if asr_emb_expanded is not None:
                        try:
                            silence_asr = self._get_silence_asr_embeddings_from_template(
                                space,
                                device=asr_emb_expanded.device,
                                dtype=asr_emb_expanded.dtype,
                            )
                        except Exception:
                            silence_asr = torch.zeros(
                                space,
                                asr_emb_expanded.shape[2],
                                device=asr_emb_expanded.device,
                                dtype=asr_emb_expanded.dtype,
                            )
                        asr_emb_expanded[b, insertion_start:insertion_start + space] = silence_asr
                    offset += space
                
                current_pos = insert_pos
            
            # Copy remaining content after last insertion
            remaining_length = T_original - current_pos
            if remaining_length > 0:
                new_start = current_pos + offset
                new_end = new_start + remaining_length
                gen_text_expanded[b, new_start:new_end] = original_text[current_pos:T_original]
                gen_function_expanded[b, new_start:new_end] = original_function[current_pos:T_original]
                input_embeds_expanded[b, new_start:new_end] = original_embeds[current_pos:T_original]
                is_prompt_expanded[b, new_start:new_end] = original_prompt_mask[current_pos:T_original]
                if gen_asr_expanded is not None:
                    gen_asr_expanded[b, new_start:new_end] = original_asr[current_pos:T_original]
                if asr_emb_expanded is not None:
                    asr_emb_expanded[b, new_start:new_end] = original_asr_emb[current_pos:T_original]
                
                if b == 0 and self.cfg.get("fc_log", False):
                    logging.info(f"[FC Expand] Copied remaining original[{current_pos}:{T_original}] → expanded[{new_start}:{new_end}]")
            
            if b == 0 and self.cfg.get("fc_log", False):
                logging.info(f"[FC Expand] Batch {b}: Final expanded length = {T_expanded}, offset = {offset}")
                # Show what's in gen_function_expanded
                non_pad_positions = (gen_function_expanded[b] != self.text_pad_id).nonzero(as_tuple=True)[0]
                if len(non_pad_positions) > 0:
                    logging.info(f"[FC Expand] Non-PAD positions in gen_function_expanded: {non_pad_positions.tolist()[:20]}")
                    for pos in non_pad_positions[:10]:
                        token_id = gen_function_expanded[b, pos].item()
                        try:
                            token_text = self.tokenizer.ids_to_text([token_id])
                        except:
                            token_text = f"<ID:{token_id}>"
                        logging.info(f"[FC Expand]   Pos {pos}: {token_text} (id={token_id})")
        
        # Update inference state with expanded tensors
        inference_state["gen_text"] = gen_text_expanded
        inference_state["gen_function"] = gen_function_expanded
        inference_state["input_embeds"] = input_embeds_expanded
        inference_state["is_prompt_position_mask"] = is_prompt_expanded
        if gen_asr_expanded is not None:
            inference_state["gen_asr"] = gen_asr_expanded
        if asr_emb_expanded is not None:
            inference_state["asr_emb"] = asr_emb_expanded
        inference_state["T"] = T_expanded
        inference_state["T_expanded_local"] = T_expanded_local
        inference_state["function_response_spans"] = response_spans
        
        if self.cfg.get("debug_fc", False):
            logging.info(f"[FC Expand] ===== EXPANSION SUMMARY =====")
            logging.info(f"[FC Expand] Original length: {T_original} → Expanded length: {T_expanded}")
            for b in range(min(B, 2)):
                non_pad_count = (gen_function_expanded[b] != self.text_pad_id).sum().item()
                logging.info(f"[FC Expand] Batch {b}: {non_pad_count} non-PAD tokens in gen_function_expanded")
                
                # Decode the entire gen_function channel to see what's in it
                all_tokens = gen_function_expanded[b].cpu().tolist()
                # Remove trailing PADs for cleaner display
                last_non_pad = T_expanded - 1
                while last_non_pad >= 0 and all_tokens[last_non_pad] == self.text_pad_id:
                    last_non_pad -= 1
                
                if last_non_pad >= 0:
                    relevant_tokens = all_tokens[:last_non_pad + 1]
                    decoded_text = self.tokenizer.ids_to_text([t for t in relevant_tokens if t != self.text_pad_id])
                    logging.info(f"[FC Expand] Batch {b} gen_function (non-PAD decoded): {decoded_text[:200]}")

        if self.cfg.get("debug_fc", False):
            chunk_size = int(self.cfg.get("debug_fc_chunk_size", 200))
            for b in range(B):
                self._log_long_list(
                    f"[FC Expand] FULL gen_function_expanded[{b}] len={gen_function_expanded.shape[1]}",
                    gen_function_expanded[b].tolist(),
                    chunk_size,
                )
                logging.info(f"[FC Expand] Insertion plan (original space) batch {b}: {insertion_events[b]}")
                logging.info(f"[FC Expand] Response spans (expanded space) batch {b}: {response_spans[b]}")
        return inference_state

    @torch.no_grad()
    def offline_inference(
            self,
            input_signal: torch.Tensor,
            input_signal_lens: torch.Tensor,
            decode_audio: bool = True,
            input_pad_len: int = 0,
            force_bos_positions=None,
            prompt_tokens: torch.Tensor = None,
            prompt_token_lens: torch.Tensor = None,
            sample_id=None,
            function_calls: torch.Tensor = None,  # Not used, only lengths needed
            function_call_lengths: torch.Tensor = None,
            function_responses: torch.Tensor = None,
            function_response_lengths: torch.Tensor = None,
            function_response_steps: torch.Tensor = None,
            function_call_steps: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction (simple loop like original).
        
        For function calling: expansion/pre-filling is handled by helper function.
        """
        if self.cfg.get("fc_log", False):
            logging.info(f"╔═══ [OFFLINE_INFERENCE CALLED] ═══╗")
            logging.info(f"║ function_responses: {function_responses.shape if function_responses is not None else 'None'}")
            logging.info(f"║ function_response_lengths: {function_response_lengths.shape if function_response_lengths is not None else 'None'}")
            logging.info(f"║ function_call_steps: {function_call_steps.shape if function_call_steps is not None else 'None'}")
            logging.info(f"║ Will call _expand_for_function_calling: {function_responses is not None and function_response_lengths is not None}")
            logging.info(f"╚══════════════════════════════════╝")
        
        # Initialize inference state (basic setup)
        inference_state = self._init_inference(
            input_signal, input_signal_lens, input_pad_len,
            force_bos_positions, prompt_tokens, prompt_token_lens, sample_id
        )
        
        # Expand for function calling if needed
        if function_responses is not None and function_response_lengths is not None:
            inference_state = self._expand_for_function_calling(
                inference_state,
                function_call_lengths=function_call_lengths,
                function_responses=function_responses,
                function_response_lengths=function_response_lengths,
                function_response_steps=function_response_steps,
                function_call_steps=function_call_steps,
                prompt_token_lens=prompt_token_lens,
            )
        
        # Simple generation loop (same as original)
        ans, inference_state = self._step_zero(inference_state)
        
        for t in range(1, inference_state["T"]):
            ans = self._step_inference(t, inference_state, ans, force_bos_positions)
        
        return self._post_inference(inference_state, prompt_token_lens)

    def _extract_online_audio_window(
            self,
            input_signal: torch.Tensor,
            input_signal_lens: torch.Tensor,
            audio_frame_idx: int,
            window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract audio window for online inference at given frame index.
        
        Window: [max(0, audio_frame_idx - window_size + 1) : audio_frame_idx + 1]
        
        Args:
            input_signal: Full audio signal (B, audio_len)
            input_signal_lens: Lengths of each audio in batch (B,)
            audio_frame_idx: Current frame index
            window_size: Window size in frames
            
        Returns:
            audio_window: (B, window_size * samples_per_frame)
            audio_window_lens: (B,)
        """
        B = input_signal.shape[0]
        frame_length = 0.08  # 80ms per frame
        samples_per_frame = int(frame_length * self.source_sample_rate)
        
        # Calculate window boundaries in frames
        window_start_frame = max(0, audio_frame_idx - window_size + 1)
        window_end_frame = audio_frame_idx + 1
        
        # Convert to sample indices
        window_start_sample = window_start_frame * samples_per_frame
        window_end_sample = window_end_frame * samples_per_frame
        
        # Prepare output tensors
        audio_window = torch.zeros(B, window_size * samples_per_frame, 
                                   device=input_signal.device, dtype=input_signal.dtype)
        audio_window_lens = torch.zeros(B, dtype=torch.long, device=input_signal.device)
        
        # Extract window for each batch item
        for i in range(B):
            actual_end = min(window_end_sample, input_signal_lens[i].item())
            actual_start = min(window_start_sample, actual_end)
            actual_len = actual_end - actual_start
            
            if actual_len > 0:
                audio_window[i, :actual_len] = input_signal[i, actual_start:actual_end]
                audio_window_lens[i] = actual_len
        
        # Only return the valid portion of audio_window (up to max audio_window_lens)
        max_len = audio_window_lens.max().item()
        audio_window = audio_window[:, :max_len]
        
        return audio_window, audio_window_lens

    @torch.no_grad()
    def online_inference(
            self,
            input_signal: torch.Tensor,
            input_signal_lens: torch.Tensor,
            decode_audio: bool = True,
            input_pad_len: int = 0,
            force_bos_positions=None,
            prompt_tokens: torch.Tensor = None,
            prompt_token_lens: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Online inference simulating real-time microphone input with sliding window.
        
        For each time step t:
        - Extract audio window: [max(0, t-window_size+1) : t+1] frames
        - Pass window through encoder (causal: cannot see frames beyond t)
        - Use only the LAST frame's embedding for LLM prediction
        - Stride is always 1 frame
        
        This function maximally reuses existing helper functions:
        - _init_inference: initialize all states (prompts, cache, buffers)
        - _step_zero: process first step
        - _step_inference: process subsequent steps
        - _post_inference: post-process results
        
        The key trick is dynamically updating inference_state["input_embeds"] 
        and inference_state["asr_emb"] at each step with the last frame from 
        the sliding window.
        
        Args:
            Same as offline_inference
            
        Returns:
            Same format as offline_inference
        """
        # Get window size from config (default 70 frames = 5.6 seconds)
        window_size = self.cfg.get("online_window_size", 70)
        
        # Step 1: Initialize inference state using existing function
        # This handles prompts, cache setup, buffer allocation, etc.
        inference_state = self._init_inference(
            input_signal, input_signal_lens, input_pad_len,
            force_bos_positions, prompt_tokens, prompt_token_lens
        )
        # Reset 'input_embeds' to zeros to ensure it starts fresh in online mode
        if "input_embeds" in inference_state:
            inference_state["input_embeds"] = torch.zeros_like(inference_state["input_embeds"])
        
        # Get start position (accounts for prompts if present)
        start_gen_pos = inference_state["start_gen_pos"]
        
        # Step 2: Process first step (t=0)
        if start_gen_pos > 0:
            # We have prompts: use standard _step_zero
            ans = self._step_zero(inference_state)
        else:
            # No prompts: extract first audio window and encode
            audio_window, audio_window_lens = self._extract_online_audio_window(
                input_signal, input_signal_lens, 0, window_size
            )
            
            # Encode window (causal: only sees frames [0:1])
            source_encoded_window, _, asr_emb_window = self.perception(
                input_signal=audio_window,
                input_signal_length=audio_window_lens,
                return_encoder_emb=True,
            )
            
            # Update inference_state with LAST frame's embedding
            inference_state["input_embeds"][:, :1, :] = source_encoded_window[:, -1:, :] * self.cfg.get("duplex_user_channel_weight", 1.0)
            
            # Now call standard _step_zero
            ans = self._step_zero(inference_state)
        
        # Step 3: Main autoregressive loop (causal mode)
        for t in range(1, inference_state["T"]):
            audio_frame_idx = t - start_gen_pos
            
            if audio_frame_idx < 0:
                # Still in prompt region: use standard inference
                ans = self._step_inference(t, inference_state, ans, force_bos_positions)
            else:
                # In audio region: extract window, encode, update state
                audio_window, audio_window_lens = self._extract_online_audio_window(
                    input_signal, input_signal_lens, audio_frame_idx, window_size
                )
                
                # Encode window (causal: only sees frames [max(0,t-69):t+1])
                source_encoded_window, _, asr_emb_window = self.perception(
                    input_signal=audio_window,
                    input_signal_length=audio_window_lens,
                    return_encoder_emb=True,
                )
                
                # Update inference_state with LAST frame's embedding
                inference_state["input_embeds"][:, t:t+1, :] = source_encoded_window[:, -1:, :] * self.cfg.get("duplex_user_channel_weight", 1.0)
                
                # Call standard _step_inference
                ans = self._step_inference(t, inference_state, ans, force_bos_positions)
        
        # Step 4: Post-process using existing function
        return self._post_inference(inference_state, prompt_token_lens)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Shard(1),),
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                }

                attn_layer = transformer_block.self_attn

                try:
                    config = self.llm.config

                    num_attention_heads = getattr(config, 'num_attention_heads', None)
                    num_key_value_heads = getattr(config, 'num_key_value_heads', None)
                    hidden_size = getattr(config, 'hidden_size', None)

                    if all([num_attention_heads, num_key_value_heads, hidden_size]):
                        for attr_name, val in [("num_attention_heads", num_attention_heads),
                                               ("num_key_value_heads", num_key_value_heads),
                                               ("hidden_size", hidden_size)]:
                            if val % tp_mesh.size() != 0:
                                logging.warning(
                                    f"config.{attr_name}={val} is not divisible by {tp_mesh.size()=}: "
                                    f"set a different tensor parallelism size to avoid errors."
                                )

                        if hasattr(attn_layer, 'num_heads'):
                            attn_layer.num_heads = num_attention_heads // tp_mesh.size()
                        elif hasattr(attn_layer, 'num_attention_heads'):
                            attn_layer.num_attention_heads = num_attention_heads // tp_mesh.size()

                        if hasattr(attn_layer, 'num_key_value_heads'):
                            attn_layer.num_key_value_heads = num_key_value_heads // tp_mesh.size()

                        if hasattr(attn_layer, 'hidden_size'):
                            attn_layer.hidden_size = hidden_size // tp_mesh.size()

                        logging.info(f"Configured tensor parallel for attention: "
                                     f"heads={num_attention_heads // tp_mesh.size()}, "
                                     f"kv_heads={num_key_value_heads // tp_mesh.size()}, "
                                     f"hidden_size={hidden_size // tp_mesh.size()}")
                    else:
                        raise AttributeError("Required config attributes not found")

                except Exception as e:
                    logging.warning(f"Failed to configure tensor parallel using config: {e}")
                    logging.warning("Falling back to attention layer attributes...")

                    try:
                        for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                            if hasattr(attn_layer, attr):
                                val = getattr(attn_layer, attr)
                                if val % tp_mesh.size() != 0:
                                    logging.warning(
                                        f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                                        f"set a different tensor parallelism size to avoid errors."
                                )
                                setattr(attn_layer, attr, val // tp_mesh.size())
                    except Exception as fallback_e:
                        logging.warning(f"Both config and fallback methods failed: {fallback_e}")
                        logging.warning("Skipping tensor parallel configuration for this attention layer")

            for m in (self.lm_head,):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            if self.predict_user_text:
                self.asr_head = fully_shard(self.asr_head, **fsdp_config)
                self.embed_asr_tokens = fully_shard(self.embed_asr_tokens, **fsdp_config)
            # Function calling uses shared embeddings, only shard the head
            self.function_head = fully_shard(self.function_head, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)

