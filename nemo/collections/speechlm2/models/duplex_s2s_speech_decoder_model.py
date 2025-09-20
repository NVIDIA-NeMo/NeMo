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
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
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
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TurnTakingMetrics
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
    Safe for GPU execution.
    """
    B, T = tokens.shape
    tokens = tokens.clone()
    device = tokens.device

    # Find all EOS positions
    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    # Flattened indices of EOS tokens
    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]  # [N]
    eos_pos = eos_indices[:, 1]  # [N]
    new_pos = eos_pos + shift  # [N]

    # Filter: new position must be in bounds and not overwrite EOS or PAD
    valid = (new_pos < T)
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        # Now, check overwrite safety in new positions
        target_vals = tokens[b_idx, new_pos]
        safe = (target_vals != eos_token_id)

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            # Move EOS token: clear original, set new
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)


        # 条件加载audio codec
        if self.cfg.audio_loss_weight > 0:
            setup_audio_codec(self)
            self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
            self._num_codebooks = self.audio_codec.vector_quantizer.num_groups
            
            # to be able to load older model
            if self.cfg.get("custom_codebook_size", None):
                self._codebook_size = self.cfg.get("custom_codebook_size")
                
            # cached for quicker audio decoding
            self.register_buffer(
                "_control_codes",
                torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
            )
        else:
        
            self._codebook_size = 2048
            self._num_codebooks = 8
            self.audio_codec = None

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        
        # Load LLM first
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        
        # Handle different model types with all their specific configurations
        if 'Nemotron' in self.cfg.pretrained_llm:
            # ====== NEMOTRON-SPECIFIC HANDLING ======
            # Tokenizer with override tokens from config
            # self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True, **self.cfg.get("override_tokens", {}))
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token = '<SPECIAL_12>'


            self.llm = getattr(llm, self.cfg.get("base_model_name", "model"))



            self.lm_head = llm.lm_head


            embed_tokens_name = self.cfg.get("embed_tokens_name", "embed_tokens")

            self.embed_tokens = getattr(self.llm, embed_tokens_name)


            delattr(self.llm, embed_tokens_name)
            
        elif 'Qwen2.5' in self.cfg.pretrained_llm:
            # ====== QWEN2.5-SPECIFIC HANDLING ======
            # Tokenizer with special token setup
            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'
            
            # Standard model access
            self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
            self.lm_head = llm.lm_head
            # Note: we have to "move out" the token embedding outside of LLM to avoid
            #       messing up FSDP/TP hooks.
            self.embed_tokens = self.llm.embed_tokens
            del self.llm.embed_tokens
            
        else:

            self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
            
            # Standard model access
            self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
            self.lm_head = llm.lm_head
            # Note: we have to "move out" the token embedding outside of LLM to avoid
            #       messing up FSDP/TP hooks.
            self.embed_tokens = self.llm.embed_tokens
            del self.llm.embed_tokens


        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder(self)

        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [
                (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                for subword in llm_tokenizer_vocab_items
            ]
        if self.cfg.audio_loss_weight > 0:
            self.speech_generation = TransformerARSpeechDecoder(
                speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
                lantent_dim=self.llm.config.hidden_size,
                num_audio_codebooks=self._num_codebooks,
                num_audio_tokens_per_codebook=self.speech_vocab_size,
                llm_tokenizer_vocab_items=llm_tokenizer_vocab_items,
            )

        if self.cfg.get("pretrained_s2s_model", None):
            self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        # load pretrained TTS model
        if self.cfg.get("pretrained_tts", None):
            self.init_speech_generation_from_tts_checkpoint(self.cfg.pretrained_tts)

        # load speech decoder/speech generation module from another checkpoint
        if self.cfg.get("pretrained_tts_from_s2s", None):
            self.init_speech_generation_from_another_s2s_checkpoint(self.cfg.pretrained_tts_from_s2s)


        self._use_fsdp = False
        self._use_tp = False

        # Cache for noise file names to avoid repeated glob operations
        if self.cfg.get('noise_prob', None) and self.cfg.noise_prob > 0:
            self._noise_files_cache = {}

    def init_speech_generation_from_tts_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_speech_generation_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # filter keys to keep only speech generation keys and also
            checkpoint_state = {
                k.replace("model.speech_decoder.", "").replace("speech_generation.", ""): v
                for k, v in checkpoint_state.items()
                if "model.speech_decoder." in k or "speech_generation." in k
            }
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_delay_id", None):
            return self.cfg.get("custom_speech_delay_id")
        return self._codebook_size + 2

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
        modality_adapter_emb=None,
        asr_emb=None,
        speaker_encoder_emb=None,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """
        # Handle different cache parameter names for different models
        if 'Nemotron' in self.cfg.pretrained_llm:
            # Nemotron uses cache_params instead of past_key_values
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
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        ans = {"text_logits": text_logits}


        if self.cfg.audio_loss_weight > 0:
            if seq_mask is not None:
                # This is training Mode
                seq_mask = seq_mask[:, :, -1].reshape(seq_mask.size(0), seq_mask.size(1))
                # disable cache in training mode
                if self.speech_generation.use_input_cache:
                    self.speech_generation.reset_input_and_kv_cache(use_cache=False)


            if self.speech_generation.use_input_cache and not self.training:
                if self.cfg.get("inference_pad_boost", None):
                    text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
                if self.cfg.get("inference_bos_boost", None):
                    text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost  
                if self.cfg.get("inference_eos_boost", None):
                    text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost

                target_text_tokens = torch.argmax(text_logits, dim=-1).view(B, T).contiguous()

                if self.cfg.get('convert_pad_to_extra_id_on_speech_decoder', None):
                    target_text_tokens[target_text_tokens == self.text_pad_id] = self.tokenizer.tokenizer._tokenizer.token_to_id("<|endoftext|>")
            else:

                drop_bos_prob = getattr(self.cfg, "drop_text_bos_prob", 0.0)
                if drop_bos_prob > 0.0:
                    bos_mask = (target_text_tokens == self.text_bos_id)
                    drop_bos_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_bos_prob
                    target_text_tokens = torch.where(bos_mask & drop_bos_mask, self.text_pad_id, target_text_tokens)

                drop_eos_prob = getattr(self.cfg, "drop_text_eos_prob", 0.0)
                if drop_eos_prob > 0.0:
                    eos_mask = (target_text_tokens == self.text_eos_id)
                    drop_eos_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_eos_prob
                    target_text_tokens = torch.where(eos_mask & drop_eos_mask, self.text_pad_id, target_text_tokens)

            audio_logits, _ = self.speech_generation(
                out['last_hidden_state'].transpose(0, 1), seq_mask,
                input_audio_tokens=input_audio_tokens,
                target_text_tokens=target_text_tokens,
                modality_adapter_emb=modality_adapter_emb,
                asr_emb=asr_emb,
                speaker_encoder_emb=speaker_encoder_emb,
            )
            audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)
            ans["audio_logits"] = audio_logits

        if cache is not None:
            if 'Nemotron' in self.cfg.pretrained_llm:
                # For Nemotron, get cache from the configured cache key
                cache_key = self.cfg.get("cache_key", "cache_params")
                ans["cache"] = getattr(out, cache_key, out.get(cache_key))
            else:
                # Standard cache handling
                ans["cache"] = out["past_key_values"]

        return ans

    def add_noise_to_batch(
        self,
        batch_audio,
        noise_folder,
        snr_db=20,
        noise_prob_scale_user=0.3,
        noise_prob_scale_user_min_snr=-15,
        noise_prob_scale_user_max_snr=24,
        snr_measure_dur=0.0,
        noise_resample=True,
        noise_prob_low_pass=0.1,
    ):

        batch_size, audio_length = batch_audio.shape

        import glob
        import librosa
        import numpy as np
        import soundfile as sf
        from scipy.signal import butter, lfilter

        # Use cached noise file list to avoid repeated glob operations
        if noise_folder not in self._noise_files_cache:
            noise_files = [f for f in glob.glob(noise_folder + "/*.wav")]
            if not noise_files:
                raise ValueError(f"No noise files found in {noise_folder}")
            self._noise_files_cache[noise_folder] = noise_files
        else:
            noise_files = self._noise_files_cache[noise_folder]

        for i in range(batch_size):

            def get_scale_factor(signal, noise, snr_db):
                if snr_measure_dur > 0:
                    signal = signal[: int(snr_measure_dur * self.source_sample_rate)]
                    noise = noise[: int(snr_measure_dur * self.source_sample_rate)]
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(noise_prob_scale_user_min_snr, noise_prob_scale_user_max_snr),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor

            def get_noise(noise_files):

                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')

                # resample noise from sr to self.cfg.data.train_ds.sample_rate
                if noise_resample and sr != self.source_sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.source_sample_rate)

                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)

                noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)
                scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
                noise_tensor = noise_tensor * scaling_factor
                return noise_tensor

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise3 = get_noise(noise_files)
            noise = torch.cat([noise, noise2, noise3], axis=0)

            if noise.size(0) < audio_length:
                repeat_times = (audio_length // noise.size(0)) + 1
                # For a 1D tensor, we want to repeat its elements.
                # If noise has other dimensions, adjust the repeat_times_tuple accordingly.
                # e.g., if noise is (C, L), and we want to repeat along L,
                # repeat_times_tuple = (1, repeat_times)
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                # If noise is a PyTorch tensor
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                # Or if noise was originally a list/numpy array and you want to keep Python's random
                # start_idx = random.randint(0, len(noise) - audio_length)
                noise = noise[start_idx : start_idx + audio_length]

            # Function to create a low-pass filter
            def butter_lowpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            # Function to apply the low-pass filter to data (tmp impl on cpu)
            def lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                b = torch.tensor(b, dtype=torch.float32).cuda()
                a = torch.tensor(a, dtype=torch.float32).cuda()
                # Apply the filter using lfilter function from scipy..numpysig.numpynal (CPU)
                y_cpu = lfilter(b.cpu().numpy(), a.cpu().numpy(), data.cpu().numpy())
                # Convert the filtered data back to torch tensor and move to GPU.numpy
                y_gpu = torch.tensor(y_cpu, dtype=torch.float32).cuda()
                return y_gpu

            if random.random() < noise_prob_low_pass:
                # Define the desired cutoff frequency (in Hz)
                cutoff = 1000.0
                # Apply low-pass filter to the WAV data
                noise = lowpass_filter(noise, cutoff, self.source_sample_rate)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio

    def prepare_inputs(self, batch: dict):

        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )
     
        if self.cfg.get('noise_prob', None) and self.cfg.noise_prob > 0:

            if (
                self.training
                and batch["formatter"][0] != 's2s_duplex_overlap_as_s2s_duplex'
                and random.random() < self.cfg.noise_prob
            ):
                batch["source_audio"] = self.add_noise_to_batch(
                    batch["source_audio"],
                    os.path.join(self.cfg.noise_file_path, "*"),
                    snr_db=random.randint(20, 50),          #   noise_min_snr = 20 and noise_max_snr = 50
                    noise_prob_scale_user=0.3,
                    noise_prob_scale_user_min_snr=-15,
                    noise_prob_scale_user_max_snr=24,
                    snr_measure_dur=0.0,
                    noise_resample=True,
                    noise_prob_low_pass=0.1,
                )




        if self.cfg.audio_loss_weight > 0 and not self.training:
            speaker_encoder_emb = None
        elif self.cfg.audio_loss_weight > 0 and self.training:
            if self.speech_generation.use_speaker_encoder:
                target_first_turn_audio = batch["target_first_turn_audio"]
                target_first_turn_audio_lens = batch["target_first_turn_audio_lens"]
                speaker_encoder_emb = self.speech_generation.get_speaker_embedding(
                    target_first_turn_audio, target_first_turn_audio_lens, self.target_sample_rate
                )
            else:
                speaker_encoder_emb = None
        else:
            speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        

        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat([
                target_tokens,
                (torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id).to(torch.long),
            ], dim=-1)
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        if self.cfg.audio_loss_weight > 0:

            with fp32_precision(), torch.no_grad():
                target_codes, target_codes_lens = self.audio_codec.encode(
                    audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
                )
            target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

            if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
                if tl < sl:
                    diff = sl - tl
                    source_encoded = source_encoded[:, :tl]
                    asr_emb = asr_emb[:, :tl]
                    target_tokens = target_tokens[:, :tl]
                    torch.clamp_(source_encoded_lens, max=tl)
                else:
                    diff = tl - sl
                    target_codes = target_codes[:, :sl]
                    torch.clamp_(target_codes_lens, max=sl)
                if diff > 2:
                    logging.warning(
                        f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                        f"This may indicate significant desynchronization in longer sessions."
                    )
            btt = target_tokens[..., None]
            target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
            target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

            target_codes = torch.cat([
                torch.full([target_codes.shape[0], 1, target_codes.shape[-1]],
                          fill_value=self.speech_delay_id, device=self.device, dtype=torch.long),
                target_codes[:, :-1],
            ], dim=1)


            if self.advance_text_channel_by:
                pad = torch.full((target_tokens.shape[0], self.advance_text_channel_by),
                               fill_value=self.text_pad_id, device=target_tokens.device, dtype=torch.long)
                target_tokens = torch.cat([target_tokens[:, self.advance_text_channel_by :], pad], dim=-1)

            if self.cfg.get("delay_text_eos_by", None):
                target_tokens = delay_eos(target_tokens, self.text_eos_id, self.text_pad_id, shift=self.cfg.delay_text_eos_by)

            input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
            
            # TP处理
            if self._use_tp:
                tp_world_size = self.device_mesh["tensor_parallel"].size()
                if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                    input_ids = input_ids[:, :-remainder]
                    source_encoded = source_encoded[:, :-remainder]
                    asr_emb = asr_emb[:, :-remainder]

            text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
            text_labels = input_ids[:, 1:, -1]  # (B, T-1)
            audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
            audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

            input_embeds = self.embed_tokens(text_inputs)
            input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

            # 完整的seq_mask
            seq_mask = torch.ones_like(
                torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
                device=self.device, dtype=torch.bool,
            )

            if self.cfg.get("mask_sequence_loss", True):
                for i in range(batch["target_token_lens"].size(0)):
                    speech_end_idx = batch["target_token_lens"][i]
                    seq_mask[i, speech_end_idx:, :] = 0

            # 创建loss scale
            loss_scale = seq_mask.clone().float()
            if self.cfg.get("token_loss_weight"):
                token_weights = self.cfg.token_loss_weight
                pad_weight = token_weights.get("pad", 1.0)
                bos_weight = token_weights.get("bos", 1.0)
                eos_weight = token_weights.get("eos", 1.0)
                text_weight = token_weights.get("text", 1.0)

                text_labels_exp = text_labels.unsqueeze(-1)
                loss_scale[:, :, :1] = torch.where(
                    text_labels_exp == self.text_pad_id, pad_weight,
                    torch.where(
                        text_labels_exp == self.text_bos_id, bos_weight,
                        torch.where(
                            text_labels_exp == self.text_eos_id, eos_weight,
                            text_weight
                        )
                    )
                )
            elif self.cfg.get("scale_loss_by") == 'non_sil_t':
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) != self.text_pad_id,
                    self.cfg.get("scale_loss_mask", self.cfg.get("nonsil_weight", 4.0)),
                    loss_scale[:, :, :1],
                )

            return {
                "input_embeds": input_embeds,
                "input_lens": source_encoded_lens - 1,
                "output_lens": target_codes_lens - 1,
                "text_labels": text_labels,
                "input_audio_tokens": audio_inputs,
                "audio_labels": audio_labels,
                "seq_mask": seq_mask,
                "loss_scale": loss_scale,
                "perception_emb": source_encoded[:, :-1],
                "asr_emb": asr_emb[:, :-1],
                "speaker_encoder_emb": speaker_encoder_emb,
            }
        
        else:

            text_inputs = target_tokens[:, :-1]  # (B, T-1)
            text_labels = target_tokens[:, 1:]   # (B, T-1)
            
            input_embeds = self.embed_tokens(text_inputs)
            input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

            seq_mask = torch.ones_like(text_labels.unsqueeze(-1), device=self.device, dtype=torch.bool)
            
            if self.cfg.get("mask_sequence_loss", True):
                for i in range(batch["target_token_lens"].size(0)):
                    speech_end_idx = batch["target_token_lens"][i]
                    seq_mask[i, speech_end_idx:, :] = 0

            loss_scale = seq_mask.clone().float()
            if self.cfg.get("token_loss_weight"):
                token_weights = self.cfg.token_loss_weight
                pad_weight = token_weights.get("pad", 1.0)
                bos_weight = token_weights.get("bos", 1.0)
                eos_weight = token_weights.get("eos", 1.0)
                text_weight = token_weights.get("text", 1.0)

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
            # elif self.cfg.get("scale_loss_by") == 'non_sil_t':
            #     loss_scale = torch.where(
            #         text_labels.unsqueeze(-1) != self.text_pad_id,
            #         self.cfg.get("scale_loss_mask", self.cfg.get("nonsil_weight", 4.0)),
            #         loss_scale,
            #     )

            return {
                "input_embeds": input_embeds,
                "input_lens": source_encoded_lens - 1,
                "output_lens": source_encoded_lens - 1,  # 使用source长度
                "text_labels": text_labels,
                "loss_scale": loss_scale,
                "seq_mask": seq_mask,
            }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        if self.cfg.audio_loss_weight > 0 and is_frozen(self.speech_generation):
            self.speech_generation.eval()

        res = {"learning_rate": torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)}



        if batch["audio_data"] is not None:
            inputs = self.prepare_inputs(batch["audio_data"])
            if self.cfg.audio_loss_weight > 0:
                forward_outputs = self(
                    inputs["input_embeds"],
                    input_audio_tokens=inputs["input_audio_tokens"],
                    seq_mask=inputs["seq_mask"],
                    target_text_tokens=inputs["text_labels"],
                    modality_adapter_emb=inputs["perception_emb"],
                    asr_emb=inputs["asr_emb"],
                    speaker_encoder_emb=inputs["speaker_encoder_emb"],
                )
            else:
                forward_outputs = self(
                    inputs["input_embeds"],
                )

            num_frames = inputs["input_lens"].sum()

            with loss_parallel():

                text_logits = forward_outputs["text_logits"]


                if self.cfg.get("mask_sequence_loss", True):
                    text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

                text_loss = (
                    torch.nn.functional.cross_entropy(
                        text_logits.flatten(0, 1),
                        inputs["text_labels"].flatten(0, 1),
                        reduction="none",
                    )
                    * inputs["loss_scale"][:, :, 0].flatten(0, 1)
                ).sum(-1) / num_frames

                with torch.no_grad():

                    predicted_tokens = torch.argmax(text_logits, dim=-1)  # (B, T)
                    target_tokens = inputs["text_labels"]  # (B, T)
                    valid_mask = (target_tokens != self.text_pad_id)

                    correct_predictions = (predicted_tokens == target_tokens) & valid_mask


                    if valid_mask.sum() > 0:
                        token_accuracy = correct_predictions.sum().float() / valid_mask.sum().float()
                    else:
                        token_accuracy = torch.tensor(0.0, device=text_logits.device)



                if self.cfg.audio_loss_weight > 0:
                    audio_logits = forward_outputs["audio_logits"]
                    if self.cfg.get("mask_sequence_loss", True):
                        audio_logits = audio_logits * inputs["seq_mask"][:, :, -1].unsqueeze(-1).unsqueeze(-1)


                    audio_loss = (
                        torch.nn.functional.cross_entropy(
                            audio_logits.flatten(0, 2),
                            inputs["audio_labels"].flatten(0, 2),
                            reduction="none",
                        )
                        * inputs["loss_scale"][:, :, 1:].flatten(0, 2)
                    ).sum(-1) / (num_frames * self._num_codebooks)

                    loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss
                else:
                    loss = self.cfg.text_loss_weight * text_loss

                B, T = inputs["input_embeds"].shape[:2]
                ans = {
                    "audio_loss": loss,
                    "audio_to_text_loss": text_loss,
                    "batch": B,
                    "length": T,
                    "token_accuracy": token_accuracy,
                }
  
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

        res["loss"] = (1. - self.cfg.text_to_text_loss_weight) * res.get("audio_loss", 0.0) + \
            self.cfg.text_to_text_loss_weight * res.get("text_to_text_loss", 0.0)
        self.log_dict(res, on_step=True)
        
        return res

    def on_train_epoch_start(self) -> None:
        if self.cfg.audio_loss_weight > 0:
            setup_audio_codec(self)
            if hasattr(self.speech_generation, "use_speaker_encoder") and self.speech_generation.use_speaker_encoder:
                self.speech_generation.setup_speaker_encoder()

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()

        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()
        
        # Initialize turn taking metrics
        self.turn_taking_metrics = TurnTakingMetrics(
            eos_token_id=self.text_eos_id,
            bos_token_id=self.text_bos_id,
            tolerance=13,
            latency_multiplier=0.08
        ).reset()
        
        # Initialize text evaluation metrics
        # self.perplexity = Perplexity(ignore_index=self.text_pad_id).reset()
        # self.validation_loss = ValidationLoss(ignore_index=self.text_pad_id).reset()


    def on_validation_epoch_end(self, prefix="val") -> None:

        bleu = self.bleu.compute()
        for k, m in bleu.items():
            if "-qa" not in k:
                self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        acc_metrics = self.results_logger.compute_and_save()
        
        for name, result_dict in acc_metrics.items():
            self.log(f"{prefix}_{name}_acc", result_dict['acc'].to(self.device), on_epoch=True, sync_dist=True)
            # self.log(f"{prefix}_{name}_empty_rate", result_dict['empty_rate'].to(self.device), on_epoch=True, sync_dist=True)

        # Log turn taking metrics
        turn_taking_metrics = self.turn_taking_metrics.compute()
        for k, m in turn_taking_metrics.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)


        if self.cfg.audio_loss_weight > 0:
            asr_bleu = self.asr_bleu.compute()
            for k, m in asr_bleu.items():
                self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    def validation_step(self, batch: dict, batch_idx: int):

    
        if self.cfg.audio_loss_weight > 0:
            if self.speech_generation.use_speaker_encoder and self.speech_generation.inference_speaker_reference:
                self.speech_generation.update_inference_speaker_embedding(
                    self.speech_generation.inference_speaker_reference
                )

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            dataset_batch = dataset_batch["audio_data"]

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
            )
            
            # Always compute text metrics (BLEU, perplexity, validation loss)
            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            
            # Update turn taking metrics
            if "source_tokens" in dataset_batch and results["tokens_text"] is not None:
                self.turn_taking_metrics.update(
                    name=name,
                    source_tokens=dataset_batch["source_tokens"],
                    pred_tokens=results["tokens_text"]
                )


            if self.cfg.audio_loss_weight == 0:
                # Generate fake pred_audio based on text tokens
                fake_pred_audio, fake_audio_len = self._generate_fake_audio_from_tokens(results["tokens_text"])

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=None,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=fake_pred_audio,
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                )

                continue  # Skip audio-related metrics

            else:
                with fp32_precision():  # resample is fragile to bfloat16 default dtype
                    asr_hyps = self.asr_bleu.update(
                        name=name,
                        refs=dataset_batch["target_texts"],
                        pred_audio=resample(results["audio"], 22050, 16000),
                        pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                    )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                )



    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def on_predict_epoch_start(self) -> None:
        return self.on_train_epoch_start()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int=0):
        batch = batch["audio_data"]

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

        prediction = self.offline_inference(
            batch["source_audio"],
            batch["source_audio_lens"],
            decode_audio=self.cfg.prediction.decode_audio,
            input_pad_len=self.cfg.prediction.max_new_seconds * self.cfg.prediction.input_sample_rate,
            force_bos_positions=force_bos_positions,
        )
        prediction["sample_id"] = batch["sample_id"]
        return prediction

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    def _generate_fake_audio_from_tokens(self, tokens_text: torch.Tensor):
        """
        Generate fake audio based on text tokens for analysis when audio_loss_weight == 0.
        
        Logic:
        - Default value: 0
        - After first text_bos_id: 1
        - After first text_eos_id: back to 0
        - text_pad_id between bos and eos: 0.5
        - text_pad_id elsewhere: 0
        
        Args:
            tokens_text: (batch_size, seq_len) tensor
            
        Returns:
            fake_audio: (batch_size, audio_len) tensor
            audio_lengths: (batch_size,) tensor with audio lengths
        """
        batch_size, seq_len = tokens_text.shape
        token_duration = 0.08  # seconds per token
        samples_per_token = int(token_duration * self.target_sample_rate)
        audio_len = seq_len * samples_per_token
        
        # Initialize fake audio tensor
        fake_audio = torch.zeros(batch_size, audio_len, device=tokens_text.device, dtype=torch.float32)
        audio_lengths = torch.full((batch_size,), audio_len, device=tokens_text.device, dtype=torch.long)
        
        for b in range(batch_size):
            current_tokens = tokens_text[b].cpu().numpy()  # Convert to numpy for easier processing
            audio_values = torch.zeros(seq_len, device=tokens_text.device, dtype=torch.float32)
            
            in_speech = False  # Track whether we're between bos and eos
            
            for t in range(seq_len):
                token_id = int(current_tokens[t])
                
                if token_id == self.text_bos_id:
                    # Start of speech
                    in_speech = True
                    audio_values[t] = 1.0
                elif token_id == self.text_eos_id:
                    # End of speech
                    in_speech = False
                    audio_values[t] = 0.0
                elif token_id == self.text_pad_id:
                    if in_speech:
                        # Pad token between bos and eos
                        audio_values[t] = 0.5
                    else:
                        # Pad token outside speech
                        audio_values[t] = 0.0
                else:
                    # Regular text token
                    if in_speech:
                        audio_values[t] = 1.0
                    else:
                        audio_values[t] = 0.0
            
            # Expand each token value to cover the corresponding audio samples
            for t in range(seq_len):
                start_sample = t * samples_per_token
                end_sample = min((t + 1) * samples_per_token, audio_len)
                fake_audio[b, start_sample:end_sample] = audio_values[t]
        
        return fake_audio, audio_lengths

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
        input_pad_len: int = 0,
        force_bos_positions = None,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        if self.cfg.get("custom_sample_inference", None):
            device = input_signal.device
            input_signal, sr = torchaudio.load(self.cfg.custom_sample_inference)
            input_signal = input_signal.to(device)[:1, :]
            input_signal = resample(input_signal, sr, self.source_sample_rate)
            input_signal_lens = torch.tensor([input_signal.size(-1)]).to(device)

        if force_bos_positions is not None:
            assert input_signal.shape[0] == len(force_bos_positions), "force_bos_positions must have the same length as batch size"

        if input_pad_len > 0:
            input_signal = torch.nn.functional.pad(input_signal, (0, input_pad_len), mode='constant', value=0)
            input_signal_lens = input_signal_lens + input_pad_len

        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
        B, T_local, H = source_encoded.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:

                last_frame_source = source_encoded[:, T_local - 1 : T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_encoded = torch.cat([source_encoded, pad_source], dim=1)
                last_frame_asr = asr_emb[:, T_local - 1 : T_local, :]
                pad_asr = last_frame_asr.repeat(1, T - T_local, 1)
                asr_emb = torch.cat([asr_emb, pad_asr], dim=1)
        else:
            T = T_local

        # Apply channel weight
        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # This cache is for self.llm
        use_cache = True
        if 'Nemotron' in self.cfg.pretrained_llm:
            # For Nemotron, due to cache issues, we disable cache and use full history mode
            cache = None
            use_cache = False
            logging.info("Using no-cache mode for Nemotron (full history each step)")
        else:
            # Standard cache for other models
            cache = DynamicCache()
            use_cache = True
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        if self.cfg.audio_loss_weight > 0:
            # For Nemotron, we also disable cache in speech generation to be consistent
            speech_use_cache = use_cache
            self.speech_generation.reset_input_and_kv_cache(use_cache=speech_use_cache)

        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        if self.cfg.audio_loss_weight > 0:
            gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)
        else:

            gen_audio = torch.zeros(B, T, self._num_codebooks, device=self.device, dtype=torch.long)


        input_embeds[:, 0] += self._get_bos_embedding()
        if self.cfg.audio_loss_weight > 0:
            first_audio = torch.full(
                [B, 1, self._num_codebooks],
                fill_value=self.speech_delay_id,
                device=self.device,
                dtype=torch.long,
            )
        else:

            first_audio = torch.zeros([B, 1, self._num_codebooks], device=self.device, dtype=torch.long)

        ans = self(
            input_embeds[:, :1],
            cache=cache,
            input_audio_tokens=first_audio,
            seq_mask=None,
            target_text_tokens=None,  # text input will be sampled from llm backbone
            modality_adapter_emb=source_encoded[:, :1],
            asr_emb=asr_emb[:, :1],
            speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        if self.cfg.audio_loss_weight > 0:
            gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        speech_state = torch.zeros(B, device=self.device, dtype=torch.long)
        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            if force_bos_positions is not None:
                for batch_idx in range(last_emb.shape[0]):
                    if force_bos_positions[batch_idx] == t and not (gen_text[batch_idx, :t] == self.text_bos_id).any():
                        last_emb[batch_idx] = self.embed_tokens(torch.full((1,), fill_value=self.text_bos_id, device=self.device))

            input_embeds[:, t] += last_emb

            if self.cfg.audio_loss_weight > 0:
                current_audio = gen_audio[:, t - 1 : t, :]
            else:
                current_audio = torch.zeros([B, 1, self._num_codebooks], device=self.device, dtype=torch.long)
            
            if use_cache:
                # Standard cached mode - pass only current step
                ans = self(
                    input_embeds[:, t : t + 1],
                    cache=ans["cache"],
                    input_audio_tokens=current_audio if self.cfg.audio_loss_weight > 0 else None,
                    seq_mask=None,
                    target_text_tokens=None,  # text input will be sampled from llm backbone
                    modality_adapter_emb=source_encoded[:, t : t + 1],
                    asr_emb=asr_emb[:, t : t + 1],
                    speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
                )
                gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            else:
                # No-cache mode for Nemotron - pass full history up to current step
                if self.cfg.audio_loss_weight > 0:
                    full_audio_history = gen_audio[:, :t, :]
                else:
                    full_audio_history = None
                    
                ans = self(
                    input_embeds[:, :t + 1],
                    cache=None,
                    input_audio_tokens=full_audio_history,
                    seq_mask=None,
                    target_text_tokens=None,  # text input will be sampled from llm backbone
                    modality_adapter_emb=source_encoded[:, :t + 1],
                    asr_emb=asr_emb[:, :t + 1],
                    speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
                )
                gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)


            if self.cfg.audio_loss_weight > 0:
                gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)


            if self.cfg.audio_loss_weight > 0:
                if self.cfg.get('inference_force_speech_state', None):
                    # state 0 - silence, state 1 - speech
                    speech_state = torch.where(
                        gen_text[:, t] == self.text_bos_id, torch.ones_like(speech_state), speech_state
                    )
                    speech_state = torch.where(
                        gen_text[:, t] == self.text_eos_id, torch.zeros_like(speech_state), speech_state
                    )
                    gen_audio[:, t] = torch.where(
                        speech_state.unsqueeze(-1) == 0,
                        gen_audio[:, 0],  # silence
                        gen_audio[:, t],  # speech
                    )
                # inference trick force speech decoder eos/bos to make the model more robust
                num_speech_delay = 1
                if self.cfg.get('inference_force_speech_bos', None) and num_speech_delay < gen_text.shape[1]:
                    gen_audio[:, t] = torch.where(
                        (gen_text[:, t - num_speech_delay].unsqueeze(-1) == self.text_bos_id)
                        * (torch.sum(gen_audio[:, t - num_speech_delay :] == self.speech_bos_id, 1) == 0),
                        self.speech_bos_id,
                        gen_audio[:, t],
                    )

                if self.cfg.get('inference_force_speech_eos', None) and gen_text.shape[
                    1
                ] > num_speech_delay + self.cfg.get("advance_text_channel_by", 0):
                    # tmp solution: force to stop talking if user interruption is detected
                    gen_audio[:, t] = torch.where(
                        (
                            (
                                gen_text[:, t - num_speech_delay - self.cfg.get("advance_text_channel_by", 0)].unsqueeze(
                                    -1
                                )
                                == self.text_eos_id
                            )
                        ),
                        self.speech_eos_id,
                        gen_audio[:, t],
                    )

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]


        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,  # 即使audio_loss_weight=0也返回，但内容是dummy的
            "tokens_len": lengths,
            "source_audio": input_signal,
            "source_audio_len": input_signal_lens,
        }


        if decode_audio and self.cfg.audio_loss_weight > 0:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        if self.cfg.audio_loss_weight > 0:
            self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        if self.cfg.get("custom_sample_inference", None):
            print(ans["audio"].shape, input_signal.shape)
            self.results_logger.merge_and_save_audio(self.cfg.custom_sample_inference+"inf.wav", pred_audio=ans["audio"][0], pred_audio_sr=self.target_sample_rate, user_audio=input_signal[0], user_audio_sr=self.source_sample_rate)
            exit()
        return ans


    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
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
        # TODO(pzelasko): refactor into separate module re-usable across models
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
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
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
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                
                # Get values from model config instead of attention layer (for different model compatibility)
                try:
                    config = self.llm.config
                    
                    # Get config values
                    num_attention_heads = getattr(config, 'num_attention_heads', None)
                    num_key_value_heads = getattr(config, 'num_key_value_heads', None)
                    hidden_size = getattr(config, 'hidden_size', None)
                    
                    if all([num_attention_heads, num_key_value_heads, hidden_size]):
                        # Check divisibility constraints
                        for attr_name, val in [("num_attention_heads", num_attention_heads), 
                                             ("num_key_value_heads", num_key_value_heads), 
                                             ("hidden_size", hidden_size)]:
                            if val % tp_mesh.size() != 0:
                                logging.warning(
                                    f"config.{attr_name}={val} is not divisible by {tp_mesh.size()=}: "
                                    f"set a different tensor parallelism size to avoid errors."
                                )
                        
                        # Set sharded values if attributes exist on attention layer
                        if hasattr(attn_layer, 'num_heads'):
                            attn_layer.num_heads = num_attention_heads // tp_mesh.size()
                        elif hasattr(attn_layer, 'num_attention_heads'):
                            attn_layer.num_attention_heads = num_attention_heads // tp_mesh.size()
                            
                        if hasattr(attn_layer, 'num_key_value_heads'):
                            attn_layer.num_key_value_heads = num_key_value_heads // tp_mesh.size()
                            
                        if hasattr(attn_layer, 'hidden_size'):
                            attn_layer.hidden_size = hidden_size // tp_mesh.size()
                            
                        logging.info(f"Configured tensor parallel for attention: "
                                   f"heads={num_attention_heads//tp_mesh.size()}, "
                                   f"kv_heads={num_key_value_heads//tp_mesh.size()}, "
                                   f"hidden_size={hidden_size//tp_mesh.size()}")
                    else:
                        raise AttributeError("Required config attributes not found")
                        
                except Exception as e:
                    logging.warning(f"Failed to configure tensor parallel using config: {e}")
                    logging.warning("Falling back to attention layer attributes...")
                    
                    # Fallback: try original method using attention layer attributes
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
            if self.cfg.audio_loss_weight > 0:
                self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)


    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
