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
import torch
import random
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor
import torchaudio
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
import tempfile
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, setup_audio_codec, setup_speech_encoder, set_model_dict_for_partial_init
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging


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

        # compute source fps
        self.source_fps = self.source_sample_rate / (self.source_sample_rate * cfg.data.frame_length) # conver frame rate in fps

        setup_audio_codec(self)
        self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # to be able to load older model
        if self.cfg.get("custom_codebook_size", None):
            self._codebook_size = self.cfg.get("custom_codebook_size")

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame
        # compute interpolation factor to interpolate 
        self.interpolation_factor = self.target_fps / self.source_fps
        # x = torch.nn.functional.interpolate(x.unsqueeze(1), size=None, scale_factor=[1, self.interpolation_factor], mode='nearest-exact', align_corners=None, recompute_scale_factor=None, antialias=False)

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
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
            llm_tokenizer_vocab_items = [(subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword)) for subword in llm_tokenizer_vocab_items]

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

        if self.cfg.get("pretrained_tts_from_s2s", None):
            self.init_speech_generation_from_another_s2s_checkpoint(self.cfg.pretrained_tts_from_s2s)

        self.embed_audio_tokens = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                for _ in range(self._num_codebooks)
            ]
        )
        self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False

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
            checkpoint_state = {k.replace("model.speech_decoder.", "").replace("speech_generation.", ""): v for k, v in checkpoint_state.items() if "model.speech_decoder." in k or "speech_generation." in k}
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

    def forward(self, input_embeds: Tensor, cache=None, input_audio_tokens=None, seq_mask=None, target_text_tokens=None, modality_adapter_emb=None, asr_emb=None, speaker_encoder_emb=None) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        if seq_mask is not None:
            # This is training Mode
            seq_mask = seq_mask[:, :, -1].reshape(seq_mask.size(0), seq_mask.size(1))
            # disable cache in training mode
            if self.speech_generation.use_input_cache:
                self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        # if inference time, uses the target text tokens sampled from the llm backbone
        if self.speech_generation.use_input_cache and not self.training:
            if self.cfg.get("inference_pad_boost", None):
                print("inference_pad_boost")
                text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
            if self.cfg.get("inference_bos_boost", None):
                text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost
            if self.cfg.get("inference_eos_boost", None):
                text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost
                print("inference_eos_boost")

            target_text_tokens = torch.argmax(text_logits, dim=-1).view(B, T).contiguous()

        audio_logits, _  = self.speech_generation(
            out['last_hidden_state'].transpose(0, 1), seq_mask, input_audio_tokens=input_audio_tokens, target_text_tokens=target_text_tokens, modality_adapter_emb=modality_adapter_emb, asr_emb=asr_emb, speaker_encoder_emb=speaker_encoder_emb
        )

        audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans


    def add_noise_to_batch(self, batch_audio, noise_folder, snr_db=20, noise_prob_scale_user=0.3, noise_prob_scale_user_min_snr=-15, noise_prob_scale_user_max_snr=24, snr_measure_dur=0.0, noise_resample=True, noise_prob_low_pass=0.1):

        batch_size, audio_length = batch_audio.shape

        import glob
        import soundfile as sf
        from scipy.signal import butter, lfilter
        import librosa

        noise_files = [f for f in glob.glob(noise_folder + "/*.wav")]
        if not noise_files:
            raise ValueError(f"No noise files found in {noise_folder}")

        for i in range(batch_size):

            def get_scale_factor(signal, noise, snr_db):
                if snr_measure_dur > 0:
                    signal = signal[: (snr_measure_dur * self.source_sample_rate)]
                    noise = noise[: (snr_measure_dur * self.source_sample_rate)]
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(
                        noise_prob_scale_user_min_snr,
                        noise_prob_scale_user_max_snr
                    ),
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
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'seq_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        if self.cfg.get('use_old_noise_aug', None):
            # ToDo we are applying it in all datasets, old codebase does not applied in real conv data
            noise_prob = 0.99
            noise_min_snr = 20
            noise_max_snr = 50
            noise_path = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/dns5/dns5_demand_noise/"
            noise_path_name = "*"
            no_noise_audio = batch["source_audio"].clone()
            if self.training and 's2s_duplex_overlap_as_s2s_duplex' not in batch["formatter"] and noise_prob and random.random() < noise_prob:
                batch["source_audio"] = self.add_noise_to_batch(batch["source_audio"], os.path.join(noise_path, noise_path_name), snr_db=random.randint(noise_min_snr, noise_max_snr), noise_prob_scale_user=0.3, noise_prob_scale_user_min_snr=-15, noise_prob_scale_user_max_snr=24, snr_measure_dur=0.0, noise_resample=True, noise_prob_low_pass=0.1)
        else:
            # change audio volume randomly
            if self.training and random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                min_scale_val = self.cfg.get('noise_scale_user_min', 0.0631) # -15 snr
                max_scale_val = self.cfg.get('noise_scale_user_min', 5.6234) # 24 snr

                # get a random float value between min and max
                scaling_factor = torch.rand(batch["source_audio"].size(0), device=batch["source_audio"].device) * (max_scale_val - min_scale_val) + min_scale_val
                batch["source_audio"] = batch["source_audio"] * scaling_factor.unsqueeze(-1)

            # apply low pass filter
            if self.training and random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                cutoff_freq = self.cfg.get('noise_low_pass_cutoff_freq', 1000.0)
                # note here we are using a biquad filter, older codebase we are using a filter of order 5
                batch["source_audio"] = torchaudio.functional.lowpass_biquad(waveform=batch["source_audio"], sample_rate=self.source_sample_rate, cutoff_freq=cutoff_freq)


        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"], return_encoder_emb=True,
        )

        # if inference return speaker embedding None and it will uses the cached speaker embedding
        if not self.training:
            speaker_encoder_emb = None
        else: # if training or eval extract embedding from first agent turn returned by the dataloader 
            if self.speech_generation.use_speaker_encoder:
                target_first_turn_audio = batch["target_first_turn_audio"]
                target_first_turn_audio_lens = batch["target_first_turn_audio_lens"]
                speaker_encoder_emb = self.speech_generation.get_speaker_embedding(target_first_turn_audio, target_first_turn_audio_lens, self.target_sample_rate)
            else:
                speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

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

        # ToDo: implement in a way that we can set the number of speech delay > 1
        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )
        # move back text channel by x, in inference it advance the text channel prediction
        # it is the oposite of speech delay applied on text channel
        if self.advance_text_channel_by:
            pad = torch.full(
                (target_tokens.shape[0], self.advance_text_channel_by),
                fill_value=self.text_pad_id, 
                device=target_tokens.device,
                dtype=torch.long,
            )
            target_tokens = torch.cat([target_tokens[:, self.advance_text_channel_by:], pad], dim=-1)

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
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

        seq_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        if self.cfg.get("mask_sequence_loss", True):
            # set the mask based on the target_token_lens to disconsider sequence padding in loss
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                seq_mask[i, speech_end_idx:, :] = 0

            # check new mask consistency
            mask_lengths = seq_mask[:, :, 0].sum(-1)
            assert torch.allclose(batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0)

        # scale loss by factors
        loss_scale = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.float32,
        )

        if self.cfg.get("scale_loss_by", None):
            if self.cfg.scale_loss_by == 'non_sil_t':
                loss_scale[:, :, :1] = torch.where(text_labels.unsqueeze(-1) != self.text_pad_id, 4.0, loss_scale[:, :, :1])
            elif self.cfg.scale_loss_by == 'dynamic_text_non_sil_4x_and_bos_eos':
                # Set text loss weights
                for i in range(text_labels.size(0)):
                    current_scale = loss_scale[i, :, :1]
                    labels = text_labels.unsqueeze(-1)
                    current_scale = loss_scale[i, :, :1]
                    num_real_padding_tokens = (torch.numel(current_scale) - current_scale.sum()).item()
                    silence_idxs = labels[i, :, :1] == self.text_pad_id

                    # compute dynamic silence/nonsilence factor
                    silence_idxs = silence_idxs * current_scale.bool()
                    num_silence_tokens = silence_idxs.sum().item()
                    num_non_silence = torch.numel(silence_idxs) - num_real_padding_tokens
                    factor = num_silence_tokens / num_non_silence

                    # make silence text tokens 2 x times less relevant in the loss than the silence tokens
                    new_weight = factor / 2
                    loss_scale[i, :, :1] = torch.where(silence_idxs, new_weight, loss_scale[i, :, :1])

                # set eos/bos 6x more important than a speech tokens and 12x more than a silence, this is that high because we will have only one bos/eos per turn and if it is nor right predicted the model will not produce text/speech
                loss_scale[:, :, :1] = torch.where(labels == self.text_bos_id, 6.0, loss_scale[:, :, :1])
                loss_scale[:, :, :1] = torch.where(labels == self.text_eos_id, 6.0, loss_scale[:, :, :1])
            elif self.cfg.scale_loss_by == 'non_sil_4_eos_bos_12':
                loss_scale[:, :, :1] = torch.where(text_labels.unsqueeze(-1) != self.text_pad_id, 4.0, loss_scale[:, :, :1])
                # set eos/bos 3x more important than a speech tokens and 12x more than a silence, this is that high because we will have only one bos/eos per turn and if it is nor right predicted the model will not produce text/speech
                loss_scale[:, :, :1] = torch.where(text_labels.unsqueeze(-1) == self.text_bos_id, 12.0, loss_scale[:, :, :1])
                loss_scale[:, :, :1] = torch.where(text_labels.unsqueeze(-1) == self.text_eos_id, 12.0, loss_scale[:, :, :1])
            else:
                raise ValueError(f"Unknown scale_loss_by: {self.cfg.scale_loss_by}")

        # debug samples:
        if self.cfg.get("debug_dataloader_audios_path", None) and self.training:
            def count_leading_silence_tokens(tensor: torch.Tensor, silence_token: int = 0) -> int:
                """
                Count the number of consecutive silence tokens at the beginning of a 1D tensor.

                Args:
                    tensor (torch.Tensor): 1D tensor of tokens.
                    silence_token (int): The token considered as silence (default: 0).

                Returns:
                    int: Number of consecutive silence tokens at the beginning.
                """
                if tensor.ndim != 1:
                    raise ValueError("Input tensor must be 1D.")

                count = 0
                for token in tensor:
                    if token.item() == silence_token:
                        count += 1
                    else:
                        break
                return count
            def write_wave(one_audio_signal, file_name, sr=None):
                import numpy as np
                import soundfile as sf
                one_audio_signal = one_audio_signal.cpu().numpy()
                one_audio_signal = one_audio_signal.astype(np.float32)
                if sr is None:
                    sr = self.target_sample_rate
                # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
                sf.write(file_name, one_audio_signal, sr)    

            # encode and decode the audio
            with fp32_precision(), torch.no_grad():
                lengths = torch.tensor([batch["target_audio"].shape[1]]*batch["target_audio"].shape[0]).to(self.audio_codec.device)
                reconstructed_audio_from_wav, _ = self.audio_codec(audio=batch["target_audio"], audio_len=lengths)
                # reconstruct wav
                audio_labels_ = replace_control_speech_codes(audio_labels, self._control_codes)
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([audio_labels_.shape[1]]*audio_labels_.shape[0]).to(self.audio_codec.device)
                    reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                        tokens=audio_labels_.transpose(1, 2), tokens_len=lengths
                    )

            for i in range(audio_labels_.shape[0]):
                write_wave(
                    batch["target_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_{i}.wav"),
                    sr=self.target_sample_rate
                )
                write_wave(
                    batch["target_first_turn_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"speaker_ref_{i}.wav"),
                    sr=self.target_sample_rate
                )
                write_wave(
                    batch["source_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"source_audio_{i}.wav"),
                    sr=self.source_sample_rate
                )
                
                write_wave(
                    reconstructed_audio_from_tokens[i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_tokens_{i}.wav"),
                    sr=self.target_sample_rate
                )

                write_wave(
                    reconstructed_audio_from_wav[i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_waveform_{i}.wav"),
                    sr=self.target_sample_rate
                )

            # check text
            print("text_labels decoded:", tokens_to_str(text_labels[-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))
            print("target labels from dataloader decoded:",  tokens_to_str(batch["target_tokens"][-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))
            print("Number of padding tokens on the begining:", count_leading_silence_tokens(text_labels[-1:].squeeze(), self.text_pad_id))
            print(batch["formatter"])
            exit()

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

    def track_param_updates(self, param_filter: str = "speech_generation.text_embeddings", verbose=True):
        if not hasattr(self, "_param_tracker_state"):
            # First-time call: cache current weights
            self._param_tracker_state = {
                name: p.clone().detach()
                for name, p in self.named_parameters()
                if param_filter in name
            }
            if verbose:
                print(f"[Tracker] Initialized snapshot for: {[k for k in self._param_tracker_state]}")
            return

        if verbose:
            print(f"\n📊 [Tracker] Comparing parameters with filter '{param_filter}':")

        for name, p in self.named_parameters():
            if param_filter not in name:
                continue
            prev = self._param_tracker_state[name]
            now = p.detach()
            delta = (now - prev).abs().sum()
            mean_delta = delta / p.numel()

            if delta.item() == 0.0:
                print(f"❌ {name:50s} has NOT been updated (Δsum = 0.0)")
            else:
                print(f"✅ {name:50s} Δsum={delta.item():.4e}  Δmean={mean_delta.item():.4e}")

            # update tracker state
            self._param_tracker_state[name] = now.clone().detach()

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
            if is_frozen(m):
                m.eval()
        
        # self.track_param_updates("speech_generation.")

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            seq_mask=inputs["seq_mask"],
            target_text_tokens=inputs["text_labels"],
            modality_adapter_emb=inputs["perception_emb"],
            asr_emb=inputs["asr_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            # mask audio logits to ignore sequence padding
            text_logits = forward_outputs["text_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

            text_loss = (
                (torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="none",
                ) * inputs["loss_scale"][:, :, 0].flatten(0, 1)).sum(-1)
                / num_frames
            )

            # mask audio logits to ignore sequence padding
            audio_logits = forward_outputs["audio_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                audio_logits = audio_logits * inputs["seq_mask"][:, :, -1].unsqueeze(-1).unsqueeze(-1)

            audio_loss = (
                (torch.nn.functional.cross_entropy(
                    audio_logits.flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                    inputs["audio_labels"].flatten(0, 2),
                    reduction="none",
                ) * inputs["loss_scale"][:, :, 1:].flatten(0, 2)).sum(-1)
                / (num_frames * self._num_codebooks)
            )

        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32
        if hasattr(self.speech_generation, "use_speaker_encoder") and self.speech_generation.use_speaker_encoder:
            self.speech_generation.setup_speaker_encoder() # potentially reloads the speaker encoder to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()

        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()
        tolerance = int(self.cfg.get("val_acc_tolerance", 160)/(1000/self.target_fps)) # 160 ms as default tolerance --> 2 tokens for 12.5FPS and 1 for 25FPS
        self.text_bos_acc = TokenAccuracy(token_name="text_bos", token_id=self.text_bos_id, tolerance=tolerance).reset()
        self.text_eos_acc = TokenAccuracy(token_name="text_eos", token_id=self.text_eos_id, tolerance=tolerance).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_bos_acc = self.text_bos_acc.compute()
        for k, m in text_bos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_eos_acc = self.text_eos_acc.compute()
        for k, m in text_eos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):

        # Update speaker embedding to reflect the one in the prompt during inference
        if self.speech_generation.use_speaker_encoder and self.speech_generation.inference_speaker_reference:
            self.speech_generation.update_inference_speaker_embedding(self.speech_generation.inference_speaker_reference)

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
            )

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

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            self.text_bos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])
            self.text_eos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
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
        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
            return_encoder_emb=True
        )
        B, T_local, H = source_encoded.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame = source_encoded[:, T_local - 1 : T_local, :]  # (B,1,H)
                pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                source_encoded = torch.cat([source_encoded, pad], dim=1)
                asr_emb = torch.cat([asr_emb, pad], dim=1)
        else:
            T = T_local

        # Apply channel weight
        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # This cache is for self.llm
        cache = DynamicCache()
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(
            input_embeds[:, :1], 
            cache=cache,
            input_audio_tokens=first_audio,
            seq_mask=None,
            target_text_tokens=None, # text input will be sampled from llm backbone
            modality_adapter_emb=source_encoded[:, :1],
            asr_emb=asr_emb[:, :1],
            speaker_encoder_emb=None, # for inference uses the cached inference_speaker_embedding
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(
                input_embeds[:, t : t + 1],
                cache=ans["cache"],
                input_audio_tokens=current_audio,
                seq_mask=None,
                target_text_tokens=None, # text input will be sampled from llm backbone
                modality_adapter_emb=source_encoded[:, t : t + 1],
                asr_emb=asr_emb[:, t : t + 1],
                speaker_encoder_emb=None, # for inference uses the cached inference_speaker_embedding
            )
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

            # inference trick force speech decoder eos/bos to make the model more robust
            num_speech_delay = 1
            if (
                self.cfg.get('inference_force_speech_bos', None)
                and  num_speech_delay < gen_text.shape[1]
            ):
                gen_audio[:, -1] = torch.where(
                    (gen_text[:, -1 - num_speech_delay].unsqueeze(-1) == self.text_bos_id)
                    * (
                        torch.sum(gen_audio[:, -1 - num_speech_delay :] == self.speech_bos_id, 1)
                        == 0
                    ),
                    self.speech_bos_id,
                    gen_audio[:, -1],
                )

            if self.cfg.get('inference_force_speech_eos', None):
                # tmp solution: force to stop talking if user interruption is detected
                gen_audio[:, -1] = torch.where(
                    ((gen_text[:, -1].unsqueeze(-1) == self.text_eos_id)),
                    self.speech_eos_id,
                    gen_audio[:, -1],
                )

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        # Call reset_input_and_kv_cache to reset cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=False)
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
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
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
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
