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
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from safetensors.torch import load_file
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS

from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel

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


def generate_multiturn_speaking_mask(input_ids: torch.Tensor, bos_token_id: int = 0, eos_token_id: int = 1):
    """
    Efficient, batched speaking mask generator that marks 1 between <bos> and <eos> pairs.
    If <eos> is missing after a <bos>, mask continues to end. Handles multiple turns.

    Args:
        input_ids (torch.Tensor): LongTensor of shape (B, T)
        bos_token_id (int): Token ID for <bos>
        eos_token_id (int): Token ID for <eos>

    Returns:
        torch.Tensor: FloatTensor of shape (B, T), with 1.0 for speaking, 0.0 for silence.

    Note BOS is considered as speaking (1) and EOS as non speaking 0
    """
    B, T = input_ids.shape
    device = input_ids.device
    bos_mask = (input_ids == bos_token_id).to(torch.int32).to(device)
    eos_mask = (input_ids == eos_token_id).to(torch.int32).to(device)
    bos_cumsum = torch.cumsum(bos_mask, dim=1)
    eos_cumsum = torch.cumsum(eos_mask, dim=1)
    speaking_mask = (bos_cumsum > eos_cumsum).to(torch.float32)
    return speaking_mask.long()


def add_structured_noise_preserve_tail(
    mask: torch.Tensor,
    span_prob: float = 0.05,
    min_len: int = 2,
    max_len: int = 3,
    min_preserve: int = 4,
):
    """
    Adds structured noise to a binary mask by flipping random spans (2–3 tokens at a time),
    while preserving the last `min_preserve` tokens of each speaking region (1s).

    Args:
        mask (torch.Tensor): Binary mask of shape (B, T), values in {0, 1}
        span_prob (float): Probability of inserting a noisy span per token
        min_len (int): Minimum span length to flip
        max_len (int): Maximum span length to flip
        min_preserve (int): Number of 1s at the end of each span to protect from flipping

    Returns:
        torch.Tensor: Noised mask (same shape)
    """
    B, T = mask.shape
    noised_mask = mask.clone()

    for b in range(B):
        i = 0
        while i < T:
            if mask[b, i] == 1:
                # Start of a speaking region
                start = i
                while i < T and mask[b, i] == 1:
                    i += 1
                end = i  # exclusive
                span_len = end - start

                if span_len > min_preserve:
                    allowed_start = start
                    allowed_end = end - min_preserve
                    j = allowed_start
                    while j < allowed_end:
                        if random.random() < span_prob:
                            flip_len = random.randint(min_len, max_len)
                            flip_end = min(j + flip_len, allowed_end)
                            noised_mask[b, j:flip_end] = (noised_mask[b, j:flip_end] + 1) % 2
                            j = flip_end
                        else:
                            j += 1
            else:
                i += 1
    return noised_mask


class NemotronVoiceChat(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.full_cfg = cfg
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # Load Duplex STT model
        self.stt_model = DuplexSTTModel(OmegaConf.to_container(self.cfg.stt, resolve=True))

        # Load Duplex TTS model
        # delete old config name for old version compatibility
        # if (
        #     hasattr(self.cfg, "speech_generation")
        #     and hasattr(self.cfg.speech_generation, "model")
        #     and hasattr(self.cfg.speech_generation.model, "tts_config")
        #     and hasattr(self.cfg.speech_generation.model.tts_config, "cas_config")
        #     and hasattr(self.cfg.speech_generation.model.tts_config.cas_config, "pretrained_tokenizer_name")
        # ):
        #     del self.cfg.speech_generation.model.tts_config.cas_config.pretrained_tokenizer_name

        self.tts_model = DuplexEARTTS(OmegaConf.to_container(self.cfg.speech_generation, resolve=True))
        # reset silence tokens to avoid inference issues
        self.tts_model.codec_silence_tokens = self.tts_model.get_codec_silence_frame()

        self.target_fps = self.tts_model.target_fps
        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        if self.cfg.get("pretrained_s2s_model", None):
            logging.info(f"Loading pretrained s2s model from {self.cfg.pretrained_s2s_model}")
            if os.path.isdir(self.cfg.pretrained_s2s_model):
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
                        tensor = f.get_tensor(key)

                        # recreates the audio_prompt_latents if needed
                        if "audio_prompt_latents." in key:
                            self.tts_model.maybe_recreate_cached_audio_prompt_latents_structure({key: tensor})
                            # refresh so the new nn.Parameter appears in the dict
                            model_state_dict = self.state_dict()

                        if key in model_state_dict:
                            # Load tensor and copy to model parameter
                            model_state_dict[key].copy_(tensor)
                            loaded_keys.append(key)
                        else:
                            missing_keys.append(key)
                        del tensor  # Free memory immediately

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

    def training_step(self, batch: dict, batch_idx: int):

        return None

    def on_train_epoch_start(self) -> None:
        self.tts_model.on_train_epoch_start()
        self.stt_model.on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()

        # Load separate Parakeet-TDT model for timestamp detection
        scoring_asr_model = self.cfg.get('scoring_asr', 'nvidia/parakeet-tdt-1.1b')
        if scoring_asr_model == 'stt_en_fastconformer_transducer_large':
            scoring_asr_model = 'nvidia/parakeet-tdt-1.1b'

        from nemo.collections.asr.models import ASRModel

        if os.path.exists(scoring_asr_model) and scoring_asr_model.endswith('.nemo'):
            logging.info(f"Loading local ASR model for timestamps from: {scoring_asr_model}")
            self.asr_timestamp_model = ASRModel.restore_from(restore_path=scoring_asr_model, map_location=self.device)
        else:
            logging.info(f"Loading HuggingFace ASR model for timestamps: {scoring_asr_model}")
            self.asr_timestamp_model = ASRModel.from_pretrained(model_name=scoring_asr_model)

        self.bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        
        # Combine per-rank files into one per dataset (ifeval_rank0.json + ifeval_rank1.json + ... -> ifeval.json)
        self.results_logger.compute_and_save()

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            dataset_batch = dataset_batch["audio_data"]

            prompt_tokens = dataset_batch.get("prompt_tokens", None)
            prompt_token_lens = dataset_batch.get("prompt_token_lens", None)
            function_calls = dataset_batch.get("function_calls", None)
            function_call_lengths = dataset_batch.get("function_call_lengths", None)
            function_call_steps = dataset_batch.get("function_call_steps", None)
            function_responses = dataset_batch.get("function_responses", None)
            function_response_lengths = dataset_batch.get("function_response_lengths", None)
            function_response_steps = dataset_batch.get("function_response_steps", None)

            # Trailing silence on user waveform (must use STT encoder SR — same as DuplexSTTModel._init_inference).
            extra_decoding_seconds = self.cfg.get("extra_decoding_seconds", 0.0)
            pad_sr = getattr(self.stt_model, "source_sample_rate", None) or self.source_sample_rate
            input_pad_len = int(extra_decoding_seconds * pad_sr)

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens,
                input_pad_len=input_pad_len,
                function_calls=function_calls,
                function_call_lengths=function_call_lengths,
                function_call_steps=function_call_steps,
                function_responses=function_responses,
                function_response_lengths=function_response_lengths,
                function_response_steps=function_response_steps,
                temperature=float(self.cfg.get("temperature", 0.0)),
                top_p=float(self.cfg.get("top_p", 1.0)),
                repetition_penalty=float(self.cfg.get("repetition_penalty", 1.0)),
                presence_penalty=float(self.cfg.get("presence_penalty", 0.0)),
            )

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )

                # Add agent speech start timestamps using ASR model with timestamp detection
                asr_hyps_with_timestamps = []

                # Check if timestamp detection should be used
                use_timestamps = self.cfg.get('use_asr_timestamps', True)

                if use_timestamps and hasattr(self, 'asr_timestamp_model'):
                    try:
                        # Prepare audio batch for timestamp detection
                        resampled_audio = resample(results["audio"], 22050, 16000)
                        audio_list = []
                        for i in range(len(asr_hyps)):
                            audio_len = int((results["audio_len"][i] / 22050 * 16000).item())
                            audio_list.append(resampled_audio[i, :audio_len].cpu())

                        # Transcribe with timestamps
                        timestamp_outputs = self.asr_timestamp_model.transcribe(
                            audio_list,
                            batch_size=len(audio_list),
                            timestamps=True,
                            verbose=False,
                        )

                        for i, asr_text in enumerate(asr_hyps):
                            # Extract first word timestamp (speech onset)
                            try:
                                if hasattr(timestamp_outputs[i], 'timestamp') and timestamp_outputs[i].timestamp:
                                    word_timestamps = timestamp_outputs[i].timestamp.get('word', [])
                                    if word_timestamps and len(word_timestamps) > 0:
                                        # First word's start time is the speech onset
                                        start_time = word_timestamps[0]['start']
                                        asr_text_with_ts = f"<|{start_time:.2f}|> {asr_text}" if asr_text else f"<|{start_time:.2f}|>"
                                    else:
                                        asr_text_with_ts = asr_text
                                else:
                                    asr_text_with_ts = asr_text
                            except (AttributeError, IndexError, KeyError, TypeError) as e:
                                logging.debug(f"Timestamp extraction failed: {e}")
                                asr_text_with_ts = asr_text

                            asr_hyps_with_timestamps.append(asr_text_with_ts)

                    except Exception as e:
                        logging.warning(f"Timestamp detection failed, using transcriptions without timestamps: {e}")
                        asr_hyps_with_timestamps = asr_hyps
                else:
                    asr_hyps_with_timestamps = asr_hyps

                # Decode function channel tokens to text (same as DuplexSTTModel.validation_step)
                function_channel_text = None
                function_channel_with_inserted_response = None
                function_call_positions = None
                func_tokens_for_pred = results.get("tokens_function_pred", results.get("tokens_function", None))
                if func_tokens_for_pred is not None:
                    function_channel_text = tokens_to_str(
                        func_tokens_for_pred,
                        results["tokens_len"],
                        tokenizer=self.stt_model.tokenizer,
                        pad_id=self.stt_model.text_pad_id,
                        user_bos_id=self.stt_model.user_bos_id,
                        eval_text_turn_taking=False,
                        sil_id=None
                    )

                    # Also decode full function channel (includes prefilled responses) for debugging
                    if results.get("tokens_function") is not None:
                        function_channel_with_inserted_response = tokens_to_str(
                            results["tokens_function"],
                            results["tokens_len"],
                            tokenizer=self.stt_model.tokenizer,
                            pad_id=self.stt_model.text_pad_id,
                            user_bos_id=self.stt_model.user_bos_id,
                            eval_text_turn_taking=False,
                            sil_id=None
                        )

                    # Extract function call positions/timing
                    function_call_positions = self.stt_model._extract_function_call_positions(
                        func_tokens_for_pred,
                        results["tokens_len"],
                        results["tokens_text"]
                    )

                    # Log function channel predictions before saving
                    logging.info(f"[Function Channel Predictions - {name}]:")
                    for idx, fc_text in enumerate(function_channel_text):
                        sample_id = dataset_batch['sample_id'][idx]
                        logging.info(f"  Sample {sample_id}: {fc_text}")
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

                # Decode ground truth function channel (target)
                target_function_channel = None
                if function_calls is not None:
                    B_fc = function_calls.shape[0]
                    target_function_channel_list = []
                    for b in range(B_fc):
                        calls_for_batch = []
                        if function_call_lengths is not None:
                            num_calls = (function_call_lengths[b] > 0).sum().item()
                            for t in range(num_calls):
                                call_length = function_call_lengths[b, t].item()
                                if call_length > 0:
                                    call_tokens = function_calls[b, t, :call_length]
                                    call_text = self.stt_model.tokenizer.ids_to_text(call_tokens.tolist())
                                    calls_for_batch.append(call_text)
                        target_text = "".join(calls_for_batch) if calls_for_batch else ""
                        target_function_channel_list.append(target_text)
                    target_function_channel = target_function_channel_list
                    logging.info(f"[Target Function Channel - {name}]:")
                    for idx, target_fc_text in enumerate(target_function_channel):
                        sample_id = dataset_batch['sample_id'][idx]
                        logging.info(f"  Sample {sample_id}: {target_fc_text}")
                # import pdb; pdb.set_trace()
                # Logging: use STT post-process outputs for the user channel — not dataset_batch["source_audio"].
                # results["source_audio"] is inference_state["input_signal"] after _init_inference, i.e. the waveform
                # actually encoded by perception, including trailing zeros from input_pad_len (extra_decoding_seconds).
                # results["source_audio_len"] already matches that tensor (same as input_signal_lens post-pad).
                # Do not add input_pad_len again here; it would double-count. input_pad_len is only needed above when
                # calling offline_inference(..., input_pad_len=...).
                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps_with_timestamps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=results["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    src_refs=dataset_batch.get("source_texts", None),
                    src_hyps=results.get("src_text", None),
                    src_hyps_asr_head=results.get("src_text_asr_head"),
                    src_hyps_rnnt=results.get("src_text_rnnt"),
                    audio_lens=results["source_audio_len"],
                    eou_pred=(
                        results["gen_eou"]
                        if "gen_eou" in results
                        else None
                    ),
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.stt_model.tokenizer,
                    system_prompt=dataset_batch.get("system_prompt", None),
                    system_prompt_supervision_0=dataset_batch.get("system_prompt_supervision_0", None),
                    function_channel_text=function_channel_text,
                    function_channel_with_inserted_response=function_channel_with_inserted_response,
                    target_function_channel=target_function_channel,
                    function_call_positions=function_call_positions,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        prompt_tokens: torch.Tensor = None,
        prompt_token_lens: torch.Tensor = None,
        input_pad_len: int = 0,
        force_bos_positions=None,
        decode_audio: bool = True,
        incremental_audio_decoding: bool = False,
        function_calls: torch.Tensor = None,
        function_call_lengths: torch.Tensor = None,
        function_responses: torch.Tensor = None,
        function_response_lengths: torch.Tensor = None,
        function_response_steps: torch.Tensor = None,
        function_call_steps: torch.Tensor = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
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

        inference_state = self.stt_model._init_inference(
            input_signal, input_signal_lens, input_pad_len,
            force_bos_positions, prompt_tokens, prompt_token_lens
        )
        inference_state["temperature"] = float(temperature)
        inference_state["top_p"] = float(top_p)
        inference_state["repetition_penalty"] = float(repetition_penalty)
        inference_state["presence_penalty"] = float(presence_penalty)
        inference_state["_text_special_ids"] = {
            self.stt_model.text_pad_id,
            self.stt_model.text_bos_id,
            self.stt_model.text_eos_id,
        }

        # RNNT for ASR branch: same setup as DuplexSTTModel.offline_inference (VoiceChat uses its own loop).
        inference_state["rnnt_src_text"] = None
        asr_branch_decode = self.stt_model.cfg.get("asr_branch_decode", False) if hasattr(self.stt_model.cfg, "get") else getattr(self.stt_model.cfg, "asr_branch_decode", False)
        if isinstance(asr_branch_decode, str):
            asr_branch_decode = asr_branch_decode.strip().lower() in ("true", "1", "yes")
        if asr_branch_decode and self.cfg.get("rnnt_predict_user_text", False):
            rnnt_dec = getattr(self.stt_model, "rnnt_decoder", None)
            rnnt_joint = getattr(self.stt_model, "rnnt_joint", None)
            if rnnt_dec is not None and rnnt_joint is not None:
                inference_state["_run_rnnt_in_loop"] = True
                inference_state["rnnt_partial_hypotheses"] = None
            else:
                inference_state["_run_rnnt_in_loop"] = False
        else:
            inference_state["_run_rnnt_in_loop"] = False

        # FC expand: same gating as DuplexSTTModel.offline_inference (call-only batches; lengths + steps).
        # Must snapshot lengths before expand — _expand_waveform_fc_timeline in _post_inference needs this.
        if self.stt_model.use_function_head and function_call_lengths is not None and function_call_steps is not None:
            inference_state["lengths_pre_fc"] = inference_state["lengths"].clone()
            inference_state = self.stt_model._expand_for_function_calling(
                inference_state,
                function_call_lengths=function_call_lengths,
                function_responses=function_responses,
                function_response_lengths=function_response_lengths,
                function_response_steps=function_response_steps,
                function_call_steps=function_call_steps,
                prompt_token_lens=prompt_token_lens,
            )

        ans, inference_state = self.stt_model._step_zero(inference_state)

        # RNNT step for t=0 (main loop starts at t=1, same as DuplexSTTModel.offline_inference)
        if inference_state.get("_run_rnnt_in_loop") and inference_state["asr_emb"].shape[1] > 0:
            asr_emb = inference_state["asr_emb"]
            asr_emb_lengths = inference_state["asr_emb_lengths"]
            B_rnnt = inference_state["B"]
            device = asr_emb.device
            if asr_emb_lengths.dim() == 0:
                enc_len = 1 if 0 < asr_emb_lengths.item() else 0
                encoded_lengths = torch.full((B_rnnt,), enc_len, device=device, dtype=torch.long)
            else:
                encoded_lengths = (0 < asr_emb_lengths).long().to(device)
            inference_state["rnnt_partial_hypotheses"] = self.stt_model._run_rnnt_decode_one_step(
                asr_emb[:, 0:1, :], encoded_lengths, inference_state.get("rnnt_partial_hypotheses")
            )

        B = inference_state["B"]
        T = inference_state["T"]

        # Init external Duplex TTS model
        generation_config = None
        guidance_enabled = True

        # Use pre-baked speaker latent by name if available; otherwise load from wav
        speaker_name = self.cfg.get("inference_speaker_name", None) or None
        if speaker_name is not None:
            # Use pre-baked latent stored in model weights — no wav file needed
            self.tts_model.set_init_inputs(speaker_name=speaker_name)
        else:
            speaker_audio, sr = torchaudio.load(self.cfg.inference_speaker_reference)
            speaker_audio = resample(speaker_audio, sr, self.tts_model.target_sample_rate)
            speaker_audio = speaker_audio.repeat(B, 1).to(self.device)
            speaker_audio_lens = torch.tensor([speaker_audio.size(1)]).long().repeat(B).to(self.device)
            self.tts_model.set_init_inputs(
                speaker_audio=speaker_audio,
                speaker_audio_lens=speaker_audio_lens,
            )
        init_inputs = self.tts_model.get_init_inputs(B=B)

        if generation_config is None:
            generation_config = self.tts_model._get_generation_config(guidance_enabled)

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})
        # warmup the model and generate the very first audio token
        outputs = self.tts_model.tts_model(**init_inputs)
        code = init_inputs["code"][:, -1:]

        past_key_values = outputs.past_key_values
        num_quantizers = self.tts_model.tts_model.config.num_quantizers
        gen_codes = torch.zeros(B, T, num_quantizers, device=self.device, dtype=torch.long)
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        subword_mask = torch.ones(B, T, device=self.device, dtype=torch.bool)

        # init
        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)
        tts_initialized = False

        # Autoregressive loop: TTS runs at every step (prompt region produces PAD-derived
        # codes that are trimmed per-sample after the loop).
        for t in range(1, T):
            # do one step inference on Duplex STT model
            ans = self.stt_model._step_inference(t, inference_state, ans, force_bos_positions)

            # do one step inference on Duplex TTS model
            current_subword_id = inference_state["gen_text"][:, t].unsqueeze(-1)
            if not tts_initialized:
                prev_subword_id = first_context_subword_id
                tts_initialized = True
            else:
                prev_subword_id = inference_state["gen_text"][:, t - 1].unsqueeze(-1)

            # create subword_mask
            current_subword_mask = subword_mask[:, t].unsqueeze(-1)

            code, past_key_values = self.tts_model.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
            )

            gen_codes[:, t] = code.squeeze(1)

            if decode_audio and incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.tts_model.decode_one_audio_step(
                    gen_codes[:, :t + 1],
                    number_prev_tokens=self.cfg.get("inference_codec_decoding_prev_tokens_number", None),
                )
                if audio_pred is None:
                    audio_pred = audio_pred_i
                else:
                    audio_pred = torch.cat([audio_pred, audio_pred_i], dim=1)
                audio_pred_len += audio_pred_i_len

            logging.info(f"Autoregressive inference step: {t} of {T} !")

        # Build final RNNT transcript from partial hypotheses (one asr_emb frame per step).
        if inference_state.get("rnnt_partial_hypotheses") is not None:
            inference_state["rnnt_src_text"] = self.stt_model._rnnt_hypotheses_to_src_text(
                inference_state["rnnt_partial_hypotheses"]
            )

        # FSDP: trim gen_codes to local width before global MAX pad.
        effective_len = inference_state["T_local"]
        if self._use_fsdp and T > effective_len:
            gen_codes = gen_codes[:, :effective_len]

        # Per-sample prompt trim: remove prompt region from gen_codes (mirrors
        # the per-sample text trim in _post_inference).
        prompt_token_lens_local = inference_state.get("prompt_token_lens", None)
        if prompt_token_lens_local is not None:
            max_prompt_len = int(prompt_token_lens_local.max().item())
            if max_prompt_len > 0:
                current_T_codes = gen_codes.shape[1]
                lengths_local = inference_state["lengths"]
                if self._use_fsdp:
                    lengths_local = torch.minimum(
                        lengths_local,
                        torch.tensor(effective_len, device=lengths_local.device, dtype=lengths_local.dtype),
                    )
                max_trimmed_len = 0
                for i in range(B):
                    max_trimmed_len = max(
                        max_trimmed_len,
                        int(lengths_local[i].item()) - int(prompt_token_lens_local[i].item()),
                    )
                max_trimmed_len = max(0, min(max_trimmed_len, current_T_codes))

                gen_codes_trimmed = torch.zeros(B, max_trimmed_len, num_quantizers, device=self.device, dtype=torch.long)
                tts_valid_lens = torch.zeros(B, device=self.device, dtype=torch.long)
                for i in range(B):
                    prompt_len = int(prompt_token_lens_local[i].item())
                    copy_len = min(
                        max(0, int(lengths_local[i].item()) - prompt_len),
                        current_T_codes - prompt_len,
                    )
                    copy_len = max(0, min(copy_len, max_trimmed_len))
                    if copy_len > 0:
                        gen_codes_trimmed[i, :copy_len] = gen_codes[i, prompt_len:prompt_len + copy_len]
                    tts_valid_lens[i] = copy_len
                gen_codes = gen_codes_trimmed
        else:
            tts_valid_lens = torch.full((B,), gen_codes.shape[1], device=self.device, dtype=torch.long)

        ans = self.stt_model._post_inference(inference_state, prompt_token_lens)

        if decode_audio:
            if not incremental_audio_decoding:
                gen_audio_codes_lens = tts_valid_lens.clone()
                with fp32_precision(), torch.no_grad():
                    audio_pred, audio_pred_len = self.tts_model.audio_codec.decode(
                        gen_codes, gen_audio_codes_lens
                    )
            ans["audio"] = audio_pred.squeeze(1)
            ans["audio_len"] = audio_pred_len

        return ans

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
