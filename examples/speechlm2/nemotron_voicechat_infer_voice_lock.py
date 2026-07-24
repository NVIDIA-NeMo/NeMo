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
#
# Voice-lock variant of nemotron_voicechat_infer.py.
# Adds two steps before save_pretrained:
#   1. register_speaker_dict  — pre-bakes named speaker latents into weights
#                               using the still-valid audio_prompt_projection_W.
#   2. reinit_audio_prompt_frozen_projection — randomizes audio_prompt_projection_W
#                               so any new user-supplied audio prompt produces
#                               incoherent conditioning (only pre-baked latents work).
#
# Both flags are read from the Hydra config (same as nemotron_voicechat_to_hf.py).
import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset
from nemo.collections.speechlm2.models.duplex_ear_tts import load_audio_librosa
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.collections.audio.parts.utils.resampling import resample
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Same as used by HF save/load (config.json in checkpoint dir)
HF_CONFIG_NAME = "config.json"

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _merge_stt_config_from_checkpoint(cfg, model_config):
    """
    S2T-style flow: load STT config from checkpoint (if HF dir), then merge
    inference overrides on top. Modifies model_config in place so that
    model.stt.model = merge(ckpt_config["model"], inference model.stt.model).
    """
    stt_ckpt = (
        OmegaConf.select(cfg, "model.stt.model.pretrained_s2s_model", default=None)
        or OmegaConf.select(cfg, "model.pretrained_s2s_model", default=None)
    )
    stt_ckpt = str(stt_ckpt) if stt_ckpt else None
    if not stt_ckpt or not os.path.isdir(stt_ckpt):
        return
    config_path = os.path.join(stt_ckpt, HF_CONFIG_NAME)
    if not os.path.isfile(config_path):
        logging.warning(
            "[nemotron_voicechat_infer] No %s in STT checkpoint dir %s; using inference config only.",
            HF_CONFIG_NAME,
            stt_ckpt,
        )
        return
    ckpt_config = OmegaConf.load(config_path)
    ckpt_model = ckpt_config.get("model", ckpt_config)
    if not hasattr(ckpt_model, "keys") and not isinstance(ckpt_model, dict):
        ckpt_model = OmegaConf.to_container(ckpt_model, resolve=True)
    else:
        ckpt_model = OmegaConf.to_container(OmegaConf.create(ckpt_model), resolve=True)
    inference_stt_model = model_config.get("model", {}).get("stt", {}).get("model", {})
    merged = OmegaConf.merge(OmegaConf.create(ckpt_model), OmegaConf.create(inference_stt_model))
    model_config["model"]["stt"]["model"] = OmegaConf.to_container(merged, resolve=True)
    # Match S2T: do not load pretrained LLM/ASR; we load full checkpoint weights instead
    model_config["model"]["stt"]["model"]["pretrained_weights"] = False
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logging.info(
            "[nemotron_voicechat_infer] Merged STT config: checkpoint=%s, inference overrides on top (S2T-style).",
            stt_ckpt,
        )


def _clear_hardcoded_paths(model_config):
    """Clear paths that are training/export artifacts and not needed at inference."""

    def _set(d, *keys, value=""):
        for k in keys[:-1]:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict) and keys[-1] in d:
            d[keys[-1]] = value

    # data paths
    _set(model_config, "data", "train_ds", "real_conv_data", "input_cfg")
    _set(model_config, "data", "train_ds", "noise_path")
    _set(model_config, "data", "validation_ds", "datasets", "ultrachat", "shar_path")

    # top-level fields
    _set(model_config, "hf_export_dir")
    _set(model_config, "exp_manager", "explicit_log_dir")
    _set(model_config, "exp_manager", "name")
    _set(model_config, "exp_manager", "wandb_logger_kwargs", "name")
    _set(model_config, "exp_manager", "wandb_logger_kwargs", "project")

    # top-level model paths
    _set(model_config, "model", "audio_save_path")
    _set(model_config, "model", "inference_speaker_reference")

    # STT model paths
    _set(model_config, "model", "stt", "model", "audio_save_path")
    _set(model_config, "model", "stt", "model", "backchannel_debug_path")
    _set(model_config, "model", "stt", "model", "backchannel_file_path")
    _set(model_config, "model", "stt", "model", "micir_aug_path")
    _set(model_config, "model", "stt", "model", "nemo_text_norm_cache_dir")
    _set(model_config, "model", "stt", "model", "nsynth_noise_aug_path")
    _set(model_config, "model", "stt", "model", "old_noise_aug_path")
    model_config.get("model", {}).get("stt", {}).get("model", {}).pop("pretrained_audio_codec", None)
    _set(model_config, "model", "stt", "model", "pretrained_s2s_model")
    _set(model_config, "model", "stt", "model", "roomir_aug_path")

    # TTS paths
    _set(model_config, "model", "speech_generation", "model", "audio_save_path")
    _set(model_config, "model", "speech_generation", "model", "pretrained_model")

    # RNNT merge info paths (only present in s2s_rnnt checkpoints)
    _set(model_config, "_rnnt_merge_info", "s2s_source")
    _set(model_config, "_rnnt_merge_info", "rnnt_source")
    _set(model_config, "_rnnt_merge_info", "rnnt_source_checkpoint")

    logging.info("Hardcoded training/export paths cleared.")


def _register_speakers(model, register_speaker_dict):
    """Pre-bake named speaker latents into model weights using the current (valid) W."""
    model.tts_model.to(model.device)
    for speaker_name, audio_path in register_speaker_dict.items():
        speaker_audio, sr = load_audio_librosa(audio_path)
        speaker_audio = resample(speaker_audio, sr, model.tts_model.target_sample_rate).to(model.device)
        speaker_audio_lens = (
            torch.tensor([speaker_audio.size(1)]).long().repeat(speaker_audio.size(0)).to(model.device)
        )
        model.tts_model.set_audio_prompt_latent(
            speaker_audio,
            speaker_audio_lens,
            system_prompt=None,
            batch_size=1,
            name=speaker_name,
        )
        logging.info("Speaker '%s' registered from %s", speaker_name, audio_path)


def _make_perception_self_contained(model, model_config):
    """
    Copy fully-resolved perception config (preprocessor + encoder + output_dim)
    from the loaded model into model_config, and clear external file references
    that are redundant once weights are in model.safetensors:
      - pretrained_asr: ASR .nemo no longer needed — architecture is now in config
      - pretrained_codec_model: codec weights are already in model.safetensors
    """
    perception_cfg = OmegaConf.to_container(model.stt_model.cfg.perception, resolve=True)
    stt = model_config["model"]["stt"]["model"]
    stt["perception"]["preprocessor"] = perception_cfg["preprocessor"]
    stt["perception"]["encoder"] = perception_cfg["encoder"]
    stt["perception"]["output_dim"] = perception_cfg["output_dim"]
    stt["pretrained_asr"] = ""
    stt["pretrained_weights"] = False

    tts = model_config["model"]["speech_generation"]["model"]
    tts.pop("pretrained_codec_model", None)

    logging.info("Perception config made self-contained; pretrained_asr and pretrained_codec_model cleared.")


def _reinit_audio_prompt_projection(model):
    """Randomize audio_prompt_projection_W to disable voice cloning from arbitrary audio."""
    D = model.tts_model.tts_model.hidden_size
    model.tts_model.tts_model.audio_prompt_projection_W.copy_(
        torch.randn(
            D,
            D,
            device=model.tts_model.tts_model.audio_prompt_projection_W.device,
            dtype=model.tts_model.tts_model.audio_prompt_projection_W.dtype,
        )
    )
    logging.info("Audio frozen projection reinitialized!")


@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model_config = OmegaConf.to_container(cfg, resolve=True)
        # S2T-style: STT config = checkpoint config first, then inference YAML overrides
        _merge_stt_config_from_checkpoint(cfg, model_config)
        model = NemotronVoiceChat(model_config)

        # load pretrained TTS checkpoint if available
        if model.tts_model.cfg.get("pretrained_model", None):
            model.tts_model.restore_from_pretrained_checkpoint(model.tts_model.cfg.pretrained_model)

    # Save merged config (checkpoint + inference overrides) as soon as model is ready
    OmegaConf.save(OmegaConf.create(model_config), log_dir / "merged_config.yaml")
    logging.info("Saving merged config (from model_config used to build model) to %s", log_dir / "merged_config.yaml")

    # Log which checkpoints were loaded (rank 0 only)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        stt_ckpt = getattr(model.stt_model, "cfg", None) and model.stt_model.cfg.get("pretrained_s2s_model", None)
        tts_ckpt = getattr(model.tts_model, "cfg", None) and model.tts_model.cfg.get("pretrained_model", None)
        speaker_ref = getattr(model, "cfg", None) and model.cfg.get("inference_speaker_reference", None)
        logging.info("=" * 60)
        logging.info("[nemotron_voicechat_infer_voice_lock] Loaded checkpoints:")
        logging.info("  STT (pretrained_s2s_model): %s", stt_ckpt or "(none)")
        logging.info("  TTS (pretrained_model):     %s", tts_ckpt or "(none)")
        logging.info("  Speaker reference:         %s", speaker_ref or "(none)")
        logging.info("=" * 60)

    model_cfg = (
        OmegaConf.to_container(model.stt_model.cfg, resolve=True)
        if hasattr(model, "stt_model") and hasattr(model.stt_model, "cfg")
        else {}
    )

    dataset = DuplexS2SDataset(
        tokenizer=model.stt_model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        include_turn_metadata=True,
        force_align_user_text=False,
        cfg=OmegaConf.to_container(cfg.data, resolve=True),
        model_cfg=model_cfg,
        early_interruption_prob=0.0,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.stt_model.tokenizer, dataset=dataset)

    hf_export_dir = model_config.get("hf_export_dir", None)
    if hf_export_dir:
        # Pop export-time flags so they are not written into config.json
        register_speaker_dict = model_config.pop("register_speaker_dict", {})
        reinit_projection = model_config.pop("reinit_audio_prompt_frozen_projection", False)

        # Step 1: register speaker latents while W is still valid
        if register_speaker_dict:
            _register_speakers(model, register_speaker_dict)

        # Step 2: destroy W so arbitrary audio prompts no longer work
        if reinit_projection:
            _reinit_audio_prompt_projection(model)

        # Step 3: make perception config self-contained (no ASR .nemo needed at load time)
        _make_perception_self_contained(model, model_config)

        # Step 4: clear hardcoded training/export paths
        _clear_hardcoded_paths(model_config)

        model.save_pretrained(hf_export_dir, config=model_config)
        print("Hugging face compatible checkpoint saved at:", hf_export_dir)

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    inference()
