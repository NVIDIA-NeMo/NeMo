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
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf, open_dict
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(
    model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32, trust_remote_code: bool = False
):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Args:
        model_path_or_name: Path or name of the model to load
        pretrained_weights: Whether to load pretrained weights (True) or random init (False)
        dtype: Data type for the model
        trust_remote_code: Whether to trust remote code when loading model (needed for some models like Nemotron)
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(
            model_path_or_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    else:
        config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=trust_remote_code)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.

    If user config specifies encoder parameters, they will override the pretrained model's config.
    """
    if pretrained_weights:
        # Save user-specified encoder config before loading pretrained model
        user_encoder_config = {}

        if 'encoder' in model.cfg.perception:
            user_encoder_config = OmegaConf.to_container(model.cfg.perception.encoder, resolve=True)

        pretrained_asr_path = os.environ.get("S2S_PRETRAINED_ASR", model.cfg.pretrained_asr)
        logging.info(f"setup_speech_encoder: loading ASR from {pretrained_asr_path}")
        asr = load_pretrained_nemo(ASRModel, pretrained_asr_path).eval()
        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size
            # Override with user-specified encoder parameters, e.g. initializiing a non-causal encoder for causal setup.
            if user_encoder_config:
                for key, value in user_encoder_config.items():
                    if value is not None:  # Only override if user explicitly set a value
                        model.cfg.perception.encoder[key] = value
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        model.perception.load_state_dict(asr.state_dict(), strict=False)
        # Expose RNNT decoder+joint for EOU/BOU turn-taking (object.__setattr__ avoids PyTorch submodule registration)
        if hasattr(asr, 'decoder') and hasattr(asr, 'joint'):
            object.__setattr__(model, '_rnnt_decoder', asr.decoder.eval())
            object.__setattr__(model, '_rnnt_joint', asr.joint.eval())
            object.__setattr__(model, '_rnnt_blank_id', getattr(asr.decoding, 'blank_id', 1024))
            logging.info(f"setup_speech_encoder: RNNT decoder+joint exposed (blank_id={model._rnnt_blank_id})")
    else:
        model.perception = AudioPerceptionModule(model.cfg.perception).train()


def setup_rnnt_from_combined_checkpoint(
    model: torch.nn.Module,
    rnnt_config: dict,
    tokenizer_dir: str = None,
):
    """
    Initialize RNNT decoder/joint module structure from config saved in a
    combined checkpoint (produced by ``combine_s2s_rnnt_checkpoint.py``).

    Unlike :func:`setup_rnnt_decoder_joint`, this does **not** load a ``.nemo``
    file.  It recreates the modules from the saved decoder/joint OmegaConf dicts
    and optionally restores the tokenizer from files on disk.  The actual
    *weights* are expected to be loaded separately via ``load_state_dict`` from
    the combined ``model.safetensors``.

    Args:
        model: The DuplexSTTModel instance.
        rnnt_config: The ``_rnnt_merge_info`` dict from the combined checkpoint's
            ``config.json``.  Must contain ``decoder_config`` and ``joint_config``.
        tokenizer_dir: Path to the ``rnnt_tokenizer/`` directory saved by the
            combine script.  If ``None``, tokenizer is set to ``None``.
    """
    decoder_cfg = rnnt_config.get("decoder_config")
    joint_cfg = rnnt_config.get("joint_config")
    if decoder_cfg is None or joint_cfg is None:
        logging.warning(
            "Combined checkpoint _rnnt_merge_info is missing decoder_config or "
            "joint_config.  Cannot initialise RNNT modules from combined checkpoint."
        )
        model.rnnt_decoder = None
        model.rnnt_joint = None
        model.rnnt_tokenizer = None
        return

    decoder_cfg = OmegaConf.create(decoder_cfg)
    joint_cfg = OmegaConf.create(joint_cfg)

    decoder_cls_name = rnnt_config.get("decoder_class", "nemo.collections.asr.modules.rnnt.RNNTDecoder")
    joint_cls_name = rnnt_config.get("joint_class", "nemo.collections.asr.modules.rnnt.RNNTJoint")

    def _import_cls(fqn: str):
        mod_path, cls_name = fqn.rsplit(".", 1)
        import importlib
        return getattr(importlib.import_module(mod_path), cls_name)

    try:
        decoder_cls = _import_cls(decoder_cls_name)
        joint_cls = _import_cls(joint_cls_name)
    except Exception as e:
        logging.warning("Failed to import RNNT decoder/joint class (%s): %s", e, e)
        model.rnnt_decoder = None
        model.rnnt_joint = None
        model.rnnt_tokenizer = None
        return

    model.rnnt_decoder = decoder_cls.from_config_dict(decoder_cfg)
    model.rnnt_joint = joint_cls.from_config_dict(joint_cfg)
    # Also expose underscore-prefixed aliases used by ga/eou turn-taking code
    vocab = getattr(model.rnnt_joint, 'vocabulary', None)
    blank_id = len(vocab) if vocab is not None else 1024
    object.__setattr__(model, '_rnnt_decoder', model.rnnt_decoder)
    object.__setattr__(model, '_rnnt_joint',   model.rnnt_joint)
    object.__setattr__(model, '_rnnt_blank_id', blank_id)
    logging.info(
        "Initialised RNNT decoder (%s) and joint (%s) module structure from combined checkpoint config. blank_id=%d",
        decoder_cls_name, joint_cls_name, blank_id,
    )

    model.rnnt_tokenizer = None
    if tokenizer_dir and os.path.isdir(tokenizer_dir):
        tok_cfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
        if os.path.isfile(tok_cfg_path):
            import json as _json
            with open(tok_cfg_path, "r") as f:
                tok_cfg = _json.load(f)
            try:
                from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
                import glob
                sp_files = glob.glob(os.path.join(tokenizer_dir, "*.model"))
                if sp_files:
                    model.rnnt_tokenizer = SentencePieceTokenizer(model_path=sp_files[0])
                    logging.info("Loaded RNNT SentencePiece tokenizer from: %s", sp_files[0])
                else:
                    logging.warning("No .model file found in %s, RNNT tokenizer not loaded.", tokenizer_dir)
            except Exception as e:
                logging.warning("Failed to load RNNT tokenizer from %s: %s", tokenizer_dir, e)
        else:
            logging.info("No tokenizer_config.json in %s, RNNT tokenizer not loaded.", tokenizer_dir)


def set_model_dict_for_partial_init(
    pretrained_dict: Dict[str, torch.Tensor], model_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Partially initialize a model's state dictionary with a pretrained state dictionary.
    This function safely copies compatible layers from a pretrained model into a new model,
    ignoring layers with mismatched shapes or missing keys.

    Steps:
        1. Remove layers from the pretrained dictionary if their shape does not match the target model.
        2. Keep only keys that exist in the target model.
        3. Update the model dictionary with the filtered pretrained weights.

    Args:
        pretrained_dict (Dict[str, torch.Tensor]):
            The state dictionary of the pretrained model.
        model_dict (Dict[str, torch.Tensor]):
            The state dictionary of the target model to be partially initialized.

    Returns:
        Dict[str, torch.Tensor]:
            The updated model state dictionary with compatible layers loaded from the pretrained dictionary.

    Example:
        >>> model_dict = model.state_dict()
        >>> pretrained_dict = load_checkpoint("pretrained_model.ckpt")
        >>> model_dict = set_model_dict_for_partial_init(pretrained_dict, model_dict)
        >>> model.load_state_dict(model_dict)
    """
    # 1. Remove layers where pretrained shape differs from model shape
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(f" | > Layer with shape mismatch in the model definition: {k}")

    # 2. Keep only keys that exist in the target model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 3. Update model dictionary with filtered pretrained layers
    model_dict.update(pretrained_dict)
    logging.info(f" | > {len(pretrained_dict)} / {len(model_dict)} layers are restored.")

    return model_dict


def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from disk.

    Supports loading checkpoints stored in either PyTorch (`.ckpt`, `.pt`) or
    SafeTensors (`.safetensors`) formats. All parameters are loaded onto CPU
    regardless of the original device.

    Args:
        checkpoint_path (str):
            Path to the checkpoint file. If the filename contains `.safetensors`,
            it is loaded using the SafeTensors backend; otherwise, it is assumed
            to be a PyTorch checkpoint containing a `state_dict` field.

    Returns:
        dict:
            A state dictionary mapping parameter names to tensors.
    """
    if ".safetensors" in checkpoint_path:
        checkpoint_state = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    return checkpoint_state
