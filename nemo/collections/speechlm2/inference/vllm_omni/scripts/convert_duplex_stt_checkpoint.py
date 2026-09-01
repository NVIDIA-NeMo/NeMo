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
Convert the DuplexSTT component of a NemotronVoiceChat checkpoint to vLLM format.

This script extracts weights from a HuggingFace-format NemotronVoiceChat
checkpoint with tensors such as:
- stt_model.llm.layers.*
- stt_model.lm_head.*
- stt_model.asr_head.*
- stt_model.embed_asr_tokens.*
- stt_model.function_head.*
- stt_model.embed_tokens.*

And converts them to a HuggingFace layout that can be loaded by vLLM with the
custom WeightsMapper defined in nemotron_duplex_h.py.

Which auxiliary channels a checkpoint carries varies:

- ``predict_user_text=True`` gives ``asr_head`` + ``embed_asr_tokens``
- ``use_function_head=True`` gives ``function_head`` (reusing ``embed_tokens``
  for its feedback, so it has no embedding table of its own)

The converter records whichever heads are present as
``use_asr_head`` / ``use_function_head``, because ``NemotronDuplexHForCausalLM``
has to decide which modules to build *before* it sees any weights.
"""

import argparse
import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from nemo.utils import logging


def load_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """
    Load a NemotronVoiceChat checkpoint state dict.

    Args:
        checkpoint_path: Path to a checkpoint directory, safetensors file, or PyTorch checkpoint file.

    Returns:
        Dictionary of tensor names to tensors
    """
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")

    if checkpoint_path.endswith('.safetensors'):
        logging.info(f"Loading safetensors from {checkpoint_path}")
        return load_file(checkpoint_path)
    else:
        logging.info(f"Loading PyTorch checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        elif 'model' in ckpt:
            return ckpt['model']
        else:
            return ckpt


def filter_tensors(state_dict: dict[str, torch.Tensor], prefixes_to_keep: list[str]) -> dict[str, torch.Tensor]:
    """
    Filter tensors to keep only those with specified prefixes.

    Args:
        state_dict: Full state dictionary
        prefixes_to_keep: List of prefixes to keep (e.g., ["stt_model.llm", "stt_model.asr_head"])

    Returns:
        Filtered state dictionary
    """
    filtered_dict = {}
    for name, tensor in state_dict.items():
        if any(name.startswith(prefix) for prefix in prefixes_to_keep):
            filtered_dict[name] = tensor
            logging.debug(f"Keeping: {name} with shape {tensor.shape}")
        else:
            logging.debug(f"Skipping: {name}")

    logging.info(f"Total tensors kept: {len(filtered_dict)}")
    return filtered_dict


def _apply_source_special_tokens(base_config, tokenizer, source_config: dict | None) -> None:
    """Match the converted tokenizer/config to the VoiceChat channel tokens.

    VoiceChat training overrides the LLM-backbone tokenizer specials
    (typically ``</s>`` for EOS and ``<SPECIAL_12>`` for PAD). Keeping the
    backbone's original EOS/PAD ids corrupts system-prompt prefill even
    though text tokenization itself appears valid. Copy whatever the source
    VoiceChat config actually used.
    """
    try:
        model_config = source_config["model"]["stt"]["model"]
    except (KeyError, TypeError):
        return

    overrides = model_config.get("override_tokens", {}) or {}
    special_tokens = {
        name: overrides.get(name) or model_config.get(name)
        for name in ("bos_token", "eos_token", "pad_token")
    }
    special_tokens = {name: token for name, token in special_tokens.items() if token}
    if not special_tokens:
        return

    vocabulary = tokenizer.get_vocab()
    missing = [token for token in special_tokens.values() if token not in vocabulary]
    if missing:
        raise ValueError(
            "VoiceChat special tokens must already exist in the backbone vocabulary; "
            f"missing={missing}"
        )
    added = tokenizer.add_special_tokens(special_tokens)
    if added:
        raise ValueError(
            "VoiceChat special-token overrides unexpectedly expanded the vocabulary; "
            f"added={added}"
        )

    for name, token in special_tokens.items():
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        setattr(base_config, f"{name}_id", token_id)
    logging.info(
        "VoiceChat special-token IDs: bos=%s eos=%s pad=%s",
        getattr(base_config, "bos_token_id", None),
        getattr(base_config, "eos_token_id", None),
        getattr(base_config, "pad_token_id", None),
    )


def convert_to_vllm_format(
    checkpoint_path: str,
    output_dir: str,
    config_path: str | None = None,
    pretrained_llm: str | None = None,
    tensors_to_keep: list[str] | None = None,
    dtype: str = "float32",
) -> None:
    """
    Convert the DuplexSTT component to vLLM-compatible HuggingFace format.

    Args:
        checkpoint_path: Path to the NeMo checkpoint (.safetensors or .pt)
        output_dir: Directory to save the converted checkpoint
        config_path: Path to config.json (if None, will look in same dir as checkpoint)
        pretrained_llm: HuggingFace model name to get base config from
        tensors_to_keep: List of tensor prefixes to keep (default: all stt_model.* tensors)
        dtype: Data type for tensors ("float32", "float16", "bfloat16")
    """
    # Default prefixes to keep. The auxiliary-channel entries are only present
    # in some checkpoints; absent ones simply match nothing.
    if tensors_to_keep is None:
        tensors_to_keep = [
            "stt_model.llm",
            "stt_model.lm_head",
            "stt_model.asr_head",
            "stt_model.embed_asr_tokens",
            "stt_model.function_head",
            "stt_model.embed_tokens",
        ]

    # Load config to get pretrained_llm if not provided
    if config_path is None:
        ckpt_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        config_path = os.path.join(ckpt_dir, "config.json")

    config = None
    if os.path.exists(config_path):
        logging.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)

        try:
            pretrained_llm = config["model"]["stt"]["model"]["pretrained_llm"]
            logging.info(f"Found pretrained_llm in config: {pretrained_llm}")
        except KeyError:
            if pretrained_llm is None:
                raise ValueError("Could not find pretrained_llm in config and none provided via argument")
    else:
        if pretrained_llm is None:
            raise ValueError(f"Config file not found at {config_path} and pretrained_llm not provided")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base config from pretrained model
    logging.info(f"Loading base config from {pretrained_llm}")
    base_config = AutoConfig.from_pretrained(pretrained_llm, trust_remote_code=True)

    # Load tokenizer from pretrained model
    logging.info(f"Loading tokenizer from {pretrained_llm}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_llm, trust_remote_code=True)
    _apply_source_special_tokens(base_config, tokenizer, config)

    # Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path)

    # Filter tensors
    logging.info(f"Filtering tensors to keep prefixes: {tensors_to_keep}")
    filtered_state_dict = filter_tensors(state_dict, tensors_to_keep)

    if len(filtered_state_dict) == 0:
        raise ValueError(
            f"No tensors found with prefixes {tensors_to_keep}. "
            f"Available prefixes: {set(k.split('.')[0] for k in state_dict.keys())}"
        )

    # Record which auxiliary channels this checkpoint actually carries, so the
    # vLLM model builds exactly those modules. Detected from the weights that
    # made it through the filter rather than from the source config, so a
    # narrowed --tensors-to-keep stays consistent with what gets saved.
    has_asr_head = any(name.startswith("stt_model.asr_head") for name in filtered_state_dict)
    has_function_head = any(name.startswith("stt_model.function_head") for name in filtered_state_dict)

    if has_asr_head and not any(name.startswith("stt_model.embed_asr_tokens") for name in filtered_state_dict):
        raise ValueError(
            "Checkpoint has stt_model.asr_head but no stt_model.embed_asr_tokens; "
            "the ASR channel needs both (the head to predict the token and the "
            "embedding table to feed it back on the next step)."
        )

    # The function channel scales its feedback embedding by this weight, matching
    # DuplexSTTModel.build_input_embedding.
    function_channel_weight = 1.0
    if has_function_head and config is not None:
        try:
            function_channel_weight = float(config["model"]["stt"]["model"].get("duplex_function_channel_weight", 1.0))
        except (KeyError, TypeError):
            logging.warning("Could not read duplex_function_channel_weight from source config; defaulting to 1.0")

    custom_outputs = ["text_logits"]
    if has_asr_head:
        custom_outputs += ["asr_tokens", "asr_logits"]
    if has_function_head:
        custom_outputs += ["function_tokens", "function_logits"]

    base_config.update(
        {
            "custom_input_specs": [{"name": "combined_embeds", "dtype": dtype, "dim": base_config.hidden_size}],
            "custom_outputs": custom_outputs,
            "use_asr_head": has_asr_head,
            "use_function_head": has_function_head,
            "duplex_function_channel_weight": function_channel_weight,
        }
    )
    logging.info(
        f"Auxiliary channels: asr_head={has_asr_head}, function_head={has_function_head} "
        f"(function_channel_weight={function_channel_weight})"
    )

    # Save tensors
    output_model_path = output_path / "model.safetensors"
    logging.info(f"Saving tensors to {output_model_path}")
    save_file(filtered_state_dict, str(output_model_path))

    # Save config
    output_config_path = output_path / "config.json"
    logging.info(f"Saving config to {output_config_path}")
    base_config.save_pretrained(str(output_path))

    # Save tokenizer
    logging.info(f"Saving tokenizer to {output_path}")
    tokenizer.save_pretrained(str(output_path))

    logging.info(f"Conversion completed successfully! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo STT checkpoint to HuggingFace format for vLLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NeMo checkpoint file (.safetensors or .pt/.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save converted checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: same directory as checkpoint)",
    )
    parser.add_argument(
        "--pretrained-llm",
        type=str,
        default=None,
        help="HuggingFace model name to use as base (default: read from config)",
    )
    parser.add_argument(
        "--tensors-to-keep",
        type=str,
        nargs="+",
        default=None,
        help="Tensor prefixes to keep (default: all stt_model.* backbone llm related tensors)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"],
        help="Target dtype for tensors (default: float32)",
    )

    args = parser.parse_args()

    convert_to_vllm_format(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        pretrained_llm=args.pretrained_llm,
        tensors_to_keep=args.tensors_to_keep,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
