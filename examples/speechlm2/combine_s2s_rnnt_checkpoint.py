#!/usr/bin/env python3
"""
Combine an S2S (STT+TTS) checkpoint with RNN-T decoder/joint weights
into a single HuggingFace-style checkpoint directory.

Inputs:
  --s2s_checkpoint_dir   Path to the existing S2S HF checkpoint directory
                         (must contain model.safetensors and config.json)
  --rnnt_nemo_path       Path to a .nemo ASR checkpoint with RNNT decoder/joint
  --output_dir           Where to write the combined checkpoint

Output directory will contain:
  model.safetensors      Combined weights (S2S + stt_model.rnnt_decoder.* + stt_model.rnnt_joint.*)
  config.json            Copied from the S2S checkpoint (with rnnt metadata appended)
  rnnt_tokenizer/        Tokenizer files extracted from the .nemo checkpoint

Usage:
  python combine_s2s_rnnt_checkpoint.py \
      --s2s_checkpoint_dir /path/to/s2s_hf_ckpt \
      --rnnt_nemo_path /path/to/asr_rnnt.nemo \
      --output_dir /path/to/combined_ckpt
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import torch


def load_rnnt_weights_from_nemo(nemo_path: str):
    """
    Extract RNNT decoder and joint state dicts from a .nemo ASR checkpoint.

    Returns:
        rnnt_state_dict: dict with keys prefixed as stt_model.rnnt_decoder.* / stt_model.rnnt_joint.*
        tokenizer_dir: path to a temp directory containing the tokenizer files
                       (caller is responsible for copying and cleaning up)
        rnnt_info: dict with metadata about the RNNT model (vocab_size, etc.)
    """
    from nemo.collections.asr.models import ASRModel

    print(f"Loading RNNT ASR model from: {nemo_path}")
    t0 = time.time()
    asr_model = ASRModel.restore_from(nemo_path, map_location="cpu")
    asr_model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if not (hasattr(asr_model, "decoder") and hasattr(asr_model, "joint")):
        raise ValueError(
            f"The ASR checkpoint at {nemo_path} does not have decoder/joint attributes. "
            f"Got model type: {type(asr_model).__name__}. "
            f"Make sure it is an RNNT-based ASR model."
        )

    asr_sd = asr_model.state_dict()

    rnnt_state_dict = {}
    decoder_count = 0
    joint_count = 0
    for key, value in asr_sd.items():
        if key.startswith("decoder."):
            rnnt_state_dict[f"stt_model.rnnt_decoder.{key[len('decoder.'):]}"] = value
            decoder_count += 1
        elif key.startswith("joint."):
            rnnt_state_dict[f"stt_model.rnnt_joint.{key[len('joint.'):]}"] = value
            joint_count += 1

    print(f"  Extracted {decoder_count} decoder params, {joint_count} joint params")
    print(f"  Total RNNT params: {sum(v.numel() for v in rnnt_state_dict.values()):,}")

    from omegaconf import OmegaConf, open_dict

    with open_dict(asr_model.cfg.decoder):
        if getattr(asr_model.cfg.decoder, "vocab_size", None) is None and hasattr(asr_model, "joint"):
            asr_model.cfg.decoder.vocab_size = len(asr_model.joint.vocabulary)
    with open_dict(asr_model.cfg.joint):
        if getattr(asr_model.cfg.joint, "num_classes", None) is None and hasattr(asr_model.joint, "vocabulary"):
            asr_model.cfg.joint.num_classes = len(asr_model.joint.vocabulary)
        if getattr(asr_model.cfg.joint, "vocabulary", None) is None and hasattr(asr_model.joint, "vocabulary"):
            asr_model.cfg.joint.vocabulary = asr_model.joint.vocabulary

    decoder_cfg_dict = OmegaConf.to_container(asr_model.cfg.decoder, resolve=True)
    joint_cfg_dict = OmegaConf.to_container(asr_model.cfg.joint, resolve=True)
    decoder_cls_fqn = f"{type(asr_model.decoder).__module__}.{type(asr_model.decoder).__name__}"
    joint_cls_fqn = f"{type(asr_model.joint).__module__}.{type(asr_model.joint).__name__}"

    print(f"  Decoder class: {decoder_cls_fqn}")
    print(f"  Joint class:   {joint_cls_fqn}")

    rnnt_info = {
        "rnnt_source_checkpoint": os.path.basename(nemo_path),
        "rnnt_decoder_params": decoder_count,
        "rnnt_joint_params": joint_count,
        "decoder_config": decoder_cfg_dict,
        "joint_config": joint_cfg_dict,
        "decoder_class": decoder_cls_fqn,
        "joint_class": joint_cls_fqn,
    }
    if hasattr(asr_model.joint, "vocabulary"):
        rnnt_info["rnnt_vocab_size"] = len(asr_model.joint.vocabulary)

    tokenizer_dir = None
    if hasattr(asr_model, "tokenizer") and asr_model.tokenizer is not None:
        tokenizer_dir = tempfile.mkdtemp(prefix="rnnt_tokenizer_")
        tokenizer = asr_model.tokenizer
        sp_saved = False

        sp = getattr(tokenizer, "tokenizer", None)
        if sp is not None and hasattr(sp, "serialized_model_proto"):
            model_path = os.path.join(tokenizer_dir, "tokenizer.model")
            with open(model_path, "wb") as f:
                f.write(sp.serialized_model_proto())
            print(f"  Saved SentencePiece model via serialized_model_proto()")
            sp_saved = True

        if not sp_saved and hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file and os.path.exists(tokenizer.vocab_file):
            shutil.copy2(tokenizer.vocab_file, os.path.join(tokenizer_dir, os.path.basename(tokenizer.vocab_file)))
            print(f"  Copied tokenizer vocab file: {os.path.basename(tokenizer.vocab_file)}")

        if hasattr(tokenizer, "vocab"):
            vocab_path = os.path.join(tokenizer_dir, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(tokenizer.vocab if isinstance(tokenizer.vocab, dict) else list(tokenizer.vocab), f)

        if hasattr(asr_model.cfg, "tokenizer"):
            tok_cfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
            from omegaconf import OmegaConf
            with open(tok_cfg_path, "w") as f:
                json.dump(OmegaConf.to_container(asr_model.cfg.tokenizer, resolve=True), f, indent=2)
            print(f"  Saved tokenizer config")

    del asr_model
    return rnnt_state_dict, tokenizer_dir, rnnt_info


def main():
    parser = argparse.ArgumentParser(
        description="Combine S2S checkpoint with RNNT decoder/joint into a single checkpoint directory."
    )
    parser.add_argument(
        "--s2s_checkpoint_dir", type=str, required=True,
        help="Path to existing S2S HF checkpoint (with model.safetensors and config.json)",
    )
    parser.add_argument(
        "--rnnt_nemo_path", type=str, required=True,
        help="Path to .nemo ASR checkpoint with RNNT decoder and joint",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for combined checkpoint",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output_dir if it already exists",
    )
    args = parser.parse_args()

    safetensors_path = os.path.join(args.s2s_checkpoint_dir, "model.safetensors")
    config_path = os.path.join(args.s2s_checkpoint_dir, "config.json")

    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(f"model.safetensors not found in {args.s2s_checkpoint_dir}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {args.s2s_checkpoint_dir}")
    if not os.path.isfile(args.rnnt_nemo_path):
        raise FileNotFoundError(f"RNNT .nemo file not found: {args.rnnt_nemo_path}")

    if os.path.exists(args.output_dir):
        if args.overwrite:
            print(f"Output dir exists, overwriting: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"Output dir already exists: {args.output_dir}. Use --overwrite to replace."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Load RNNT weights from .nemo ---
    print("\n=== Step 1: Extracting RNNT weights from .nemo ===")
    rnnt_sd, tokenizer_tmp_dir, rnnt_info = load_rnnt_weights_from_nemo(args.rnnt_nemo_path)

    # --- Step 2: Load S2S safetensors ---
    print("\n=== Step 2: Loading S2S checkpoint ===")
    from safetensors.torch import load_file, save_file

    t0 = time.time()
    s2s_sd = load_file(safetensors_path)
    print(f"  Loaded {len(s2s_sd)} tensors in {time.time() - t0:.1f}s")
    print(f"  Total S2S params: {sum(v.numel() for v in s2s_sd.values()):,}")

    existing_rnnt_keys = [k for k in s2s_sd if "rnnt_decoder" in k or "rnnt_joint" in k]
    if existing_rnnt_keys:
        print(f"  WARNING: S2S checkpoint already contains {len(existing_rnnt_keys)} RNNT keys. "
              f"They will be overwritten.")

    # --- Step 3: Merge ---
    print("\n=== Step 3: Merging weights ===")
    combined_sd = {**s2s_sd, **rnnt_sd}
    print(f"  Combined tensor count: {len(combined_sd)}")
    print(f"  Combined total params: {sum(v.numel() for v in combined_sd.values()):,}")

    del s2s_sd, rnnt_sd

    # --- Step 4: Save combined safetensors ---
    print("\n=== Step 4: Saving combined checkpoint ===")
    output_safetensors = os.path.join(args.output_dir, "model.safetensors")
    t0 = time.time()
    save_file(combined_sd, output_safetensors)
    file_size_gb = os.path.getsize(output_safetensors) / (1024 ** 3)
    print(f"  Saved {output_safetensors} ({file_size_gb:.2f} GB) in {time.time() - t0:.1f}s")

    del combined_sd

    # --- Step 5: Copy and update config.json ---
    print("\n=== Step 5: Saving config ===")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["_rnnt_merge_info"] = {
        "s2s_source": "",
        "rnnt_source": "",
        **rnnt_info,
        "rnnt_source_checkpoint": "",
    }

    output_config = os.path.join(args.output_dir, "config.json")
    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved {output_config}")

    # --- Step 6: Copy tokenizer ---
    if tokenizer_tmp_dir and os.path.isdir(tokenizer_tmp_dir):
        output_tokenizer_dir = os.path.join(args.output_dir, "rnnt_tokenizer")
        shutil.copytree(tokenizer_tmp_dir, output_tokenizer_dir)
        shutil.rmtree(tokenizer_tmp_dir)
        tok_files = os.listdir(output_tokenizer_dir)
        print(f"  Saved tokenizer ({len(tok_files)} files) to {output_tokenizer_dir}/")
        for tf in tok_files:
            print(f"    {tf}")

    # --- Step 7: Copy any other files from the source checkpoint ---
    print("\n=== Step 6: Copying other checkpoint files ===")
    for fname in os.listdir(args.s2s_checkpoint_dir):
        if fname in ("model.safetensors", "config.json"):
            continue
        src = os.path.join(args.s2s_checkpoint_dir, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
            print(f"  Copied {fname}/")

    # --- Done ---
    print(f"\n{'=' * 60}")
    print(f"Combined checkpoint saved to: {args.output_dir}")
    print(f"{'=' * 60}")
    print(f"\nContents:")
    for item in sorted(os.listdir(args.output_dir)):
        full = os.path.join(args.output_dir, item)
        if os.path.isdir(full):
            print(f"  {item}/")
        else:
            size_mb = os.path.getsize(full) / (1024 ** 2)
            print(f"  {item}  ({size_mb:.1f} MB)")

    print(f"\nTo use this checkpoint, set both model_path and llm_checkpoint_path "
          f"to:\n  {os.path.abspath(args.output_dir)}")
    print(f"\nNote: The RNNT weights are embedded in model.safetensors as "
          f"stt_model.rnnt_decoder.* and stt_model.rnnt_joint.* keys.")
    print(f"The loading code still needs pretrained_rnnt_asr to initialize the module "
          f"structure (decoder/joint architecture), but the weights will be overwritten "
          f"by the embedded ones during load_state_dict.")


if __name__ == "__main__":
    main()
