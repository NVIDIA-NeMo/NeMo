# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Evaluate an EasyMagpie checkpoint with either one-shot acoustic estimator.

This intentionally uses ``EasyMagpieTTSModel.validation_step`` so WER and
speaker similarity are computed exactly as during training: Whisper large-v3
for multilingual ASR and TitaNet-large cosine similarity against context audio.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.tts.models import EasyMagpieTTSModel
from nemo.utils import logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hparams-file",
        type=Path,
        required=True,
        help="W&B config.yaml or plain model config",
    )
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--codecmodel-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acoustic-inference-mode",
        choices=("flow", "aux_projection"),
        required=True,
        help="Regular flow sampling or direct prediction from the trained auxiliary projection.",
    )
    parser.add_argument("--tokenizer-name", default="nemotron_nano_30b")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Per-GPU validation batch size"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Optional smoke-test batch limit",
    )
    return parser.parse_args()


def _require_readable(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a readable file: {path}")


def _load_model_config(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if "cfg" in loaded:
        loaded = loaded.cfg
    if isinstance(loaded, DictConfig) and "value" in loaded:
        loaded = loaded.value
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            f"Expected a model DictConfig in {path}, got {type(loaded).__name__}"
        )
    return OmegaConf.create(OmegaConf.to_container(loaded, resolve=True))


def _prepare_model_config(args: argparse.Namespace) -> DictConfig:
    model_cfg = _load_model_config(args.hparams_file)
    validation_ds = {
        "dataset": {
            "_target_": "nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDataset",
            "min_duration": 0.2,
            "max_duration": 20.0,
            "dataset_meta": {
                "english_validation": {
                    "manifest_path": str(args.manifest_path),
                    "audio_dir": str(args.audio_dir),
                    "tokenizer_names": [args.tokenizer_name],
                }
            },
        },
        "dataloader_params": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": True,
        },
    }
    with open_dict(model_cfg):
        model_cfg.codecmodel_path = str(args.codecmodel_path)
        model_cfg.train_ds = None
        model_cfg.validation_ds = OmegaConf.create(validation_ds)
        model_cfg.run_val_inference = True
        model_cfg.use_multilingual_asr = True
        model_cfg.use_utmos = False
        model_cfg.oneshot_acoustic_inference_mode = args.acoustic_inference_mode
        # Validation loss is not part of this comparison. One noise sample keeps
        # its unavoidable teacher-forced computation small without changing inference.
        model_cfg.local_flow_matching_train_num_noise_samples = 1
    return model_cfg


def _load_checkpoint(model: EasyMagpieTTSModel, checkpoint_file: Path) -> None:
    logging.info("Loading checkpoint weights from %s", checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_file}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint


def main() -> None:
    args = _parse_args()
    _require_readable(args.hparams_file, "hparams")
    _require_readable(args.checkpoint_file, "checkpoint")
    _require_readable(args.codecmodel_path, "codec")
    _require_readable(args.manifest_path, "manifest")
    if not args.audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory is not readable: {args.audio_dir}")
    if args.batch_size < 1 or args.devices < 1 or args.num_workers < 0:
        raise ValueError(
            "batch-size/devices must be positive and num-workers must be non-negative"
        )

    mp.set_start_method("spawn", force=True)
    pl.seed_everything(args.seed, workers=True)
    mode_output_dir = args.output_dir / args.acoustic_inference_mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = {
        "accelerator": "gpu",
        "devices": args.devices,
        "num_nodes": 1,
        "strategy": "ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        "precision": "bf16-mixed",
        "logger": False,
        "enable_checkpointing": False,
        "default_root_dir": str(mode_output_dir),
        "use_distributed_sampler": False,
    }
    if args.limit_val_batches is not None:
        if args.limit_val_batches < 1:
            raise ValueError("limit-val-batches must be positive")
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    model_cfg = _prepare_model_config(args)
    logging.info("Acoustic inference mode: %s", args.acoustic_inference_mode)
    model = EasyMagpieTTSModel(cfg=model_cfg, trainer=trainer)
    _load_checkpoint(model, args.checkpoint_file)

    results = trainer.validate(model, verbose=True)
    if trainer.is_global_zero:
        result = results[0] if results else {}
        summary = {
            "acoustic_inference_mode": args.acoustic_inference_mode,
            "seed": args.seed,
            "manifest_path": str(args.manifest_path),
            "manifest_records": sum(
                1 for line in args.manifest_path.open(encoding="utf-8") if line.strip()
            ),
            "checkpoint_file": str(args.checkpoint_file),
            "wer": result.get("val/wer_lang_en", result.get("val/wer")),
            "ssim": result.get("val/ssim"),
            "all_validation_metrics": result,
        }
        summary_path = mode_output_dir / "metrics.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + os.linesep, encoding="utf-8"
        )
        logging.info("Wrote evaluation summary to %s", summary_path)


if __name__ == "__main__":
    main()
