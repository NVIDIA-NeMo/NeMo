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

"""Build a fresh EasyMagpie one-shot-flow seed from a semantic checkpoint.

The target model is initialized from the requested Hydra config. Only the
Nemotron-H backbone (``decoder.*``) and the semantic rows of ``final_proj`` are
then copied from the source checkpoint. All other target-model weights retain
their fresh random initialization.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_CONFIG_NAME = "easy_magpietts_lhotse_oneshot_flow"
DEFAULT_OUTPUT_NAME = "easy_magpietts_oneshot_flow_seed.nemo"


def _source_key_candidates(target_key: str) -> list[str]:
    """Return common checkpoint spellings for one EasyMagpie state key."""

    base_keys = [target_key]
    if target_key.startswith("decoder."):
        base_keys.append(f"backbone.{target_key.removeprefix('decoder.')}")

    prefixes = ("", "model.", "module.", "model.module.")
    return [f"{prefix}{base_key}" for base_key in base_keys for prefix in prefixes]


def _get_source_tensor(source_state: Mapping[str, Any], target_key: str) -> tuple[str, torch.Tensor]:
    matches = [key for key in _source_key_candidates(target_key) if key in source_state]
    if not matches:
        raise ValueError(
            f"Source checkpoint is missing target tensor {target_key!r}. "
            f"Tried keys: {_source_key_candidates(target_key)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Source checkpoint has ambiguous matches for {target_key!r}: {matches}")

    source_key = matches[0]
    tensor = source_state[source_key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Source checkpoint entry {source_key!r} is {type(tensor).__name__}, not a tensor.")
    return source_key, tensor


def transfer_backbone_and_semantic_projection(
    target_model: nn.Module,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only the backbone and semantic-output rows into ``target_model``.

    EasyMagpie packs codebook heads contiguously into ``final_proj``. Semantic
    codebooks are first, so the transferable row count is::

        num_semantic_codebooks * frame_stacking_factor * num_all_tokens_per_codebook

    Every shape is validated before any target tensor is changed.
    """

    if getattr(target_model, "decoder_type", None) != "nemotron_h":
        raise ValueError(
            "The target model must use decoder_type='nemotron_h'; "
            f"got {getattr(target_model, 'decoder_type', None)!r}."
        )

    required_attributes = (
        "num_semantic_codebooks",
        "frame_stacking_factor",
        "num_all_tokens_per_codebook",
    )
    missing_attributes = [name for name in required_attributes if not hasattr(target_model, name)]
    if missing_attributes:
        raise ValueError(f"Target model is missing semantic-head metadata: {missing_attributes}")

    target_state = target_model.state_dict()
    backbone_keys = sorted(key for key in target_state if key.startswith("decoder."))
    if not backbone_keys:
        raise ValueError("Target model state contains no decoder.* tensors.")

    copy_plan: list[tuple[str, torch.Tensor, str, torch.Tensor]] = []
    for target_key in backbone_keys:
        source_key, source_tensor = _get_source_tensor(source_state, target_key)
        target_tensor = target_state[target_key]
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Backbone shape mismatch for {target_key!r}: "
                f"source {tuple(source_tensor.shape)} vs target {tuple(target_tensor.shape)}."
            )
        copy_plan.append((target_key, target_tensor, source_key, source_tensor))

    projection_weight_key = "final_proj.weight"
    projection_bias_key = "final_proj.bias"
    if projection_weight_key not in target_state or projection_bias_key not in target_state:
        raise ValueError("Target model must have final_proj.weight and final_proj.bias tensors.")

    source_weight_key, source_weight = _get_source_tensor(source_state, projection_weight_key)
    source_bias_key, source_bias = _get_source_tensor(source_state, projection_bias_key)
    target_weight = target_state[projection_weight_key]
    target_bias = target_state[projection_bias_key]

    semantic_channels = int(target_model.num_semantic_codebooks) * int(target_model.frame_stacking_factor)
    tokens_per_codebook = int(target_model.num_all_tokens_per_codebook)
    semantic_rows = semantic_channels * tokens_per_codebook

    if semantic_rows <= 0:
        raise ValueError(f"Computed a non-positive semantic projection size: {semantic_rows} rows.")
    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise ValueError("final_proj.weight must be a rank-2 tensor in both source and target models.")
    if source_bias.ndim != 1 or target_bias.ndim != 1:
        raise ValueError("final_proj.bias must be a rank-1 tensor in both source and target models.")
    if source_weight.shape[1:] != target_weight.shape[1:]:
        raise ValueError(
            "Semantic projection input shape mismatch: "
            f"source {tuple(source_weight.shape)} vs target {tuple(target_weight.shape)}."
        )
    if source_weight.shape[0] != source_bias.shape[0]:
        raise ValueError(
            "Source final projection weight/bias row mismatch: " f"{source_weight.shape[0]} vs {source_bias.shape[0]}."
        )
    if target_weight.shape[0] != target_bias.shape[0]:
        raise ValueError(
            "Target final projection weight/bias row mismatch: " f"{target_weight.shape[0]} vs {target_bias.shape[0]}."
        )
    if source_weight.shape[0] < semantic_rows or target_weight.shape[0] < semantic_rows:
        raise ValueError(
            f"Semantic projection needs {semantic_rows} rows, but source/target have "
            f"{source_weight.shape[0]}/{target_weight.shape[0]}."
        )
    if source_weight.shape[0] % tokens_per_codebook != 0:
        raise ValueError(
            f"Source final projection has {source_weight.shape[0]} rows, which is not divisible by the target "
            f"codebook vocabulary size ({tokens_per_codebook})."
        )

    with torch.no_grad():
        for _, target_tensor, _, source_tensor in copy_plan:
            target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
        target_weight[:semantic_rows].copy_(
            source_weight[:semantic_rows].to(device=target_weight.device, dtype=target_weight.dtype)
        )
        target_bias[:semantic_rows].copy_(
            source_bias[:semantic_rows].to(device=target_bias.device, dtype=target_bias.dtype)
        )

    copied_backbone_numel = sum(target_tensor.numel() for _, target_tensor, _, _ in copy_plan)
    untouched_keys = sorted(set(target_state) - set(backbone_keys) - {projection_weight_key, projection_bias_key})
    return {
        "backbone_tensor_count": len(copy_plan),
        "backbone_parameter_count": copied_backbone_numel,
        "semantic_channels": semantic_channels,
        "tokens_per_codebook": tokens_per_codebook,
        "semantic_projection_rows_copied": semantic_rows,
        "semantic_projection_source_keys": [source_weight_key, source_bias_key],
        "acoustic_projection_rows_left_random": target_weight.shape[0] - semantic_rows,
        "other_state_tensors_left_random": len(untouched_keys),
        "other_state_tensor_keys": untouched_keys,
    }


def _resolve_checkpoint_file(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {path}")
    if path.is_file():
        return path

    preferred_names = (DEFAULT_OUTPUT_NAME, "model.nemo", "model_weights.ckpt")
    for name in preferred_names:
        candidate = path / name
        if candidate.is_file():
            return candidate

    candidates = sorted(candidate for suffix in ("*.nemo", "*.ckpt") for candidate in path.glob(suffix))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .nemo or .ckpt checkpoint found in directory: {path}")
    raise ValueError(f"Checkpoint directory is ambiguous; specify one file explicitly: {candidates}")


def load_checkpoint_state(checkpoint_path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    """Load a NeMo archive, Lightning checkpoint, or extracted checkpoint directory."""

    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    if checkpoint_file.suffix == ".nemo":
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

        with tempfile.TemporaryDirectory(prefix="easy_magpie_seed_source_") as extract_dir:
            state = SaveRestoreConnector().extract_state_dict_from(
                restore_path=str(checkpoint_file),
                save_dir=extract_dir,
            )
    else:
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
            state = payload["state_dict"]
        else:
            state = payload

    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_file} did not contain a state-dict mapping.")
    return checkpoint_file, state


def _compose_model_config(config_name: str, overrides: Sequence[str]):
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "tts" / "conf" / "magpietts"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the EasyMagpie one-shot-flow config, copy only a source Nemotron-H backbone and semantic "
            "projection, and save a reusable .nemo seed. Unrecognized arguments are treated as Hydra overrides."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, help="Source .nemo/.ckpt file or checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory in which to create the seed and report.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Seed filename; must end in .nemo.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME, help="Target EasyMagpie Hydra config name.")
    parser.add_argument("--random-seed", type=int, default=1234, help="Seed used for all freshly initialized weights.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output seed/report.")
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    args, overrides = _parse_args()
    if not args.output_name.endswith(".nemo") or Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a plain filename ending in .nemo.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = output_dir / args.output_name
    report_path = output_dir / "transfer_report.json"
    if not args.overwrite:
        existing = [path for path in (output_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing seed artifacts: {existing}")

    import lightning.pytorch as pl

    from nemo.collections.tts.models import EasyMagpieTTSModel

    pl.seed_everything(args.random_seed, workers=True)
    cfg = _compose_model_config(args.config_name, overrides)
    model = EasyMagpieTTSModel(cfg=cfg.model, trainer=None)
    if model.local_transformer_type.value != "normalizing_flow":
        raise ValueError(
            "Target config must use model.local_transformer_type=normalizing_flow; "
            f"got {model.local_transformer_type.value!r}."
        )

    source_checkpoint, source_state = load_checkpoint_state(args.source_checkpoint)
    transfer_report = transfer_backbone_and_semantic_projection(model, source_state)
    del source_state

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output_dir / f".{args.output_name}.{uuid.uuid4().hex}.tmp"
    try:
        model.save_to(str(temporary_output))
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    report = {
        "source_checkpoint": str(source_checkpoint),
        "output_seed": str(output_path),
        "config_name": args.config_name,
        "hydra_overrides": overrides,
        "random_seed": args.random_seed,
        **transfer_report,
    }
    temporary_report = output_dir / f".{report_path.name}.{uuid.uuid4().hex}.tmp"
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nUse this seed for training:\n  init_from_nemo_model={output_path}")


if __name__ == "__main__":
    main()
