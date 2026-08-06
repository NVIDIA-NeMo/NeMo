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

The target model is initialized from the requested Hydra config. Every
shape-compatible target tensor is copied from the source checkpoint. New
one-shot-flow tensors without a source counterpart retain their fresh random
initialization.
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


def _find_source_tensor(
    source_state: Mapping[str, Any], target_key: str
) -> tuple[str, torch.Tensor] | None:
    matches = [key for key in _source_key_candidates(target_key) if key in source_state]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Source checkpoint has ambiguous matches for {target_key!r}: {matches}"
        )

    source_key = matches[0]
    tensor = source_state[source_key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Source checkpoint entry {source_key!r} is {type(tensor).__name__}, not a tensor."
        )
    return source_key, tensor


def _get_source_tensor(
    source_state: Mapping[str, Any], target_key: str
) -> tuple[str, torch.Tensor]:
    match = _find_source_tensor(source_state, target_key)
    if match is None:
        raise ValueError(
            f"Source checkpoint is missing target tensor {target_key!r}. "
            f"Tried keys: {_source_key_candidates(target_key)}"
        )
    return match


def transfer_compatible_pretrained_state(
    target_model: nn.Module,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy every shape-compatible source tensor into ``target_model``.

    The target one-shot flow has no matching tensors in an autoregressive
    source checkpoint, so those tensors remain freshly initialized. Backbone
    tensors are mandatory. ``final_proj`` is copied in full when its shape
    matches; otherwise only the semantic rows are copied after validating the
    packed-codebook layout.
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
    missing_attributes = [
        name for name in required_attributes if not hasattr(target_model, name)
    ]
    if missing_attributes:
        raise ValueError(
            f"Target model is missing semantic-head metadata: {missing_attributes}"
        )

    target_state = target_model.state_dict()
    backbone_keys = sorted(key for key in target_state if key.startswith("decoder."))
    if not backbone_keys:
        raise ValueError("Target model state contains no decoder.* tensors.")

    projection_weight_key = "final_proj.weight"
    projection_bias_key = "final_proj.bias"
    projection_keys = {projection_weight_key, projection_bias_key}
    if not projection_keys.issubset(target_state):
        raise ValueError(
            "Target model must have final_proj.weight and final_proj.bias tensors."
        )

    copy_plan: list[tuple[str, torch.Tensor, str, torch.Tensor]] = []
    missing_source_keys: list[str] = []
    shape_mismatches: dict[str, dict[str, Any]] = {}
    used_source_keys: set[str] = set()
    for target_key, target_tensor in target_state.items():
        if target_key in projection_keys:
            continue
        match = _find_source_tensor(source_state, target_key)
        if match is None:
            missing_source_keys.append(target_key)
            continue
        source_key, source_tensor = match
        if source_tensor.shape != target_tensor.shape:
            if target_key.startswith("decoder."):
                raise ValueError(
                    f"Backbone shape mismatch for {target_key!r}: "
                    f"source {tuple(source_tensor.shape)} vs target {tuple(target_tensor.shape)}."
                )
            shape_mismatches[target_key] = {
                "source_key": source_key,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
            continue
        copy_plan.append((target_key, target_tensor, source_key, source_tensor))
        used_source_keys.add(source_key)

    copied_target_keys = {target_key for target_key, _, _, _ in copy_plan}
    missing_backbone_keys = sorted(set(backbone_keys) - copied_target_keys)
    if missing_backbone_keys:
        raise ValueError(
            f"Source checkpoint is missing compatible backbone tensors: {missing_backbone_keys}"
        )

    source_weight_key, source_weight = _get_source_tensor(
        source_state, projection_weight_key
    )
    source_bias_key, source_bias = _get_source_tensor(source_state, projection_bias_key)
    target_weight = target_state[projection_weight_key]
    target_bias = target_state[projection_bias_key]

    semantic_channels = int(target_model.num_semantic_codebooks) * int(
        target_model.frame_stacking_factor
    )
    tokens_per_codebook = int(target_model.num_all_tokens_per_codebook)
    semantic_rows = semantic_channels * tokens_per_codebook

    if semantic_rows <= 0:
        raise ValueError(
            f"Computed a non-positive semantic projection size: {semantic_rows} rows."
        )
    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise ValueError(
            "final_proj.weight must be a rank-2 tensor in both source and target models."
        )
    if source_bias.ndim != 1 or target_bias.ndim != 1:
        raise ValueError(
            "final_proj.bias must be a rank-1 tensor in both source and target models."
        )
    if source_weight.shape[1:] != target_weight.shape[1:]:
        raise ValueError(
            "Semantic projection input shape mismatch: "
            f"source {tuple(source_weight.shape)} vs target {tuple(target_weight.shape)}."
        )
    if source_weight.shape[0] != source_bias.shape[0]:
        raise ValueError(
            "Source final projection weight/bias row mismatch: "
            f"{source_weight.shape[0]} vs {source_bias.shape[0]}."
        )
    if target_weight.shape[0] != target_bias.shape[0]:
        raise ValueError(
            "Target final projection weight/bias row mismatch: "
            f"{target_weight.shape[0]} vs {target_bias.shape[0]}."
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

    projection_shapes_match = (
        source_weight.shape == target_weight.shape
        and source_bias.shape == target_bias.shape
    )
    projection_rows_copied = (
        target_weight.shape[0] if projection_shapes_match else semantic_rows
    )

    with torch.no_grad():
        for _, target_tensor, _, source_tensor in copy_plan:
            target_tensor.copy_(
                source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            )
        target_weight[:projection_rows_copied].copy_(
            source_weight[:projection_rows_copied].to(
                device=target_weight.device, dtype=target_weight.dtype
            )
        )
        target_bias[:projection_rows_copied].copy_(
            source_bias[:projection_rows_copied].to(
                device=target_bias.device, dtype=target_bias.dtype
            )
        )

    used_source_keys.update((source_weight_key, source_bias_key))
    copied_target_keys.update(projection_keys)
    copied_tensor_keys = sorted(copied_target_keys)
    copied_backbone_numel = sum(
        target_tensor.numel()
        for target_key, target_tensor, _, _ in copy_plan
        if target_key.startswith("decoder.")
    )
    copied_non_backbone_numel = sum(
        target_tensor.numel()
        for target_key, target_tensor, _, _ in copy_plan
        if not target_key.startswith("decoder.")
    ) + projection_rows_copied * (target_weight.shape[1] + 1)
    target_keys_left_random = sorted(set(missing_source_keys) | set(shape_mismatches))
    unused_source_keys = sorted(set(source_state) - used_source_keys)

    return {
        "backbone_tensor_count": len(backbone_keys),
        "backbone_parameter_count": copied_backbone_numel,
        "compatible_tensor_count": len(copied_tensor_keys),
        "compatible_parameter_count": copied_backbone_numel + copied_non_backbone_numel,
        "non_backbone_tensor_count": len(copied_tensor_keys) - len(backbone_keys),
        "non_backbone_parameter_count": copied_non_backbone_numel,
        "copied_state_tensor_keys": copied_tensor_keys,
        "semantic_channels": semantic_channels,
        "tokens_per_codebook": tokens_per_codebook,
        "projection_rows_copied": projection_rows_copied,
        "semantic_projection_rows_copied": semantic_rows,
        "acoustic_projection_rows_copied": max(
            0, projection_rows_copied - semantic_rows
        ),
        "acoustic_projection_rows_left_random": target_weight.shape[0]
        - projection_rows_copied,
        "projection_source_keys": [source_weight_key, source_bias_key],
        "target_state_tensors_left_random": len(target_keys_left_random),
        "target_state_tensor_keys_left_random": target_keys_left_random,
        "shape_mismatched_target_tensors": shape_mismatches,
        "source_state_tensors_unused": len(unused_source_keys),
        "source_state_tensor_keys_unused": unused_source_keys,
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

    candidates = sorted(
        candidate for suffix in ("*.nemo", "*.ckpt") for candidate in path.glob(suffix)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No .nemo or .ckpt checkpoint found in directory: {path}"
        )
    raise ValueError(
        f"Checkpoint directory is ambiguous; specify one file explicitly: {candidates}"
    )


def load_checkpoint_state(
    checkpoint_path: str | Path,
) -> tuple[Path, Mapping[str, Any]]:
    """Load a NeMo archive, Lightning checkpoint, or extracted checkpoint directory."""

    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    if checkpoint_file.suffix == ".nemo":
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

        with tempfile.TemporaryDirectory(
            prefix="easy_magpie_seed_source_"
        ) as extract_dir:
            state = SaveRestoreConnector().extract_state_dict_from(
                restore_path=str(checkpoint_file),
                save_dir=extract_dir,
            )
    else:
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and isinstance(
            payload.get("state_dict"), Mapping
        ):
            state = payload["state_dict"]
        else:
            state = payload

    if not isinstance(state, Mapping):
        raise TypeError(
            f"Checkpoint {checkpoint_file} did not contain a state-dict mapping."
        )
    return checkpoint_file, state


def _compose_model_config(config_name: str, overrides: Sequence[str]):
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "tts" / "conf" / "magpietts"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _defer_dataset_setup(model_cfg: Any) -> None:
    """Prevent ModelPT from constructing dataloaders for a weights-only seed."""

    from omegaconf import open_dict

    for dataset_name in ("train_ds", "validation_ds", "test_ds"):
        dataset_cfg = model_cfg.get(dataset_name)
        if dataset_cfg is not None:
            with open_dict(dataset_cfg):
                dataset_cfg.defer_setup = True


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the EasyMagpie one-shot-flow config, copy every shape-compatible pretrained tensor, and "
            "save a reusable .nemo seed. Unrecognized arguments are treated as Hydra overrides."
        )
    )
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help="Source .nemo/.ckpt file or checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to create the seed and report.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Seed filename; must end in .nemo.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Target EasyMagpie Hydra config name.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Seed used for all freshly initialized weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output seed/report.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    args, overrides = _parse_args()
    if (
        not args.output_name.endswith(".nemo")
        or Path(args.output_name).name != args.output_name
    ):
        raise ValueError("--output-name must be a plain filename ending in .nemo.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_path = output_dir / args.output_name
    report_path = output_dir / "transfer_report.json"
    if not args.overwrite:
        existing = [path for path in (output_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing seed artifacts: {existing}"
            )

    import lightning.pytorch as pl

    from nemo.collections.tts.models import EasyMagpieTTSModel

    pl.seed_everything(args.random_seed, workers=True)
    cfg = _compose_model_config(args.config_name, overrides)
    _defer_dataset_setup(cfg.model)
    model = EasyMagpieTTSModel(cfg=cfg.model, trainer=None)
    if model.local_transformer_type.value != "normalizing_flow":
        raise ValueError(
            "Target config must use model.local_transformer_type=normalizing_flow; "
            f"got {model.local_transformer_type.value!r}."
        )

    source_checkpoint, source_state = load_checkpoint_state(args.source_checkpoint)
    transfer_report = transfer_compatible_pretrained_state(model, source_state)
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
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_report, report_path)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nUse this seed for training:\n  init_from_nemo_model={output_path}")


if __name__ == "__main__":
    main()
