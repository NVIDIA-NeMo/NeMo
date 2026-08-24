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
one-shot-flow tensors without a source counterpart retain their constructor
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


def _find_source_tensor(source_state: Mapping[str, Any], target_key: str) -> tuple[str, torch.Tensor] | None:
    matches = [key for key in _source_key_candidates(target_key) if key in source_state]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Source checkpoint has ambiguous matches for {target_key!r}: {matches}")

    source_key = matches[0]
    tensor = source_state[source_key]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Source checkpoint entry {source_key!r} is {type(tensor).__name__}, not a tensor.")
    return source_key, tensor


def _get_source_tensor(source_state: Mapping[str, Any], target_key: str) -> tuple[str, torch.Tensor]:
    match = _find_source_tensor(source_state, target_key)
    if match is None:
        raise ValueError(
            f"Source checkpoint is missing target tensor {target_key!r}. "
            f"Tried keys: {_source_key_candidates(target_key)}"
        )
    return match


def _projection_conditioned_target_keys(target_model: nn.Module) -> set[str]:
    """Validate and return target tensors materialized from 78-D source projections."""

    if int(target_model.num_semantic_codebooks) != 1 or int(target_model.frame_stacking_factor) != 1:
        raise ValueError(
            "Projection-conditioned conversion requires one semantic codebook and frame_stacking_factor=1."
        )
    if not isinstance(getattr(target_model, "audio_in_projection", None), nn.Identity):
        raise ValueError("Projection-conditioned conversion requires audio_in_projection to be Identity.")
    if not bool(getattr(target_model, "oneshot_separate_context_input_projection", False)):
        raise ValueError(
            "Projection-conditioned conversion requires model.oneshot_separate_context_input_projection=true."
        )
    if len(target_model.audio_embeddings) != 1 or len(target_model.flow_context_audio_embeddings) != 1:
        raise ValueError("Projection-conditioned conversion requires one decoder and one context semantic table.")

    return {
        "audio_embeddings.0.weight",
        "flow_acoustic_in_projection.weight",
        "flow_acoustic_in_projection.bias",
        "flow_context_audio_embeddings.0.weight",
        "flow_context_acoustic_in_projection.weight",
        "flow_context_acoustic_in_projection.bias",
    }


def _materialize_projection_conditioned_inputs(
    target_model: nn.Module,
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str], int]:
    """Split the source codec projection into semantic tables and acoustic linears."""

    _projection_conditioned_target_keys(target_model)
    codebook_size = int(target_model.codebook_size)
    semantic_dim = int(target_model.semantic_codec_embedding_dim)
    acoustic_dim = int(target_model.acoustic_codec_embedding_dim)
    target_embedding_dim = int(target_model.cfg.embedding_dim)

    projection_specs = {
        "decoder": ("decoder_code_proj", "audio_embeddings.0.weight", "flow_acoustic_in_projection"),
        "context": (
            "context_code_proj",
            "flow_context_audio_embeddings.0.weight",
            "flow_context_acoustic_in_projection",
        ),
    }
    resolved: dict[str, dict[str, Any]] = {}
    used_source_keys: set[str] = set()
    for path_name, (source_prefix, table_key, acoustic_prefix) in projection_specs.items():
        source_weight_key, source_weight = _get_source_tensor(source_state, f"{source_prefix}.weight")
        source_bias_key, source_bias = _get_source_tensor(source_state, f"{source_prefix}.bias")
        if source_weight.shape != (target_embedding_dim, semantic_dim + acoustic_dim):
            raise ValueError(
                f"{path_name} source projection must have shape "
                f"{(target_embedding_dim, semantic_dim + acoustic_dim)}, got {tuple(source_weight.shape)}."
            )
        if source_bias.shape != (target_embedding_dim,):
            raise ValueError(
                f"{path_name} source projection bias must have shape {(target_embedding_dim,)}, "
                f"got {tuple(source_bias.shape)}."
            )
        acoustic_weight_key = f"{acoustic_prefix}.weight"
        acoustic_bias_key = f"{acoustic_prefix}.bias"
        if target_state[table_key].shape != (int(target_model.num_all_tokens_per_codebook), target_embedding_dim):
            raise ValueError(f"Unexpected target semantic table shape: {tuple(target_state[table_key].shape)}.")
        if target_state[acoustic_weight_key].shape != (target_embedding_dim, acoustic_dim):
            raise ValueError(
                f"Unexpected target acoustic projection shape: {tuple(target_state[acoustic_weight_key].shape)}."
            )
        resolved[path_name] = {
            "source_weight_key": source_weight_key,
            "source_weight": source_weight,
            "source_bias_key": source_bias_key,
            "source_bias": source_bias,
            "table_key": table_key,
            "acoustic_weight_key": acoustic_weight_key,
            "acoustic_bias_key": acoustic_bias_key,
        }
        used_source_keys.update((source_weight_key, source_bias_key))

    special_specs = {
        "decoder": (
            ("audio_bos_emb", int(target_model.audio_bos_id)),
            ("audio_eos_emb", int(target_model.audio_eos_id)),
        ),
        "context": (
            ("context_bos_emb", int(target_model.context_audio_bos_id)),
            ("context_eos_emb", int(target_model.context_audio_eos_id)),
        ),
    }
    resolved_specials: dict[str, list[tuple[str, torch.Tensor, int]]] = {}
    for path_name, specs in special_specs.items():
        resolved_specials[path_name] = []
        for source_name, target_row in specs:
            source_key, source_tensor = _get_source_tensor(source_state, source_name)
            if source_tensor.numel() != target_embedding_dim:
                raise ValueError(
                    f"Source special embedding {source_key!r} has {source_tensor.numel()} values; "
                    f"expected {target_embedding_dim}."
                )
            resolved_specials[path_name].append((source_key, source_tensor.reshape(target_embedding_dim), target_row))
            used_source_keys.add(source_key)

    device = target_state["audio_embeddings.0.weight"].device
    semantic_ids = torch.arange(codebook_size, device=device, dtype=torch.long).view(1, 1, codebook_size)
    semantic_lens = torch.tensor([codebook_size], device=device, dtype=torch.long)
    semantic_embedding = target_model._codec_helper.semantic_codes_to_embedding(semantic_ids, semantic_lens)
    expected_semantic_shape = (1, semantic_dim, codebook_size)
    if semantic_embedding.shape != expected_semantic_shape:
        raise ValueError(
            f"Target codec produced semantic embeddings shaped {tuple(semantic_embedding.shape)}; "
            f"expected {expected_semantic_shape}."
        )
    semantic_rows = semantic_embedding[0].transpose(0, 1)

    with torch.no_grad():
        for path_name, tensors in resolved.items():
            table = target_state[tensors["table_key"]]
            source_weight = tensors["source_weight"].to(device=table.device, dtype=table.dtype)
            source_bias = tensors["source_bias"].to(device=table.device, dtype=table.dtype)
            table[:codebook_size].copy_(
                semantic_rows.to(table.dtype) @ source_weight[:, :semantic_dim].T + source_bias
            )
            for _, source_special, target_row in resolved_specials[path_name]:
                table[target_row].copy_(source_special.to(device=table.device, dtype=table.dtype))
            target_state[tensors["acoustic_weight_key"]].copy_(
                source_weight[:, semantic_dim:].to(target_state[tensors["acoustic_weight_key"]].dtype)
            )
            target_state[tensors["acoustic_bias_key"]].zero_()

    initialized_parameter_count = 0
    initialized_special_count = len(next(iter(resolved_specials.values())))
    for tensors in resolved.values():
        initialized_parameter_count += (codebook_size + initialized_special_count) * target_embedding_dim
        initialized_parameter_count += target_state[tensors["acoustic_weight_key"]].numel()
        initialized_parameter_count += target_state[tensors["acoustic_bias_key"]].numel()

    all_special_rows = set(range(codebook_size, int(target_model.num_all_tokens_per_codebook)))
    initialized_special_rows = {
        path_name: sorted(target_row for _, _, target_row in specials)
        for path_name, specials in resolved_specials.items()
    }
    return (
        {
            "projection_conditioned_source": True,
            "input_projection_semantic_dim": semantic_dim,
            "input_projection_acoustic_dim": acoustic_dim,
            "input_projection_normal_rows_initialized": codebook_size,
            "input_projection_special_rows_initialized": initialized_special_rows,
            "input_projection_special_rows_left_random": {
                path_name: sorted(all_special_rows - set(rows)) for path_name, rows in initialized_special_rows.items()
            },
            "input_projection_source_keys": sorted(used_source_keys),
        },
        used_source_keys,
        initialized_parameter_count,
    )


def transfer_compatible_pretrained_state(
    target_model: nn.Module,
    source_state: Mapping[str, Any],
    *,
    projection_conditioned_source: bool = False,
) -> dict[str, Any]:
    """Copy every shape-compatible source tensor into ``target_model``.

    The target one-shot flow has no matching tensors in an autoregressive
    source checkpoint, so those tensors normally remain freshly initialized.
    With ``projection_conditioned_source=True``, learned source decoder/context
    codec projections are split into hybrid semantic lookups and continuous
    acoustic linears. Backbone tensors are mandatory. ``final_proj`` is copied
    in full when its shape matches; otherwise only the semantic rows are copied
    after validating the packed-codebook layout.
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

    projection_weight_key = "final_proj.weight"
    projection_bias_key = "final_proj.bias"
    projection_keys = {projection_weight_key, projection_bias_key}
    if not projection_keys.issubset(target_state):
        raise ValueError("Target model must have final_proj.weight and final_proj.bias tensors.")

    materialized_target_keys = (
        _projection_conditioned_target_keys(target_model) if projection_conditioned_source else set()
    )
    copy_plan: list[tuple[str, torch.Tensor, str, torch.Tensor]] = []
    missing_source_keys: list[str] = []
    shape_mismatches: dict[str, dict[str, Any]] = {}
    used_source_keys: set[str] = set()
    for target_key, target_tensor in target_state.items():
        if target_key in projection_keys or target_key in materialized_target_keys:
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
        raise ValueError(f"Source checkpoint is missing compatible backbone tensors: {missing_backbone_keys}")

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

    projection_shapes_match = source_weight.shape == target_weight.shape and source_bias.shape == target_bias.shape
    projection_rows_copied = target_weight.shape[0] if projection_shapes_match else semantic_rows

    input_projection_report: dict[str, Any] = {"projection_conditioned_source": False}
    materialized_source_keys: set[str] = set()
    materialized_parameter_count = 0
    if projection_conditioned_source:
        input_projection_report, materialized_source_keys, materialized_parameter_count = (
            _materialize_projection_conditioned_inputs(target_model, target_state, source_state)
        )

    with torch.no_grad():
        for _, target_tensor, _, source_tensor in copy_plan:
            target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
        target_weight[:projection_rows_copied].copy_(
            source_weight[:projection_rows_copied].to(device=target_weight.device, dtype=target_weight.dtype)
        )
        target_bias[:projection_rows_copied].copy_(
            source_bias[:projection_rows_copied].to(device=target_bias.device, dtype=target_bias.dtype)
        )

    used_source_keys.update((source_weight_key, source_bias_key))
    used_source_keys.update(materialized_source_keys)
    copied_target_keys.update(projection_keys)
    copied_target_keys.update(materialized_target_keys)
    copied_tensor_keys = sorted(copied_target_keys)
    copied_backbone_numel = sum(
        target_tensor.numel() for target_key, target_tensor, _, _ in copy_plan if target_key.startswith("decoder.")
    )
    copied_non_backbone_numel = (
        sum(
            target_tensor.numel()
            for target_key, target_tensor, _, _ in copy_plan
            if not target_key.startswith("decoder.")
        )
        + projection_rows_copied * (target_weight.shape[1] + 1)
        + materialized_parameter_count
    )
    target_keys_left_initialized = sorted(set(missing_source_keys) | set(shape_mismatches))
    target_keys_left_zero_initialized = sorted(
        key
        for key in target_keys_left_initialized
        if key.startswith(
            (
                "flow_acoustic_in_projection.",
                "flow_context_acoustic_in_projection.",
            )
        )
    )
    nonzero_zero_initialized_keys = [
        key for key in target_keys_left_zero_initialized if torch.count_nonzero(target_state[key]).item() != 0
    ]
    if nonzero_zero_initialized_keys:
        raise RuntimeError(
            "Expected unmatched continuous acoustic input projection tensors to remain zero-initialized, "
            f"but found nonzero values in: {nonzero_zero_initialized_keys}."
        )
    target_keys_left_random = sorted(set(target_keys_left_initialized) - set(target_keys_left_zero_initialized))
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
        "acoustic_projection_rows_copied": max(0, projection_rows_copied - semantic_rows),
        "acoustic_projection_rows_left_random": target_weight.shape[0] - projection_rows_copied,
        "projection_source_keys": [source_weight_key, source_bias_key],
        "target_state_tensors_left_initialized": len(target_keys_left_initialized),
        "target_state_tensor_keys_left_initialized": target_keys_left_initialized,
        "target_state_tensors_left_zero_initialized": len(target_keys_left_zero_initialized),
        "target_state_tensor_keys_left_zero_initialized": target_keys_left_zero_initialized,
        "target_state_tensors_left_random": len(target_keys_left_random),
        "target_state_tensor_keys_left_random": target_keys_left_random,
        "shape_mismatched_target_tensors": shape_mismatches,
        "source_state_tensors_unused": len(unused_source_keys),
        "source_state_tensor_keys_unused": unused_source_keys,
        **input_projection_report,
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


def load_checkpoint_state(
    checkpoint_path: str | Path,
) -> tuple[Path, Mapping[str, Any]]:
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
        "--projection-conditioned-source",
        action="store_true",
        help=(
            "Materialize decoder/context semantic tables and acoustic input projections from "
            "source decoder_code_proj/context_code_proj tensors."
        ),
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
    _defer_dataset_setup(cfg.model)
    model = EasyMagpieTTSModel(cfg=cfg.model, trainer=None)
    if not model.local_transformer_type.is_oneshot:
        raise ValueError(
            "Target config must use a one-shot continuous acoustic predictor; "
            f"got {model.local_transformer_type.value!r}."
        )

    source_checkpoint, source_state = load_checkpoint_state(args.source_checkpoint)
    transfer_report = transfer_compatible_pretrained_state(
        model,
        source_state,
        projection_conditioned_source=args.projection_conditioned_source,
    )
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
