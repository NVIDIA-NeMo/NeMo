# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Export a complete DPO model DCP as an indexed HuggingFace safetensors model.

This is the serving bridge for the finite SpeechLM2 DPO recipe.  It is a
normal, offline conversion in the owning package: every served tensor is read
from the candidate model DCP, and the immutable Hero2 serving directory is
used only for non-weight HuggingFace assets and a namespace/type contract.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load

from nemo.collections.speechlm2.dpo.surface import selected_parameter_names


DEFAULT_SHARD_BYTES = 4 * 1024**3
ASSET_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json")
SAFE_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    payload_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count * torch.empty((), dtype=dtype).element_size()


def dcp_state_specs(checkpoint: Path) -> tuple[dict[str, TensorSpec], list[str]]:
    """Read only DCP metadata; model values remain on Lustre until export."""

    metadata_path = checkpoint / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    specs: dict[str, TensorSpec] = {}
    extra_state: list[str] = []
    for full_name, item in metadata.items():
        if not full_name.startswith("state_dict."):
            raise RuntimeError(f"unexpected non-model DCP key: {full_name}")
        name = full_name.removeprefix("state_dict.")
        if name.endswith("._extra_state"):
            extra_state.append(name)
            continue
        if not hasattr(item, "size") or not hasattr(item, "properties"):
            raise RuntimeError(f"non-tensor model DCP key: {full_name}")
        shape = tuple(int(value) for value in item.size)
        dtype = item.properties.dtype
        specs[name] = TensorSpec(shape, dtype, _tensor_bytes(shape, dtype))
    if not specs:
        raise RuntimeError("DCP contains no model tensors")
    return dict(sorted(specs.items())), sorted(extra_state)


def read_surface_contract(trajectory: Path) -> tuple[str, ...]:
    """Require the exact selected FP32 surface recorded by the training run."""

    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    surface = payload.get("surface")
    names = surface.get("names") if isinstance(surface, dict) else None
    expected = selected_parameter_names()
    if (
        not isinstance(names, list)
        or tuple(names) != expected
        or surface.get("tensor_count") != len(expected)
        or surface.get("scalar_count") is None
        or surface.get("dtypes") != ["torch.float32"]
        or payload.get("lora") is not False
    ):
        raise RuntimeError("trajectory does not declare the exact Hero2 DPO FP32 surface")
    return expected


def _baseline_specs(baseline: Path) -> tuple[dict[str, TensorSpec], dict[str, Path]]:
    """Read a standard single-file or indexed safetensors serving baseline."""

    single = baseline / "model.safetensors"
    index = baseline / "model.safetensors.index.json"
    locations: dict[str, Path] = {}
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as model:
            locations = {name: single for name in model.keys()}
    elif index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weights = payload.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise RuntimeError(f"invalid safetensors index: {index}")
        for name, filename in weights.items():
            candidate = baseline / str(filename)
            if candidate.parent != baseline or not candidate.is_file():
                raise FileNotFoundError(candidate)
            locations[str(name)] = candidate
    else:
        raise FileNotFoundError(single)

    specs: dict[str, TensorSpec] = {}
    by_file: dict[Path, list[str]] = {}
    for name, path in locations.items():
        by_file.setdefault(path, []).append(name)
    for path, names in by_file.items():
        with safe_open(path, framework="pt", device="cpu") as model:
            for name in names:
                tensor = model.get_slice(name)
                dtype = SAFE_TO_TORCH.get(tensor.get_dtype())
                if dtype is None:
                    raise RuntimeError(f"unsupported safetensors dtype for {name}")
                shape = tuple(int(value) for value in tensor.get_shape())
                specs[name] = TensorSpec(shape, dtype, _tensor_bytes(shape, dtype))
    return dict(sorted(specs.items())), locations


def check_serving_contract(
    *,
    candidate: dict[str, TensorSpec],
    baseline: dict[str, TensorSpec],
    selected_fp32: tuple[str, ...],
) -> dict[str, Any]:
    """Reject namespace, shape, and precision drift before serving conversion."""

    candidate_names = set(candidate)
    baseline_names = set(baseline)
    if candidate_names != baseline_names:
        raise RuntimeError(
            "candidate/baseline tensor namespace differs: "
            f"missing={sorted(baseline_names - candidate_names)[:8]} "
            f"unexpected={sorted(candidate_names - baseline_names)[:8]}"
        )
    selected = set(selected_fp32)
    if not selected <= candidate_names:
        raise RuntimeError("selected training surface is absent from model DCP")
    promoted: set[str] = set()
    for name in sorted(candidate_names):
        source = candidate[name]
        template = baseline[name]
        if source.shape != template.shape:
            raise RuntimeError(f"candidate/baseline shape differs: {name}")
        if source.dtype == template.dtype:
            continue
        if name in selected and template.dtype is torch.bfloat16 and source.dtype is torch.float32:
            promoted.add(name)
            continue
        raise RuntimeError(
            f"candidate/baseline dtype differs outside declared BF16->FP32 surface: {name} "
            f"({template.dtype} -> {source.dtype})"
        )
    if promoted != selected:
        raise RuntimeError(
            "candidate FP32 surface differs from trajectory: "
            f"missing={sorted(selected - promoted)[:8]} unexpected={sorted(promoted - selected)[:8]}"
        )
    return {
        "serving_tensor_count": len(candidate),
        "extra_state_excluded": None,
        "fp32_surface_tensor_count": len(promoted),
        "fp32_surface_names_sha256": hashlib.sha256("\n".join(sorted(promoted)).encode()).hexdigest(),
    }


def shard_plan(specs: dict[str, TensorSpec], shard_bytes: int) -> list[list[str]]:
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    plan: list[list[str]] = []
    active: list[str] = []
    active_bytes = 0
    for name, spec in specs.items():
        if spec.payload_bytes > shard_bytes:
            raise RuntimeError(f"tensor exceeds shard capacity: {name}")
        if active and active_bytes + spec.payload_bytes > shard_bytes:
            plan.append(active)
            active, active_bytes = [], 0
        active.append(name)
        active_bytes += spec.payload_bytes
    if active:
        plan.append(active)
    return plan


def _copy_assets(baseline: Path, output: Path) -> dict[str, str]:
    config_source = baseline / "config.json"
    if not config_source.is_file():
        raise FileNotFoundError(config_source)
    config = json.loads(config_source.read_text(encoding="utf-8"))
    config.pop("init_from_checkpoint", None)
    config_output = output / "config.json"
    config_output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assets = {"config.json": sha256_file(config_output)}
    for filename in ASSET_FILES:
        source = baseline / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / filename
        shutil.copyfile(source, destination)
        assets[filename] = sha256_file(destination)
    return assets


def _write_shards(
    *, checkpoint: Path, output: Path, specs: dict[str, TensorSpec], plan: list[list[str]]
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    weight_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for index, names in enumerate(plan, start=1):
        filename = f"model-{index:05d}-of-{len(plan):05d}.safetensors"
        temporary = output / f".{filename}.tmp"
        destination = output / filename
        tensors = {
            name: torch.empty(specs[name].shape, dtype=specs[name].dtype, device="cpu")
            for name in names
        }
        load({"state_dict": tensors}, checkpoint_id=str(checkpoint), no_dist=True)
        save_file(dict(sorted(tensors.items())), temporary)
        os.replace(temporary, destination)
        payload_bytes = sum(specs[name].payload_bytes for name in names)
        with destination.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
        if destination.stat().st_size != 8 + header_bytes + payload_bytes:
            raise RuntimeError(f"invalid safetensors payload size: {filename}")
        with safe_open(destination, framework="pt", device="cpu") as served:
            if set(served.keys()) != set(names):
                raise RuntimeError(f"served shard key mismatch: {filename}")
            for name in names:
                value = served.get_tensor(name)
                if value.shape != tensors[name].shape or value.dtype != tensors[name].dtype:
                    raise RuntimeError(f"served tensor metadata mismatch: {name}")
                if not torch.equal(value, tensors[name]):
                    raise RuntimeError(f"served tensor value mismatch: {name}")
        weight_map.update({name: filename for name in names})
        records.append(
            {
                "name": filename,
                "tensor_count": len(names),
                "payload_bytes": payload_bytes,
                "sha256": sha256_file(destination),
            }
        )
        del tensors
        gc.collect()
    if set(weight_map) != set(specs):
        raise RuntimeError("export omitted or duplicated candidate tensors")
    return dict(sorted(weight_map.items())), records


def export_dcp_to_hf(
    *, candidate_dcp: Path, serving_baseline: Path, trajectory: Path, output: Path, shard_bytes: int = DEFAULT_SHARD_BYTES
) -> dict[str, Any]:
    """Convert a durable DPO DCP to a fresh, fully candidate-weighted HF directory."""

    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    candidate, extra_state = dcp_state_specs(candidate_dcp)
    selected = read_surface_contract(trajectory)
    baseline, _ = _baseline_specs(serving_baseline)
    contract = check_serving_contract(candidate=candidate, baseline=baseline, selected_fp32=selected)
    contract["extra_state_excluded"] = len(extra_state)
    plan = shard_plan(candidate, shard_bytes)
    output.mkdir(parents=True)
    try:
        assets = _copy_assets(serving_baseline, output)
        weights, shards = _write_shards(checkpoint=candidate_dcp, output=output, specs=candidate, plan=plan)
        (output / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": sum(spec.payload_bytes for spec in candidate.values())}, "weight_map": weights}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema": "speechlm2.dpo.full-dcp-serving-export.v1",
            "status": "ready",
            "candidate_dcp": str(candidate_dcp),
            "candidate_metadata_sha256": sha256_file(candidate_dcp / ".metadata"),
            "training_trajectory": str(trajectory),
            "training_trajectory_sha256": sha256_file(trajectory),
            "serving_baseline": str(serving_baseline),
            "asset_sha256": assets,
            "contract": contract,
            "shard_count": len(shards),
            "shards": shards,
            "checks": {
                "every_served_weight_comes_from_candidate_dcp": True,
                "all_served_tensors_exactly_round_trip": True,
                "exact_declared_fp32_training_surface_preserved": True,
                "no_adapter_or_lora_merge": True,
            },
        }
        (output / "EVAL_MODEL_READY.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
