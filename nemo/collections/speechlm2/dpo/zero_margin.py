# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""No-update, pointwise identity audit for the native SpeechLM2 DPO cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from lightning.pytorch import Callback

from nemo.collections.speechlm2.dpo.model import DPOSALMAutomodel, _sha256_file
from nemo.collections.speechlm2.dpo.objective import dpo_pair_objective


POSITIVE_ZERO_FP32_BITS = "0x00000000"
LOG2_FP32 = struct.unpack("<f", struct.pack("<f", math.log(2.0)))[0]
LOG2_FP32_BITS = "0x3f317218"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fp32_bits(value: float) -> str:
    _require(math.isfinite(float(value)), "zero-margin ledger contains a nonfinite value")
    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular zero-margin ledger: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _all_rank_values(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _optimizer_step_count(model: DPOSALMAutomodel) -> int:
    optimizer = model.optimizers()
    raw = getattr(optimizer, "optimizer", optimizer)
    values: list[int] = []
    for state in raw.state.values():
        step = state.get("step")
        if step is None:
            continue
        if isinstance(step, torch.Tensor):
            step = int(step.detach().cpu().item())
        values.append(int(step))
    return max(values, default=0)


def _optimizer_learning_rates(model: DPOSALMAutomodel) -> list[float]:
    optimizer = model.optimizers()
    raw = getattr(optimizer, "optimizer", optimizer)
    return [float(group["lr"]) for group in raw.param_groups]


def _global_surface_receipt(model: DPOSALMAutomodel) -> dict[str, Any]:
    local = model._digest_surface()
    rank_hashes = _all_rank_values(local)
    _require(all(isinstance(value, str) and len(value) == 64 for value in rank_hashes), "invalid rank surface digest")
    return {"rank_sha256": rank_hashes, "global_sha256": _stable_digest(rank_hashes)}


def _scalar(value: torch.Tensor) -> float:
    detached = value.detach().float().cpu()
    _require(detached.numel() == 1, "DPO zero-margin audit expected a scalar")
    result = float(detached.item())
    _require(math.isfinite(result), "DPO zero-margin audit observed a nonfinite scalar")
    return result


def audit_local_pairs(
    model: Any,
    shards: list[list[Any]],
    *,
    rank: int,
    world_size: int,
    pairs_per_shard: int,
    beta: float,
) -> list[dict[str, Any]]:
    """Recompute every rank-owned pair through ``_references`` and ``_policy_pair``."""

    _require(len(shards) > 0, "zero-margin preflight received no source shards")
    _require(len(model._references) == len(shards), "actual DPO reference cache is incomplete")
    rows: list[dict[str, Any]] = []
    for source_shard, pairs in enumerate(shards, 1):
        references = model._references.get(source_shard)
        _require(references is not None and len(references) == len(pairs), "reference-cache/source-shard mismatch")
        for local_index, (pair, reference) in enumerate(zip(pairs, references, strict=True)):
            within_shard_index = rank + local_index * world_size
            expected_active = within_shard_index < pairs_per_shard
            _require(bool(pair.active) == expected_active, f"{pair.pair_id}: active padding contract drift")
            # FSDP collectives must be entered in the same order on every
            # rank.  The finite DPO data contract pads shorter rank-local
            # shards by cloning their final active pair with ``active=False``;
            # the native training loop executes those forwards with zero loss.
            # Mirror that lockstep forward schedule here, but never admit a
            # padding pair to the pointwise audit ledger or objective.
            chosen_policy, rejected_policy = model._policy_pair(pair)
            if not pair.active:
                del chosen_policy, rejected_policy
                force_reshard = getattr(model, "_force_reshard", None)
                if callable(force_reshard):
                    force_reshard()
                continue
            chosen_reference = torch.tensor(reference[0], dtype=torch.float32, device=model.device)
            rejected_reference = torch.tensor(reference[1], dtype=torch.float32, device=model.device)
            chosen_delta = chosen_policy - chosen_reference
            rejected_delta = rejected_policy - rejected_reference
            objective = dpo_pair_objective(
                chosen_policy_logp=chosen_policy,
                rejected_policy_logp=rejected_policy,
                chosen_reference_logp=chosen_reference,
                rejected_reference_logp=rejected_reference,
                beta=beta,
            )
            values = {
                "chosen_policy_logp": _scalar(chosen_policy),
                "rejected_policy_logp": _scalar(rejected_policy),
                "chosen_reference_logp": _scalar(chosen_reference),
                "rejected_reference_logp": _scalar(rejected_reference),
                "chosen_delta": _scalar(chosen_delta),
                "rejected_delta": _scalar(rejected_delta),
                "dpo_margin": _scalar(objective.margin),
                "dpo_loss": _scalar(objective.loss),
            }
            row: dict[str, Any] = {
                "pair_id": str(pair.pair_id),
                "source_id": str(pair.source_id),
                "source_shard": source_shard,
                "within_shard_index": within_shard_index,
                "rank": rank,
                "active": True,
            }
            for name, value in values.items():
                row[name] = value
                row[f"{name}_fp32_bits"] = fp32_bits(value)
            rows.append(row)
            del (
                chosen_policy,
                rejected_policy,
                chosen_reference,
                rejected_reference,
                chosen_delta,
                rejected_delta,
                objective,
            )
            force_reshard = getattr(model, "_force_reshard", None)
            if callable(force_reshard):
                force_reshard()
    return rows


def summarize_pointwise_records(
    records: list[dict[str, Any]], *, expected_shards: int, pairs_per_shard: int
) -> dict[str, Any]:
    expected_rows = expected_shards * pairs_per_shard
    _require(len(records) == expected_rows, f"zero-margin ledger has {len(records)} rather than {expected_rows} rows")
    pair_ids = [str(row.get("pair_id", "")) for row in records]
    _require(
        all(pair_ids) and len(set(pair_ids)) == expected_rows, "zero-margin ledger pair IDs are missing or duplicated"
    )
    shard_counts = Counter(int(row.get("source_shard", 0)) for row in records)
    _require(
        shard_counts == Counter({index: pairs_per_shard for index in range(1, expected_shards + 1)}),
        "zero-margin shard counts drift",
    )
    for source_shard in range(1, expected_shards + 1):
        positions = sorted(int(row["within_shard_index"]) for row in records if row["source_shard"] == source_shard)
        _require(positions == list(range(pairs_per_shard)), f"zero-margin shard {source_shard} positions drift")

    expected = {
        "chosen_delta": (0.0, POSITIVE_ZERO_FP32_BITS),
        "rejected_delta": (0.0, POSITIVE_ZERO_FP32_BITS),
        "dpo_margin": (0.0, POSITIVE_ZERO_FP32_BITS),
        "dpo_loss": (LOG2_FP32, LOG2_FP32_BITS),
    }
    metrics: dict[str, Any] = {}
    all_violation_ids: set[str] = set()
    for metric, (expected_value, expected_bits) in expected.items():
        values: list[float] = []
        violation_ids: list[str] = []
        for row in records:
            value = row.get(metric)
            _require(
                isinstance(value, (int, float)) and math.isfinite(float(value)), f"{row['pair_id']}: invalid {metric}"
            )
            value = float(value)
            observed_bits = fp32_bits(value)
            _require(
                row.get(f"{metric}_fp32_bits") == observed_bits, f"{row['pair_id']}: declared {metric} bits drift"
            )
            values.append(value)
            if observed_bits != expected_bits:
                violation_ids.append(str(row["pair_id"]))
                all_violation_ids.add(str(row["pair_id"]))
        metrics[metric] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "max_abs_delta_from_expected": max(abs(value - expected_value) for value in values),
            "nonzero_or_nonexpected_count": sum(fp32_bits(value) != expected_bits for value in values),
            "expected_value": expected_value,
            "expected_fp32_bits": expected_bits,
            "bitwise_violation_count": len(violation_ids),
            "bitwise_violation_pair_ids": violation_ids,
        }
    reference_values = [
        [
            row["source_shard"],
            row["within_shard_index"],
            row["pair_id"],
            row["chosen_reference_logp_fp32_bits"],
            row["rejected_reference_logp_fp32_bits"],
        ]
        for row in records
    ]
    return {
        "checked_active_pairs": expected_rows,
        "checked_pair_ids_sha256": _stable_digest(pair_ids),
        "cached_reference_values_sha256": _stable_digest(reference_values),
        "source_shards": expected_shards,
        "pairs_per_shard": pairs_per_shard,
        "metrics": metrics,
        "any_violation_count": len(all_violation_ids),
        "any_violation_pair_ids": sorted(all_violation_ids),
        "all_pointwise_exact_and_bitwise": not all_violation_ids,
    }


def validate_preflight_artifacts(
    root: Path,
    *,
    expected_shards: int,
    pairs_per_shard: int,
    expected_world_size: int,
    expected_beta: float,
    expected_learning_rate: float,
) -> dict[str, Any]:
    receipt_path = root / "ZERO_MARGIN_PREFLIGHT.json"
    ledger_path = root / "zero_margin" / "all_pairs.jsonl"
    _require(receipt_path.is_file() and not receipt_path.is_symlink(), "missing regular zero-margin receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = _read_jsonl(ledger_path)
    _require(_sha256_path(ledger_path) == receipt.get("pointwise_ledger_sha256"), "merged ledger hash drift")
    summary = summarize_pointwise_records(records, expected_shards=expected_shards, pairs_per_shard=pairs_per_shard)
    _require(receipt.get("pointwise_summary") == summary, "pointwise zero-margin summary drift")
    _require(
        receipt.get("cached_reference_values_sha256") == summary["cached_reference_values_sha256"],
        "reference cache hash drift",
    )
    state_names = (
        "policy_state_before_reference_sha256",
        "reference_capture_policy_state_sha256",
        "policy_state_before_recompute_sha256",
        "policy_state_after_recompute_sha256",
    )
    state_hashes = [receipt.get(name) for name in state_names]
    _require(
        all(isinstance(value, str) and len(value) == 64 for value in state_hashes),
        "missing policy/reference state hash",
    )
    _require(len(set(state_hashes)) == 1, "stale or mismatched policy/reference state hash")
    for name in (
        "trainer_global_step_before",
        "trainer_global_step_after",
        "optimizer_step_count_before",
        "optimizer_step_count_after",
        "model_update_record_count_before",
        "model_update_record_count_after",
    ):
        _require(receipt.get(name) == 0, f"{name} is not zero")
    _require(receipt.get("world_size") == expected_world_size, "zero-margin world size drift")
    _require(receipt.get("beta") == expected_beta, "zero-margin beta drift")
    _require(receipt.get("learning_rate") == expected_learning_rate, "zero-margin learning-rate drift")
    local_ledgers = receipt.get("rank_ledgers", [])
    _require(len(local_ledgers) == expected_world_size, "rank-ledger cardinality drift")
    for record in local_ledgers:
        path = Path(record["path"])
        _require(path.is_file() and not path.is_symlink(), f"missing regular rank ledger: {path}")
        _require(_sha256_path(path) == record["sha256"], f"rank ledger hash drift: {path}")
    _require(receipt.get("actual_cached_reference_path") is True, "preflight did not use actual cached-reference path")
    _require(
        receipt.get("comparison") == "exact_fp32_value_and_bit_pattern_no_tolerance", "zero-margin comparison drift"
    )
    _require(receipt.get("rounded_mean_is_sufficient") is False, "rounded mean cannot pass zero-margin audit")
    if not summary["all_pointwise_exact_and_bitwise"]:
        raise RuntimeError("pointwise zero-margin violation: " + ",".join(summary["any_violation_pair_ids"]))
    _require(
        receipt.get("status") == "pass_pointwise_exact_bitwise" and receipt.get("passed") is True,
        "zero-margin receipt is not passing",
    )
    return receipt


class DPOZeroMarginPreflightModel(DPOSALMAutomodel):
    """Native DPO model variant whose successful terminal condition is zero updates."""

    def on_train_end(self) -> None:
        if int(self.trainer.global_step) != 0 or self._metrics:
            raise RuntimeError("zero-margin preflight advanced the DPO optimizer/update trajectory")
        ready = self._output_root / "ZERO_MARGIN_PREFLIGHT_READY.json"
        if not ready.is_file():
            raise RuntimeError("zero-margin preflight ended without a passing ready receipt")


class ZeroMarginPreflightCallback(Callback):
    """Audit the real initial DPO cache after ``on_fit_start`` and stop before batch 1."""

    def __init__(
        self,
        output_root: Path,
        *,
        expected_rows: int,
        pairs_per_shard: int,
        source_shards: int,
        world_size: int,
        beta: float,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.output_root = Path(output_root)
        self.expected_rows = int(expected_rows)
        self.pairs_per_shard = int(pairs_per_shard)
        self.source_shards = int(source_shards)
        self.world_size = int(world_size)
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        _require(
            self.expected_rows == self.pairs_per_shard * self.source_shards,
            "zero-margin expected-row accounting drift",
        )
        _require(math.isfinite(self.beta) and self.beta > 0.0, "zero-margin beta must be finite and positive")
        _require(
            math.isfinite(self.learning_rate) and self.learning_rate > 0.0,
            "zero-margin learning rate must be finite and positive",
        )
        self._before_reference: dict[str, Any] | None = None

    def on_fit_start(self, trainer: Any, pl_module: DPOSALMAutomodel) -> None:
        _require(int(trainer.global_step) == 0, "zero-margin preflight began after an optimizer update")
        _require(_optimizer_step_count(pl_module) == 0, "AdamW state is nonzero before reference capture")
        observed_learning_rates = _optimizer_learning_rates(pl_module)
        _require(float(pl_module.cfg.dpo.beta) == self.beta, "model/callback DPO beta drift")
        _require(
            observed_learning_rates and all(value == self.learning_rate for value in observed_learning_rates),
            f"optimizer learning-rate drift: expected {self.learning_rate}, observed {observed_learning_rates}",
        )
        self._before_reference = _global_surface_receipt(pl_module)

    def on_train_start(self, trainer: Any, pl_module: DPOSALMAutomodel) -> None:
        _require(self._before_reference is not None, "missing pre-reference policy hash")
        _require(dist.is_initialized(), "distributed zero-margin preflight requires an initialized process group")
        _require(
            trainer.fit_loop.min_steps in (None, 0) and trainer.fit_loop.min_epochs in (None, 0),
            "zero-margin preflight cannot guarantee a pre-batch stop with nonzero minimum steps or epochs",
        )
        rank = dist.get_rank()
        runtime_world_size = dist.get_world_size()
        _require(runtime_world_size == self.world_size, "zero-margin runtime world-size drift")
        _require(
            int(trainer.global_step) == 0 and not pl_module._metrics, "DPO update occurred before zero-margin audit"
        )
        optimizer_before = _optimizer_step_count(pl_module)
        reference_state = _global_surface_receipt(pl_module)
        shards = trainer.datamodule.local_shards()

        local_error: str | None = None
        local_rows: list[dict[str, Any]] = []
        try:
            local_rows = audit_local_pairs(
                pl_module,
                shards,
                rank=rank,
                world_size=runtime_world_size,
                pairs_per_shard=self.pairs_per_shard,
                beta=self.beta,
            )
        except Exception as error:  # noqa: BLE001
            local_error = f"{type(error).__name__}: {error}"
        errors = _all_rank_values(local_error)
        if any(error is not None for error in errors):
            if rank == 0:
                _write_json(
                    self.output_root / "ZERO_MARGIN_PREFLIGHT_FAILED.json",
                    {"status": "failed_local_audit", "rank_errors": errors, "optimizer_step_count": optimizer_before},
                )
            raise RuntimeError(f"rank-local zero-margin audit failed: {errors}")

        rank_path = self.output_root / "zero_margin" / f"rank{rank:02d}.jsonl"
        _write_jsonl(rank_path, local_rows)
        local_metadata = {
            "rank": rank,
            "path": str(rank_path),
            "sha256": _sha256_path(rank_path),
            "rows": len(local_rows),
            "pair_ids_sha256": _stable_digest([row["pair_id"] for row in local_rows]),
        }
        rank_ledgers = _all_rank_values(local_metadata)
        after_state = _global_surface_receipt(pl_module)
        optimizer_after = _optimizer_step_count(pl_module)
        trainer_step_after = int(trainer.global_step)
        update_records_after = len(pl_module._metrics)
        dist.barrier()

        result: dict[str, Any] | None = None
        root_error: str | None = None
        if rank == 0:
            try:
                records: list[dict[str, Any]] = []
                for metadata in sorted(rank_ledgers, key=lambda item: item["rank"]):
                    records.extend(_read_jsonl(Path(metadata["path"])))
                records.sort(key=lambda row: (row["source_shard"], row["within_shard_index"]))
                merged_path = self.output_root / "zero_margin" / "all_pairs.jsonl"
                _write_jsonl(merged_path, records)
                summary = summarize_pointwise_records(
                    records, expected_shards=self.source_shards, pairs_per_shard=self.pairs_per_shard
                )
                state_errors = []
                state_receipts = {
                    "before_reference": self._before_reference,
                    "reference_capture": reference_state,
                    "before_recompute": reference_state,
                    "after_recompute": after_state,
                }
                global_state_hashes = [value["global_sha256"] for value in state_receipts.values()]
                if len(set(global_state_hashes)) != 1:
                    state_errors.append("policy/reference selected-surface hash mismatch")
                if (
                    optimizer_before != 0
                    or optimizer_after != 0
                    or trainer_step_after != 0
                    or update_records_after != 0
                ):
                    state_errors.append("optimizer or DPO update count is nonzero")
                passed = summary["all_pointwise_exact_and_bitwise"] and not state_errors
                receipt = {
                    "schema": "speechlm2.dpo.zero-margin-preflight.v1",
                    "status": "pass_pointwise_exact_bitwise" if passed else "failed_exact_identity",
                    "passed": passed,
                    "actual_cached_reference_path": True,
                    "comparison": "exact_fp32_value_and_bit_pattern_no_tolerance",
                    "rounded_mean_is_sufficient": False,
                    "on_any_deviation": "stop_and_report_raw_deltas_without_selecting_a_tolerance",
                    "expected_rows": self.expected_rows,
                    "source_shards": self.source_shards,
                    "pairs_per_shard": self.pairs_per_shard,
                    "world_size": self.world_size,
                    "beta": self.beta,
                    "learning_rate": self.learning_rate,
                    "trainer_global_step_before": 0,
                    "trainer_global_step_after": trainer_step_after,
                    "optimizer_step_count_before": optimizer_before,
                    "optimizer_step_count_after": optimizer_after,
                    "model_update_record_count_before": 0,
                    "model_update_record_count_after": update_records_after,
                    "policy_state_before_reference_sha256": self._before_reference["global_sha256"],
                    "reference_capture_policy_state_sha256": reference_state["global_sha256"],
                    "policy_state_before_recompute_sha256": reference_state["global_sha256"],
                    "policy_state_after_recompute_sha256": after_state["global_sha256"],
                    "rank_policy_state_sha256": state_receipts,
                    "source_dcp_metadata_sha256": _sha256_file(pl_module._initial_checkpoint / ".metadata"),
                    "cached_reference_values_sha256": summary["cached_reference_values_sha256"],
                    "pointwise_ledger_path": str(merged_path),
                    "pointwise_ledger_sha256": _sha256_path(merged_path),
                    "pointwise_summary": summary,
                    "rank_ledgers": sorted(rank_ledgers, key=lambda item: item["rank"]),
                    "state_errors": state_errors,
                }
                receipt_path = self.output_root / "ZERO_MARGIN_PREFLIGHT.json"
                _write_json(receipt_path, receipt)
                if passed:
                    validate_preflight_artifacts(
                        self.output_root,
                        expected_shards=self.source_shards,
                        pairs_per_shard=self.pairs_per_shard,
                        expected_world_size=self.world_size,
                        expected_beta=self.beta,
                        expected_learning_rate=self.learning_rate,
                    )
                    ready = {
                        "schema": "speechlm2.dpo.zero-margin-preflight-ready.v1",
                        "passed": True,
                        "receipt": str(receipt_path),
                        "receipt_sha256": _sha256_path(receipt_path),
                        "pointwise_ledger_sha256": receipt["pointwise_ledger_sha256"],
                        "optimizer_step_count": 0,
                    }
                    _write_json(self.output_root / "ZERO_MARGIN_PREFLIGHT_READY.json", ready)
                result = {
                    "passed": passed,
                    "state_errors": state_errors,
                    "violation_pair_ids": summary["any_violation_pair_ids"],
                }
            except Exception as error:  # noqa: BLE001
                root_error = f"{type(error).__name__}: {error}"
                _write_json(
                    self.output_root / "ZERO_MARGIN_PREFLIGHT_FAILED.json",
                    {"status": "failed_merge_or_validation", "error": root_error},
                )
        broadcast = [result, root_error]
        dist.broadcast_object_list(broadcast, src=0)
        result, root_error = broadcast
        if root_error is not None or result is None or not result["passed"]:
            raise RuntimeError(
                f"zero-margin preflight failed before optimizer step: result={result} error={root_error}"
            )
        trainer.should_stop = True
