# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import copy
import importlib.util
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2.dpo import zero_margin


_ENTRYPOINT_PATH = Path(__file__).parents[3] / "examples" / "speechlm2" / "salm_dpo_zero_margin_preflight.py"
_ENTRYPOINT_SPEC = importlib.util.spec_from_file_location("salm_dpo_zero_margin_preflight", _ENTRYPOINT_PATH)
assert _ENTRYPOINT_SPEC is not None and _ENTRYPOINT_SPEC.loader is not None
_ENTRYPOINT = importlib.util.module_from_spec(_ENTRYPOINT_SPEC)
_ENTRYPOINT_SPEC.loader.exec_module(_ENTRYPOINT)


@dataclass
class _Pair:
    pair_id: str
    source_id: str = "source"
    active: bool = True


class _CachedReferenceHarness:
    device = torch.device("cpu")

    def __init__(self, references, policies):
        self._references = {1: references}
        self._policies = policies
        self.reshards = 0
        self.policy_pair_ids = []

    def _policy_pair(self, pair):
        self.policy_pair_ids.append(pair.pair_id)
        chosen, rejected = self._policies[pair.pair_id]
        return torch.tensor(chosen, dtype=torch.float32, requires_grad=True), torch.tensor(
            rejected, dtype=torch.float32, requires_grad=True
        )

    def _force_reshard(self):
        self.reshards += 1


def _exact_records():
    pairs = [_Pair("p0"), _Pair("p1")]
    references = [(3.0, 1.0), (-2.5, -4.0)]
    policies = {"p0": (3.0, 1.0), "p1": (-2.5, -4.0)}
    model = _CachedReferenceHarness(references, policies)
    records = zero_margin.audit_local_pairs(
        model,
        [pairs],
        rank=0,
        world_size=1,
        pairs_per_shard=2,
        beta=0.2,
    )
    return records, model


def _passing_receipt(root, records):
    rank_path = root / "zero_margin" / "rank00.jsonl"
    merged_path = root / "zero_margin" / "all_pairs.jsonl"
    zero_margin._write_jsonl(rank_path, records)
    zero_margin._write_jsonl(merged_path, records)
    summary = zero_margin.summarize_pointwise_records(records, expected_shards=1, pairs_per_shard=2)
    state_hash = "a" * 64
    receipt = {
        "status": "pass_pointwise_exact_bitwise",
        "passed": True,
        "actual_cached_reference_path": True,
        "policy_pair_forward_context": "trainer.precision_plugin.forward_context",
        "policy_pair_audio_input_dtype": "torch.float32",
        "comparison": "exact_fp32_value_and_bit_pattern_no_tolerance",
        "rounded_mean_is_sufficient": False,
        "world_size": 1,
        "beta": 0.2,
        "learning_rate": 2.5e-6,
        "trainer_global_step_before": 0,
        "trainer_global_step_after": 0,
        "optimizer_step_count_before": 0,
        "optimizer_step_count_after": 0,
        "model_update_record_count_before": 0,
        "model_update_record_count_after": 0,
        "policy_state_before_reference_sha256": state_hash,
        "reference_capture_policy_state_sha256": state_hash,
        "policy_state_before_recompute_sha256": state_hash,
        "policy_state_after_recompute_sha256": state_hash,
        "cached_reference_values_sha256": summary["cached_reference_values_sha256"],
        "pointwise_ledger_sha256": zero_margin._sha256_path(merged_path),
        "pointwise_summary": summary,
        "rank_ledgers": [
            {
                "rank": 0,
                "path": str(rank_path),
                "sha256": zero_margin._sha256_path(rank_path),
                "rows": 2,
            }
        ],
    }
    zero_margin._write_json(root / "ZERO_MARGIN_PREFLIGHT.json", receipt)
    return receipt


def test_actual_cached_reference_path_is_pointwise_exact_before_update():
    records, model = _exact_records()
    assert model.reshards == 2
    assert [row["chosen_delta_fp32_bits"] for row in records] == ["0x00000000"] * 2
    assert [row["rejected_delta_fp32_bits"] for row in records] == ["0x00000000"] * 2
    assert [row["dpo_margin_fp32_bits"] for row in records] == ["0x00000000"] * 2
    assert [row["dpo_loss_fp32_bits"] for row in records] == ["0x3f317218"] * 2
    summary = zero_margin.summarize_pointwise_records(records, expected_shards=1, pairs_per_shard=2)
    assert summary["all_pointwise_exact_and_bitwise"] is True
    assert summary["any_violation_count"] == 0
    for metric in summary["metrics"].values():
        assert metric["bitwise_violation_count"] == 0
        assert metric["nonzero_or_nonexpected_count"] == 0


def test_padding_pairs_keep_eight_rank_fsdp_forward_schedule_lockstep():
    pairs_per_shard = 434
    world_size = 8
    source_shards = 26
    all_records = []
    calls_per_rank = []
    for rank in range(world_size):
        active_count = len(range(rank, pairs_per_shard, world_size))
        shards = []
        references_by_shard = {}
        policies = {}
        for source_shard in range(1, source_shards + 1):
            pairs = [_Pair(f"s{source_shard}-r{rank}-p{index}") for index in range(active_count)]
            if active_count < 55:
                pairs.append(_Pair(pairs[-1].pair_id, active=False))
            references = [
                (float(source_shard * 100 + index), -float(source_shard * 100 + index))
                for index in range(active_count)
            ]
            if not pairs[-1].active:
                references.append(references[-1])
            policies.update({pair.pair_id: reference for pair, reference in zip(pairs, references, strict=True)})
            shards.append(pairs)
            references_by_shard[source_shard] = references
        model = _CachedReferenceHarness([], policies)
        model._references = references_by_shard

        records = zero_margin.audit_local_pairs(
            model,
            shards,
            rank=rank,
            world_size=world_size,
            pairs_per_shard=pairs_per_shard,
            beta=0.2,
        )

        calls_per_rank.append(len(model.policy_pair_ids) // source_shards)
        assert model.reshards == 55 * source_shards
        assert len(records) == active_count * source_shards
        assert all(record["active"] is True for record in records)
        all_records.extend(records)

    assert calls_per_rank == [55] * world_size
    assert len(all_records) == pairs_per_shard * source_shards
    assert Counter(record["source_shard"] for record in all_records) == Counter(
        {source_shard: pairs_per_shard for source_shard in range(1, source_shards + 1)}
    )
    for source_shard in range(1, source_shards + 1):
        assert sorted(
            record["within_shard_index"] for record in all_records if record["source_shard"] == source_shard
        ) == list(range(pairs_per_shard))


def test_stale_cached_reference_reports_exact_pair_id_and_nonzero_delta():
    pairs = [_Pair("stale")]
    model = _CachedReferenceHarness([(2.5, 1.0)], {"stale": (3.0, 1.0)})
    records = zero_margin.audit_local_pairs(
        model,
        [pairs],
        rank=0,
        world_size=1,
        pairs_per_shard=1,
        beta=0.2,
    )
    summary = zero_margin.summarize_pointwise_records(records, expected_shards=1, pairs_per_shard=1)
    assert summary["all_pointwise_exact_and_bitwise"] is False
    assert summary["any_violation_pair_ids"] == ["stale"]
    assert summary["metrics"]["chosen_delta"]["min"] == 0.5
    assert summary["metrics"]["chosen_delta"]["nonzero_or_nonexpected_count"] == 1
    assert summary["metrics"]["dpo_margin"]["bitwise_violation_pair_ids"] == ["stale"]


def test_negative_zero_is_a_bitwise_hard_failure():
    records, _ = _exact_records()
    records[0]["chosen_delta"] = -0.0
    records[0]["chosen_delta_fp32_bits"] = zero_margin.fp32_bits(-0.0)
    summary = zero_margin.summarize_pointwise_records(records, expected_shards=1, pairs_per_shard=2)
    assert summary["metrics"]["chosen_delta"]["min"] == 0.0
    assert summary["metrics"]["chosen_delta"]["mean"] == 0.0
    assert summary["metrics"]["chosen_delta"]["bitwise_violation_pair_ids"] == ["p0"]
    assert summary["all_pointwise_exact_and_bitwise"] is False


def test_receipt_validator_rejects_mismatched_state_or_nonzero_optimizer(tmp_path):
    records, _ = _exact_records()
    receipt = _passing_receipt(tmp_path, records)
    validated = zero_margin.validate_preflight_artifacts(
        tmp_path,
        expected_shards=1,
        pairs_per_shard=2,
        expected_world_size=1,
        expected_beta=0.2,
        expected_learning_rate=2.5e-6,
    )
    assert validated["optimizer_step_count_after"] == 0

    mismatched = copy.deepcopy(receipt)
    mismatched["policy_state_before_recompute_sha256"] = "b" * 64
    zero_margin._write_json(tmp_path / "ZERO_MARGIN_PREFLIGHT.json", mismatched)
    with pytest.raises(RuntimeError, match="stale or mismatched"):
        zero_margin.validate_preflight_artifacts(
            tmp_path,
            expected_shards=1,
            pairs_per_shard=2,
            expected_world_size=1,
            expected_beta=0.2,
            expected_learning_rate=2.5e-6,
        )

    advanced = copy.deepcopy(receipt)
    advanced["optimizer_step_count_after"] = 1
    zero_margin._write_json(tmp_path / "ZERO_MARGIN_PREFLIGHT.json", advanced)
    with pytest.raises(RuntimeError, match="optimizer_step_count_after is not zero"):
        zero_margin.validate_preflight_artifacts(
            tmp_path,
            expected_shards=1,
            pairs_per_shard=2,
            expected_world_size=1,
            expected_beta=0.2,
            expected_learning_rate=2.5e-6,
        )


def test_preflight_entrypoint_removes_inherited_minimum_step_and_epoch_floors(tmp_path):
    base = tmp_path / "base.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {"inherited": True},
                "trainer": {"min_steps": 12, "min_epochs": 3, "max_steps": 100},
            }
        ),
        base,
    )
    cfg = OmegaConf.create(
        {
            "model": {"base_experiment_config": str(base)},
            "dpo": {"beta": 0.2, "learning_rate": 2.5e-6},
            "trainer": {
                "devices": 8,
                "num_nodes": 1,
                "precision": "bf16-mixed",
                "max_steps": 1,
                "gradient_clip_val": 1.0,
            },
        }
    )
    _, trainer = _ENTRYPOINT._prepare_model_config(cfg)
    assert trainer.max_steps == 1
    assert trainer.min_steps == 0
    assert trainer.min_epochs == 0
    assert trainer.enable_checkpointing is False
