# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path

import pytest
import torch

from nemo.collections.speechlm2.dpo.export import TensorSpec, check_serving_contract, read_surface_contract, shard_plan
from nemo.collections.speechlm2.dpo.surface import selected_parameter_names


def _spec(dtype: torch.dtype) -> TensorSpec:
    return TensorSpec(shape=(2, 3), dtype=dtype, payload_bytes=2 * 3 * torch.empty((), dtype=dtype).element_size())


def _trajectory() -> dict:
    return {
        "lora": False,
        "surface": {
            "names": list(selected_parameter_names()),
            "tensor_count": 269,
            "scalar_count": 1_074_318_016,
            "dtypes": ["torch.float32"],
        },
    }


def test_read_surface_contract_requires_the_declared_historical_surface(tmp_path: Path):
    path = tmp_path / "TRAJECTORY.json"
    path.write_text(__import__("json").dumps(_trajectory()), encoding="utf-8")
    assert read_surface_contract(path) == selected_parameter_names()
    invalid = _trajectory()
    invalid["surface"]["names"] = invalid["surface"]["names"][:-1]
    path.write_text(__import__("json").dumps(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact Hero2 DPO FP32 surface"):
        read_surface_contract(path)


def test_export_contract_accepts_only_the_declared_bf16_to_fp32_surface():
    selected = selected_parameter_names()
    baseline = {name: _spec(torch.bfloat16) for name in selected}
    baseline["frozen"] = _spec(torch.bfloat16)
    candidate = {name: _spec(torch.float32) for name in selected}
    candidate["frozen"] = _spec(torch.bfloat16)
    report = check_serving_contract(candidate=candidate, baseline=baseline, selected_fp32=selected)
    assert report["serving_tensor_count"] == 270
    assert report["fp32_surface_tensor_count"] == 269


def test_export_contract_rejects_precision_drift_outside_selected_surface():
    selected = selected_parameter_names()
    baseline = {name: _spec(torch.bfloat16) for name in selected}
    baseline["frozen"] = _spec(torch.bfloat16)
    candidate = {name: _spec(torch.float32) for name in selected}
    candidate["frozen"] = _spec(torch.float32)
    with pytest.raises(RuntimeError, match="outside declared"):
        check_serving_contract(candidate=candidate, baseline=baseline, selected_fp32=selected)


def test_shard_plan_is_ordered_bounded_and_exhaustive():
    specs = {
        "a": TensorSpec((2,), torch.float32, 8),
        "b": TensorSpec((3,), torch.float32, 12),
        "c": TensorSpec((1,), torch.float32, 4),
    }
    assert shard_plan(specs, 16) == [["a"], ["b", "c"]]
    with pytest.raises(ValueError, match="positive"):
        shard_plan(specs, 0)


def test_r22_server_entrypoint_uses_a_verified_staged_package_artifact():
    script = (
        Path(__file__).resolve().parents[3]
        / "examples/speechlm2/serve_salm_dpo_hero2_vllm_r22.sh"
    ).read_text(encoding="utf-8")
    assert "R22_NEMO_SERVER_FINGERPRINT_SHA256" in script
    assert "stage_verified_r22_package" in script
    assert "! -name collections" in script
    assert "-P 8 cp -a -t \"$stage_root/source/nemo/collections\"" in script
    assert "--force-reinstall \"$R22_STAGE_ROOT/source\"" in script
    assert "--verify-r22-package-install" in script
    assert "python3 -m venv \"$target/venv\"" in script
    assert "PYTHONPATH" not in script
