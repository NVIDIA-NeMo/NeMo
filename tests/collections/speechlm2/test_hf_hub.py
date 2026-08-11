# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from nemo.collections.speechlm2.parts.hf_hub import (
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_SINGLE_FILE,
    _inject_local_artifact_paths,
    _load_state_dict_with_dtensors,
    _resolve_safetensors_weight_dir,
)


def _cached_file_kwargs():
    return {
        "cache_dir": None,
        "force_download": False,
        "local_files_only": True,
        "token": None,
        "revision": None,
        "_raise_exceptions_for_gated_repo": False,
        "_raise_exceptions_for_missing_entries": False,
        "_raise_exceptions_for_connection_errors": False,
    }


def _write_local_export_artifacts(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "llm_backbone").mkdir()
    (tmp_path / "llm_backbone" / "config.json").write_text("{}")


def test_inject_local_artifact_paths_salm_config(tmp_path):
    _write_local_export_artifacts(tmp_path)
    cfg = {
        "pretrained_llm": "remote-llm",
        "pretrained_asr": "remote-asr",
    }

    _inject_local_artifact_paths(cfg, str(tmp_path), _cached_file_kwargs())

    assert cfg["pretrained_llm"] == str(tmp_path / "llm_backbone")
    assert cfg["pretrained_asr"] == "remote-asr"
    assert cfg["tokenizer_path"] == str(tmp_path)


def test_inject_local_artifact_paths_duplex_eartts_config(tmp_path):
    _write_local_export_artifacts(tmp_path)
    cfg = {
        "pretrained_lm_name": "remote-llm",
        "tts_config": {},
    }

    _inject_local_artifact_paths(cfg, str(tmp_path), _cached_file_kwargs())

    assert cfg["pretrained_lm_name"] == str(tmp_path / "llm_backbone")
    assert cfg["tokenizer_path"] == str(tmp_path)


def test_inject_local_artifact_paths_no_artifacts_keeps_old_config(tmp_path):
    cfg = {
        "pretrained_llm": "remote-llm",
        "pretrained_weights": True,
    }

    _inject_local_artifact_paths(cfg, str(tmp_path), _cached_file_kwargs())

    assert cfg == {
        "pretrained_llm": "remote-llm",
        "pretrained_weights": True,
    }


def test_resolve_safetensors_weight_dir_accepts_sharded_checkpoint(tmp_path):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    shards = [
        tmp_path / "model-00001-of-00002.safetensors",
        tmp_path / "model-00002-of-00002.safetensors",
    ]
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": shards[0].name,
                    "second": shards[1].name,
                }
            }
        )
    )
    save_file({"first": torch.tensor([1.0])}, shards[0])
    save_file({"second": torch.tensor([2.0])}, shards[1])

    resolved = {
        SAFETENSORS_SINGLE_FILE: None,
        SAFETENSORS_INDEX_FILE: str(index),
        **{shard.name: str(shard) for shard in shards},
    }
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, filename, **_kwargs: resolved.get(filename),
    ):
        assert _resolve_safetensors_weight_dir("sharded-model", {}) == tmp_path


def test_resolve_safetensors_weight_dir_keeps_single_file_checkpoint(tmp_path):
    weights = tmp_path / SAFETENSORS_SINGLE_FILE
    weights.write_bytes(b"")
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, filename, **_kwargs: str(weights)
        if filename == SAFETENSORS_SINGLE_FILE
        else None,
    ):
        assert _resolve_safetensors_weight_dir("single-model", {}) == tmp_path


def test_resolve_safetensors_weight_dir_rejects_malformed_index(tmp_path):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    index.write_text("{")
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, filename, **_kwargs: str(index)
        if filename == SAFETENSORS_INDEX_FILE
        else None,
    ):
        with pytest.raises(RuntimeError, match="Invalid model.safetensors.index.json"):
            _resolve_safetensors_weight_dir("malformed-model", {})


@pytest.mark.parametrize("weight_map", [{}, [], {"": "model.safetensors"}])
def test_resolve_safetensors_weight_dir_rejects_invalid_weight_map(tmp_path, weight_map):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    index.write_text(json.dumps({"weight_map": weight_map}))
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, filename, **_kwargs: str(index)
        if filename == SAFETENSORS_INDEX_FILE
        else None,
    ):
        with pytest.raises(RuntimeError, match="Invalid model.safetensors.index.json"):
            _resolve_safetensors_weight_dir("invalid-map-model", {})


def test_resolve_safetensors_weight_dir_rejects_missing_indexed_shard(tmp_path):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    index.write_text(json.dumps({"weight_map": {"tensor": "missing.safetensors"}}))
    resolved = {
        SAFETENSORS_SINGLE_FILE: None,
        SAFETENSORS_INDEX_FILE: str(index),
    }
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, filename, **_kwargs: resolved.get(filename),
    ):
        with pytest.raises(RuntimeError, match="Missing safetensors shard"):
            _resolve_safetensors_weight_dir("missing-shard-model", {})


@pytest.mark.parametrize(
    "filename",
    ["../outside.safetensors", "/absolute/model.safetensors", "model.bin", 7],
)
def test_resolve_safetensors_weight_dir_rejects_unsafe_shard_filename(tmp_path, filename):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    index.write_text(json.dumps({"weight_map": {"tensor": filename}}))
    resolved = {
        SAFETENSORS_SINGLE_FILE: None,
        SAFETENSORS_INDEX_FILE: str(index),
    }
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, name, **_kwargs: resolved.get(name),
    ):
        with pytest.raises(RuntimeError, match="Invalid shard filename"):
            _resolve_safetensors_weight_dir("unsafe-shard-model", {})


def test_resolve_safetensors_weight_dir_rejects_unindexed_safetensors(tmp_path):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    shard = tmp_path / "model-00001-of-00001.safetensors"
    extra = tmp_path / "adapter.safetensors"
    index.write_text(json.dumps({"weight_map": {"tensor": shard.name}}))
    shard.write_bytes(b"indexed")
    extra.write_bytes(b"not indexed")
    resolved = {
        SAFETENSORS_SINGLE_FILE: None,
        SAFETENSORS_INDEX_FILE: str(index),
        shard.name: str(shard),
    }
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, name, **_kwargs: resolved.get(name),
    ):
        with pytest.raises(RuntimeError, match="Unindexed safetensors files"):
            _resolve_safetensors_weight_dir("extra-file-model", {})


def test_resolve_safetensors_weight_dir_rejects_index_header_drift(tmp_path):
    index = tmp_path / SAFETENSORS_INDEX_FILE
    shard = tmp_path / "model-00001-of-00001.safetensors"
    index.write_text(json.dumps({"weight_map": {"declared": shard.name}}))
    save_file({"actual": torch.tensor([1.0])}, shard)
    resolved = {
        SAFETENSORS_SINGLE_FILE: None,
        SAFETENSORS_INDEX_FILE: str(index),
        shard.name: str(shard),
    }
    with patch(
        "nemo.collections.speechlm2.parts.hf_hub.cached_file",
        side_effect=lambda _model_id, name, **_kwargs: resolved.get(name),
    ):
        with pytest.raises(RuntimeError, match="index/header mapping mismatch"):
            _resolve_safetensors_weight_dir("header-drift-model", {})


def test_distributed_reader_loads_valid_two_shard_safetensors(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"first": torch.tensor([1.0, 2.0])}, first)
    save_file({"second": torch.tensor([3.0, 4.0])}, second)
    (tmp_path / SAFETENSORS_INDEX_FILE).write_text(
        json.dumps({"weight_map": {"first": first.name, "second": second.name}})
    )

    class TwoParameterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.zeros(2))
            self.second = torch.nn.Parameter(torch.zeros(2))

    model = TwoParameterModel()
    _load_state_dict_with_dtensors(model, str(tmp_path))
    torch.testing.assert_close(model.first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.second, torch.tensor([3.0, 4.0]))


def test_distributed_reader_rejects_partial_parameter_coverage(tmp_path):
    save_file({"first": torch.tensor([1.0, 2.0])}, tmp_path / SAFETENSORS_SINGLE_FILE)

    class TwoParameterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.zeros(2))
            self.second = torch.nn.Parameter(torch.zeros(2))

    with pytest.raises(RuntimeError, match="missing model parameters.*second"):
        _load_state_dict_with_dtensors(TwoParameterModel(), str(tmp_path))


def test_distributed_reader_loads_single_file_safetensors(tmp_path):
    save_file({"weight": torch.tensor([5.0, 6.0])}, tmp_path / SAFETENSORS_SINGLE_FILE)

    class OneParameterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    model = OneParameterModel()
    _load_state_dict_with_dtensors(model, str(tmp_path))
    torch.testing.assert_close(model.weight, torch.tensor([5.0, 6.0]))
