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

from nemo.collections.speechlm2.parts.hf_hub import (
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_SINGLE_FILE,
    _inject_local_artifact_paths,
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
    for shard in shards:
        shard.write_bytes(b"")

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
