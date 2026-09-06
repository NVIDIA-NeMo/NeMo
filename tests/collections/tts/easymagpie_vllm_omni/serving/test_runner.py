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
"""Tests for the vLLM-Omni 0.24 streaming runner compatibility layer."""
from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import yaml

import easymagpie_vllm_omni.runner as runner_module
from conftest import EASYMAGPIE_ROOT
from easymagpie_vllm_omni.runner import (
    EasyMagpieGPUARModelRunner,
    batch_waveforms_to_cpu,
    merge_streaming_additional_information,
)

WORKER_CLS = "easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker"


def test_streaming_update_preserves_model_state_and_replaces_latest_chunk():
    cached = {
        "decode_offset": 7,
        "text_tokens": [10, 20],
        "text_token": [20],
        "meta": {"num_processed_tokens": 3},
    }

    merged = merge_streaming_additional_information(cached, {"text_token": [30]})

    assert merged["decode_offset"] == 7
    assert merged["text_tokens"] == [10, 20]
    assert merged["text_token"] == [30]
    assert merged["meta"]["num_processed_tokens"] == 0
    assert merged["meta"]["resumable"] is True


def test_streaming_update_accumulates_declared_tensor_keys():
    cached = {"hidden_states": {"output": torch.tensor([[1.0]])}}
    incoming = {"hidden_states": {"output": torch.tensor([[2.0]])}}

    merged = merge_streaming_additional_information(
        cached,
        incoming,
        accumulated_keys={("hidden_states", "output")},
    )

    torch.testing.assert_close(merged["hidden_states"]["output"], torch.tensor([[1.0], [2.0]]))


def test_top_level_gpu_resident_updates_are_cloned_as_resident_state():
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    request = SimpleNamespace()
    runner.requests = {"request": request}
    runner.model_intermediate_buffer = {}
    runner.model = SimpleNamespace(
        gpu_resident_buffer_keys={"last_audio_codes", ("hidden_states", "last")}
    )
    audio_codes = torch.tensor([[1, 2]])
    hidden = torch.tensor([[3.0]])

    runner._update_intermediate_buffer(
        "request",
        {"last_audio_codes": audio_codes, "hidden_states": {"last": hidden}},
    )

    cached = runner.model_intermediate_buffer["request"]
    torch.testing.assert_close(cached["last_audio_codes"], audio_codes)
    torch.testing.assert_close(cached["hidden_states"]["last"], hidden)
    assert cached["last_audio_codes"].data_ptr() != audio_codes.data_ptr()
    assert cached["hidden_states"]["last"].data_ptr() != hidden.data_ptr()
    assert request.additional_information_cpu is cached


def test_stage0_trace_records_exact_request_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "_STAGE0_TRACE_DIR", tmp_path, raising=False)
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    request = SimpleNamespace()
    runner.requests = {"request": request}
    runner.model_intermediate_buffer = {
        "request": {"decode_offset": 8, "_omni_is_prefill": False}
    }
    runner.model = SimpleNamespace(
        gpu_resident_buffer_keys={"last_audio_codes", "last_phoneme_token"}
    )

    runner._update_intermediate_buffer(
        "request",
        {
            "last_audio_codes": torch.tensor([[1, 2]]),
            "last_phoneme_token": torch.tensor([[3]]),
        },
    )

    trace_file = next(tmp_path.glob("stage0.*.jsonl"))
    row = json.loads(trace_file.read_text())
    assert row == {
        "request_id": "request",
        "decode_offset": 7,
        "phoneme_tokens": [3],
        "audio_codes": [1, 2],
    }
    cached = runner.model_intermediate_buffer["request"]
    torch.testing.assert_close(cached["last_audio_codes"], torch.tensor([[1, 2]]))
    torch.testing.assert_close(cached["last_phoneme_token"], torch.tensor([[3]]))

    cached["_omni_is_prefill"] = True
    runner._update_intermediate_buffer(
        "request",
        {
            "last_audio_codes": torch.tensor([[4, 5]]),
            "last_phoneme_token": torch.tensor([[6]]),
        },
    )
    assert len(trace_file.read_text().splitlines()) == 1


def test_async_output_uses_padded_length_to_slice_codes():
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    runner.model = SimpleNamespace(omni_pooler_payload_include_hidden=False)
    hidden = torch.zeros(40, 3)
    codes = torch.arange(80).view(40, 2)

    snapshot = runner._build_omni_async_snapshot_payload(
        hidden_states=hidden,
        staged_hidden_states_cpu=None,
        multimodal_outputs={"codes": {"audio": codes}},
    )
    output = runner._build_omni_mm_payload(
        combined_multimodal_outputs=None,
        mm_cpu={"codes.audio": codes},
        rid="request-7",
        idx=7,
        start=7,
        end=8,
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=snapshot["hidden_states"].shape[0],
        scheduled_seq_len=34,
    )

    assert snapshot["hidden_states"].shape == (40, 0)
    torch.testing.assert_close(output["codes.audio"], codes[7:8])


def test_deploy_configs_select_compatibility_worker_for_lm():
    for filename in ("easymagpie_lm.yaml", "easymagpie.yaml"):
        deploy = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / filename).read_text())
        lm_stage = next(stage for stage in deploy["stages"] if stage["stage_id"] == 0)
        assert lm_stage["engine_extras"]["worker_cls"] == WORKER_CLS


def test_batched_waveform_copy_preserves_bits_shapes_and_order():
    first = torch.tensor([0, -2147483648, 1065353216, 2143294004], dtype=torch.int32).view(torch.float32).view(2, 2)
    second = torch.tensor([1073741824], dtype=torch.int32).view(torch.float32)
    third = torch.tensor([-1082130432, 1082130432], dtype=torch.int32).view(torch.float32).view(1, 2)

    copied = batch_waveforms_to_cpu([first, second, third])

    assert [x.shape for x in copied] == [first.shape, second.shape, third.shape]
    for actual, expected in zip(copied, (first, second, third), strict=True):
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
