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
"""vLLM-Omni 0.24 streaming-input compatibility classes.

vLLM-Omni 0.24 no longer merges a resumed request's
``additional_information`` into ``model_intermediate_buffer``. Consequently,
per-chunk EasyMagpie ``text_token`` payloads reach the scheduler but not the
model runner. The custom runner restores the merge performed by 0.21 while
preserving model-generated state such as ``decode_offset`` and ``text_tokens``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.worker import gpu_ar_worker
from vllm_omni.worker import gpu_generation_worker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker

_trace_dir = os.environ.get("EASYMAGPIE_STAGE0_TRACE_DIR")
_STAGE0_TRACE_DIR = Path(_trace_dir) if _trace_dir else None


def merge_streaming_additional_information(
    cached: dict[str, Any],
    incoming: dict[str, Any],
    accumulated_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge one streaming chunk without dropping persistent model state."""
    accumulated_keys = accumulated_keys or set()
    merged = dict(cached)

    for key, value in incoming.items():
        if not isinstance(value, dict):
            merged[key] = value
            continue

        old_value = merged.get(key)
        merged_sub = dict(old_value) if isinstance(old_value, dict) else {}
        for subkey, subvalue in value.items():
            if (key, subkey) in accumulated_keys and isinstance(subvalue, torch.Tensor):
                new_tensor = subvalue.detach().to("cpu").contiguous()
                old_tensor = merged_sub.get(subkey)
                merged_sub[subkey] = new_tensor if old_tensor is None else torch.cat((old_tensor, new_tensor), dim=0)
            else:
                merged_sub[subkey] = subvalue
        merged[key] = merged_sub

    meta = dict(merged.get("meta", {}))
    meta["num_processed_tokens"] = 0
    meta["resumable"] = True
    merged["meta"] = meta
    return merged


class EasyMagpieGPUARModelRunner(GPUARModelRunner):
    """GPU AR runner that restores streaming chunk metadata propagation."""

    def _build_omni_async_snapshot_payload(
        self,
        *,
        hidden_states: torch.Tensor,
        staged_hidden_states_cpu: torch.Tensor | None,
        multimodal_outputs: Any,
    ) -> dict[str, Any]:
        payload = super()._build_omni_async_snapshot_payload(
            hidden_states=hidden_states,
            staged_hidden_states_cpu=staged_hidden_states_cpu,
            multimodal_outputs=multimodal_outputs,
        )
        payload.setdefault("hidden_states", hidden_states[:, :0])
        return payload

    def _update_intermediate_buffer(self, req_id: str, upd: dict) -> None:
        if not isinstance(upd, dict) or not upd:
            return
        request = self.requests.get(req_id)
        if request is None:
            return

        model = getattr(self, "model", None)
        gpu_keys = getattr(model, "gpu_resident_buffer_keys", set())
        top_level_gpu_keys = {key for key in gpu_keys if isinstance(key, str)}
        nested_gpu_keys = {key for key in gpu_keys if isinstance(key, tuple) and len(key) == 2}
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for key, value in upd.items():
            if isinstance(value, dict):
                existing_sub = existing.setdefault(key, {})
                resident_qualifiers = {qualifier for type_key, qualifier in nested_gpu_keys if type_key == key}
                for qualifier, subvalue in value.items():
                    self._store_value(existing_sub, qualifier, subvalue, resident_qualifiers)
            else:
                self._store_value(existing, key, value, top_level_gpu_keys)
        self._trace_stage0_prediction(req_id, upd, existing)
        request.additional_information_cpu = existing

    @staticmethod
    def _trace_stage0_prediction(req_id: str, upd: dict, existing: dict) -> None:
        if _STAGE0_TRACE_DIR is None or existing.get("_omni_is_prefill", False):
            return
        audio = upd.get("last_audio_codes")
        phoneme = upd.get("last_phoneme_token")
        if not isinstance(audio, torch.Tensor) or not isinstance(phoneme, torch.Tensor):
            return

        phoneme_count = phoneme.numel()
        values = torch.cat((phoneme.reshape(-1), audio.reshape(-1))).to("cpu").tolist()
        row = {
            "request_id": req_id,
            "decode_offset": int(existing.get("decode_offset", 1)) - 1,
            "phoneme_tokens": values[:phoneme_count],
            "audio_codes": values[phoneme_count:],
        }
        _STAGE0_TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = _STAGE0_TRACE_DIR / f"stage0.{os.getpid()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def _update_streaming_request(self, req_id, new_req_data):
        payload = getattr(new_req_data, "additional_information", None)
        incoming = deserialize_additional_information(payload)
        if isinstance(incoming, dict) and incoming:
            model = getattr(self, "model", None)
            accumulated_keys = getattr(model, "streaming_accumulated_keys", set())
            cached = self.model_intermediate_buffer.get(req_id, {})
            merged = merge_streaming_additional_information(cached, incoming, accumulated_keys)
            self.model_intermediate_buffer[req_id] = merged
            setattr(self.requests[req_id], "additional_information_cpu", merged)

        return super()._update_streaming_request(req_id, new_req_data)


class EasyMagpieGPUARWorker(GPUARWorker):
    """GPU AR worker that constructs :class:`EasyMagpieGPUARModelRunner`."""

    def init_device(self):
        # GPUARWorker hardcodes its module-level GPUARModelRunner symbol rather
        # than exposing a runner-class hook. Swap it only while the base method
        # constructs this worker's runner; each worker lives in its own process.
        original_runner_cls = gpu_ar_worker.GPUARModelRunner
        gpu_ar_worker.GPUARModelRunner = EasyMagpieGPUARModelRunner
        try:
            return super().init_device()
        finally:
            gpu_ar_worker.GPUARModelRunner = original_runner_cls


def batch_waveforms_to_cpu(outputs: Any) -> Any:
    """Copy a per-request waveform list to CPU in one transfer."""
    if not isinstance(outputs, list) or len(outputs) < 2 or not all(isinstance(x, torch.Tensor) for x in outputs):
        return outputs
    if len({(x.device, x.dtype) for x in outputs}) != 1:
        return outputs

    sizes = [x.numel() for x in outputs]
    shapes = [x.shape for x in outputs]
    packed = torch.cat([x.detach().reshape(-1) for x in outputs]).to("cpu").contiguous()
    return [part.view(shape) for part, shape in zip(packed.split(sizes), shapes, strict=True)]


class EasyMagpieCodecGPUGenerationModelRunner(GPUGenerationModelRunner):
    """Batch Stage-1 waveform D2H before upstream output handling."""

    def sample_tokens(self, grammar_output=None):
        state = self.execute_model_state
        if state is not None and isinstance(state.multimodal_outputs, Mapping):
            multimodal_outputs = dict(state.multimodal_outputs)
            outputs = multimodal_outputs.get("model_outputs")
            batched = batch_waveforms_to_cpu(outputs)
            if batched is not outputs:
                multimodal_outputs["model_outputs"] = batched
                self.execute_model_state = state._replace(multimodal_outputs=multimodal_outputs)
        return super().sample_tokens(grammar_output)


class EasyMagpieCodecGPUGenerationWorker(GPUGenerationWorker):
    """Construct the codec runner through the upstream worker."""

    def init_device(self):
        original_runner_cls = gpu_generation_worker.GPUGenerationModelRunner
        gpu_generation_worker.GPUGenerationModelRunner = EasyMagpieCodecGPUGenerationModelRunner
        try:
            return super().init_device()
        finally:
            gpu_generation_worker.GPUGenerationModelRunner = original_runner_cls
