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
"""Streaming scheduler that propagates EasyMagpie request metadata.

Configure it on a single-stage deployment with::

    "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"
"""
from __future__ import annotations

import threading
from importlib.metadata import version
from time import monotonic, sleep
from types import MethodType

import torch
from packaging.version import Version
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter

_UPSTREAM_HAS_SEGMENT_STOP_ACCOUNTING_FIX = Version(version("vllm-omni")) >= Version("0.26.0")


def _connector_extra(vllm_config) -> dict:
    connector = getattr(vllm_config.model_config, "stage_connector_config", {})
    if isinstance(connector, dict):
        return connector.get("extra", {}) or {}
    return getattr(connector, "extra", {}) or {}


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Forward each chunk's token limit and additional information.

    This class also works around a bug in vLLM-Omni's async segment-stop
    handling that deadlocks paced streaming sessions. On a resumable segment
    stop, ``OmniARScheduler.update_from_output`` does::

        request.async_tokens_to_discard = 1        # hardcoded
        request.num_output_placeholders = 0

    i.e. it assumes exactly one async token is in flight and, unlike omni's own
    *resume* path, it never rolls ``num_computed_tokens`` back for the tokens it
    is about to discard. Combined with vLLM 0.24's async accounting (a discarded
    token returns early from ``AsyncScheduler._update_request_with_output``
    without decrementing ``num_output_placeholders``) this leaves the re-admitted
    session in an unschedulable state:

    * a leaked placeholder (``placeholders>0`` with ``num_computed==num_tokens``)
      permanently trips the scheduler's async skip-optimisation, or
    * ``num_computed_tokens == num_tokens`` with ``placeholders==0`` yields
      ``num_new_tokens==0``.

    Either way the request is never scheduled again and paced clients hang.

    The fix mirrors omni's resume path: snapshot the *true* number of in-flight
    async tokens at the moment of the stop, then after ``update_from_output`` set
    ``async_tokens_to_discard`` to that count (0 when nothing is in flight, so no
    spurious discard) and roll ``num_computed_tokens`` back by the same amount.

    TODO(upstream): fix ``OmniARScheduler.update_from_output`` directly so the
    segment-stop branch uses ``async_tokens_to_discard = num_output_placeholders``
    and ``num_computed_tokens -= num_output_placeholders`` (matching the resume
    branch), then drop this override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        wait_ms = float(_connector_extra(self.vllm_config).get("stage0_admission_coalesce_ms", 0))
        if not 0 <= wait_ms <= 10:
            raise ValueError("stage0_admission_coalesce_ms must be in [0, 10]")
        self._admission_wait_s = wait_ms / 1000
        self._admission_deadline = None

    def _should_defer_waiting_admission(self) -> bool:
        if not self._admission_wait_s or not self.waiting or self.running:
            self._admission_deadline = None
            return False
        if any(request.num_computed_tokens > 0 for request in self.waiting):
            self._admission_deadline = None
            return False
        if len(self.waiting) >= self.max_num_running_reqs:
            self._admission_deadline = 0.0
            return False

        now = monotonic()
        if self._admission_deadline is None:
            self._admission_deadline = now + self._admission_wait_s
        return now < self._admission_deadline

    def _update_request_with_output(self, request: Request, new_token_ids):
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)
        if stopped:
            # After super() has decremented the placeholder for the stopping
            # token, ``num_output_placeholders`` is the number of *other* async
            # tokens still in flight for this request — the value omni's stop
            # handler should have used but overwrites with a hardcoded 1. Record
            # it so update_from_output can restore the correct accounting. Only
            # tracked while inside update_from_output, so there is no per-step
            # cost beyond the (rare) segment stops themselves.
            pending = getattr(self, "_emp_stopped_this_step", None)
            if pending is not None:
                pending.append((request, request.num_output_placeholders))
        return new_token_ids, stopped

    def update_from_output(self, scheduler_output, model_runner_output):
        if _UPSTREAM_HAS_SEGMENT_STOP_ACCOUNTING_FIX:
            return super().update_from_output(scheduler_output, model_runner_output)

        self._emp_stopped_this_step = []
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for request, snap in self._emp_stopped_this_step:
                # Only correct resumable stops where omni actually armed a discard.
                if getattr(request, "async_tokens_to_discard", 0) > 0:
                    request.async_tokens_to_discard = snap
                    if snap > 0:
                        request.num_computed_tokens -= snap
        finally:
            self._emp_stopped_this_step = None
        return outputs

    def _handle_stopped_request(self, request: Request) -> bool:
        # The input engine queues ``None`` after the final StreamingInput but
        # leaves the existing session's ``resumable`` flag set. Clear it before
        # the base handler consumes the sentinel so the chunk-transfer adapter
        # emits a true terminal payload and releases request-persistent codec
        # state. An empty queue still means "waiting for more websocket input".
        streaming_queue = getattr(request, "streaming_queue", None)
        if getattr(request, "resumable", False) and streaming_queue and streaming_queue[0] is None:
            request.resumable = False
        return super()._handle_stopped_request(request)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        super()._update_request_as_session(session, update)

        # Upstream hardcodes one discard on resume even when multiple async
        # outputs are outstanding. Its rollback is otherwise correct, so retain
        # it and replace only the discard count with the captured real value.
        if outstanding_async_tokens > 0 and getattr(session, "async_tokens_to_discard", 0) > 0:
            session.async_tokens_to_discard = outstanding_async_tokens

        new_max_tokens = getattr(update, "max_tokens", None)
        if new_max_tokens is not None:
            session.max_tokens = new_max_tokens

        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info

        # Defensive guard: if a resumed session has every token already computed
        # (``num_computed_tokens >= num_tokens``), the upstream scheduler computes
        # ``num_new_tokens == 0`` and trips ``assert num_new_tokens > 0``. Roll
        # back one token so there is always something to recompute and sample from
        # — the same "recompute the last token" corrective vLLM applies on a full
        # prompt cache hit (see Scheduler._update_waiting_for_remote_kv).
        if session.num_computed_tokens >= session.num_tokens:
            session.num_computed_tokens = session.num_tokens - 1


def _codec_payload_frames(info, num_quantizers: int) -> int:
    """Return the number of time-major acoustic rows in a connector payload."""
    codes = info.get("codes", {}) if isinstance(info, dict) else {}
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return 0
    if audio.ndim == 2:
        return int(audio.shape[0])
    if audio.ndim == 1 and audio.numel() % num_quantizers == 0:
        return int(audio.numel() // num_quantizers)
    raise ValueError(f"invalid native codec payload shape: {tuple(audio.shape)}")


def _without_consumed_codec_audio(info):
    """Copy request metadata without the previous chunk's audio codes."""
    if not isinstance(info, dict):
        return info
    codes = info.get("codes")
    if not isinstance(codes, dict) or "audio" not in codes:
        return info
    return {**info, "codes": {key: value for key, value in codes.items() if key != "audio"}}


def _poll_native_codec_chunk_unlocked(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Receive a chunk without resetting the vLLM state-cache position."""
    old_num_computed_tokens = request.num_computed_tokens
    old_additional_information = request.additional_information
    # Async-chunk prewarm may install one unscheduled placeholder before the
    # first real payload. Only tokens with materialized state are retained.
    old_prompt = list(request.prompt_token_ids or [])[:old_num_computed_tokens]
    old_all_token_ids = list(request._all_token_ids)[:old_num_computed_tokens]

    # vLLM-Omni merges incoming generation payloads into this object. Remove
    # consumed audio so a control-only boundary cannot inherit and replay it.
    poll_information = _without_consumed_codec_audio(old_additional_information)
    request.additional_information = poll_information
    received = OmniChunkTransferAdapter._poll_single_request(adapter, request)
    if received:
        frames = _codec_payload_frames(request.additional_information, adapter._easymagpie_num_quantizers)
        if frames > 0:
            placeholders = [0] * frames
            request.prompt_token_ids = old_prompt + placeholders
            request._all_token_ids[:] = old_all_token_ids + placeholders
            request.num_prompt_tokens = len(request.prompt_token_ids)
            request.num_computed_tokens = old_num_computed_tokens
            request.update_block_hashes()
            return True
        adapter._finished_load_reqs.discard(request.request_id)
    elif request.additional_information is poll_information:
        request.additional_information = old_additional_information

    request.prompt_token_ids = old_prompt
    request._all_token_ids[:] = old_all_token_ids
    request.num_prompt_tokens = len(old_prompt)
    request.num_computed_tokens = old_num_computed_tokens
    request.update_block_hashes()
    return False


def _poll_native_codec_chunk(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Publish connector readiness only after the request payload is coherent."""
    with adapter._easymagpie_chunk_ready:
        received = _poll_native_codec_chunk_unlocked(adapter, request)
        if received:
            adapter._easymagpie_chunk_ready.notify_all()
        return received


class EasyMagpieCodecScheduler(OmniGenerationScheduler):
    """Keep each Stage-1 stream on one append-only native vLLM request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adapter = self.chunk_transfer_adapter
        if adapter is None:
            raise ValueError("the native EasyMagpie codec requires async_chunk")
        config = self.vllm_config.model_config.hf_config
        num_quantizers = int(getattr(config, "num_stacked_codebooks", 0))
        if num_quantizers <= 0:
            raise ValueError("native EasyMagpie codec config has no stacked codebooks")
        wait_ms = float(_connector_extra(self.vllm_config).get("codec_startup_coalesce_ms", 0))
        if not 0 <= wait_ms <= 2:
            raise ValueError("codec_startup_coalesce_ms must be in [0, 2]")
        self._codec_startup_wait_s = wait_ms / 1000
        busy_wait_ms = float(_connector_extra(self.vllm_config).get("codec_busy_coalesce_ms", 0))
        if not 0 <= busy_wait_ms <= 4:
            raise ValueError("codec_busy_coalesce_ms must be in [0, 4]")
        self._codec_busy_wait_s = busy_wait_ms / 1000
        adapter._easymagpie_chunk_lock = threading.Lock()
        adapter._easymagpie_chunk_ready = threading.Condition(adapter._easymagpie_chunk_lock)
        adapter._easymagpie_num_quantizers = num_quantizers
        adapter._poll_single_request = MethodType(_poll_native_codec_chunk, adapter)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """Resume connector polling without resetting the stateful codec.

        Every incremental text update prewarms downstream stages again. vLLM
        turns the duplicate Stage-1 request into a streaming update, but the
        generation scheduler's default handler replaces its prompt and resets
        ``num_computed_tokens``. For the native codec the update carries no new
        codec input; it only signals that another upstream segment is coming.
        Keep the append-only prompt position and vLLM-managed codec state intact.
        """
        prompt_token_ids = session.prompt_token_ids
        all_token_ids = list(session._all_token_ids)
        num_prompt_tokens = session.num_prompt_tokens
        num_computed_tokens = session.num_computed_tokens
        additional_information = session.additional_information

        super()._update_request_as_session(session, update)

        session.prompt_token_ids = prompt_token_ids
        session._all_token_ids[:] = all_token_ids
        session.num_prompt_tokens = num_prompt_tokens
        session.num_computed_tokens = num_computed_tokens
        session.additional_information = additional_information
        session.update_block_hashes()

    def _handle_stopped_request(self, request: Request) -> bool:
        finished = super()._handle_stopped_request(request)
        stopped_sessions = getattr(self, "_easymagpie_stopped_sessions", None)
        if not finished and stopped_sessions is not None:
            stopped_sessions.append(request)
        return finished

    def _resume_codec_after_segment(self, session: Request) -> None:
        """Keep a resumable codec request on the worker's cached-request path."""
        waiting_for_input = session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        if session in self.waiting:
            self.waiting.remove_requests((session,))
        if session in self.skipped_waiting:
            self.skipped_waiting.remove_requests((session,))
        if waiting_for_input:
            self.num_waiting_for_streaming_input -= 1

        session.status = RequestStatus.RUNNING
        if session not in self.running:
            self.running.append(session)
        self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)

    def update_from_output(self, scheduler_output, model_runner_output):
        # A segment finish must reach the output processor, but Stage 1 must not
        # be re-admitted through the generation scheduler's ``scheduled_new``
        # path afterward. That path recreates the worker batch row, losing the
        # codec's recurrent cache even when ``num_computed_tokens`` is retained.
        # Move resumable segment stops back to ``running`` after the base method
        # has emitted the finish and removed them. Their next codec frames are
        # then scheduled as cached tokens against the same state pages.
        self._easymagpie_stopped_sessions = []
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for session in self._easymagpie_stopped_sessions:
                self._resume_codec_after_segment(session)
        finally:
            self._easymagpie_stopped_sessions = None
        return outputs

    def _should_coalesce_codec_startup(self) -> bool:
        if not self._codec_startup_wait_s or self.running:
            return False
        adapter = self.chunk_transfer_adapter
        with adapter._easymagpie_chunk_lock:
            return any(
                request.status == RequestStatus.WAITING_FOR_CHUNK
                and request.num_computed_tokens == 0
                and request.request_id in adapter._finished_load_reqs
                for request in self.waiting
            )

    def _ready_codec_requests(self):
        ready = self.chunk_transfer_adapter._finished_load_reqs
        return [request for request in (*self.running, *self.waiting) if request.request_id in ready]

    def _codec_busy_wait_done(self) -> bool:
        ready = self.chunk_transfer_adapter._finished_load_reqs
        return self._has_ready_codec_start() or all(request.request_id in ready for request in self.running)

    def _has_ready_codec_start(self) -> bool:
        return any(request.num_computed_tokens == 0 for request in self._ready_codec_requests())

    def _should_coalesce_codec_busy(self) -> bool:
        if not self._codec_busy_wait_s or len(self.running) < 2:
            return False
        ready = self._ready_codec_requests()
        return (
            any(request.num_computed_tokens > 0 for request in ready)
            and not self._codec_busy_wait_done()
        )

    def schedule(self, *args, **kwargs):
        adapter = self.chunk_transfer_adapter
        if self._should_coalesce_codec_startup():
            sleep(self._codec_startup_wait_s)
        with adapter._easymagpie_chunk_ready:
            if self._should_coalesce_codec_busy():
                adapter._easymagpie_chunk_ready.wait_for(
                    self._codec_busy_wait_done,
                    timeout=self._codec_busy_wait_s,
                )
            return super().schedule(*args, **kwargs)
