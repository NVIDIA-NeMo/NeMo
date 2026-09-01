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

"""Pair-aware synchronous vLLM-Omni scheduler for EarTTS CFG.

EarTTS classifier-free guidance is represented by two ordinary top-level
requests.  Their ``SamplingParams.extra_args`` must contain::

    {
        "cfg_enabled": True,
        "cfg_role": "cond" | "uncond",
        "cfg_pair_id": "<shared id>",
        "cfg_scale": 0.5,
    }

The model runner is responsible for blending the pair's model outputs.  This
scheduler supplies the ordering and lock-step contract required by that
operation.  It deliberately subclasses the synchronous Omni AR scheduler:
async placeholder scheduling can put the two members on different token
positions before either result reaches the scheduler.
"""

from __future__ import annotations

import math
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

logger = init_logger(__name__)

_COND = "cond"
_UNCOND = "uncond"
_ROLES = (_COND, _UNCOND)


def _extra_args(request: Any) -> dict[str, Any]:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    return extra_args if isinstance(extra_args, dict) else {}


def _normalized_sampled_tokens(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return (value,)
    return tuple(int(token_id) for token_id in value)


class EarTTSCFGScheduler(OmniARScheduler):
    """Synchronous Omni AR scheduler that keeps EarTTS CFG pairs lock-step."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cfg_pairs: dict[str, dict[str, str]] = {}
        self._cfg_req_to_pair: dict[str, str] = {}
        self._cfg_pair_scales: dict[str, float] = {}

        max_num_seqs = int(getattr(self.scheduler_config, "max_num_seqs", 0) or 0)
        if max_num_seqs and max_num_seqs < 2:
            raise ValueError("EarTTSCFGScheduler requires max_num_seqs >= 2")

    @staticmethod
    def _cfg_metadata(request: Request) -> tuple[str, str, float] | None:
        extra_args = _extra_args(request)
        if not bool(extra_args.get("cfg_enabled", False)):
            return None

        role = extra_args.get("cfg_role")
        if role not in _ROLES:
            raise ValueError(f"CFG request {request.request_id!r}: cfg_role must be 'cond' or 'uncond', got {role!r}")

        raw_pair_id = extra_args.get("cfg_pair_id")
        if raw_pair_id is None or not str(raw_pair_id):
            raise ValueError(f"CFG request {request.request_id!r}: cfg_pair_id must be non-empty")
        pair_id = str(raw_pair_id)

        raw_scale = extra_args.get("cfg_scale")
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, int | float):
            raise ValueError(f"CFG request {request.request_id!r}: cfg_scale must be a finite non-negative number")
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"CFG request {request.request_id!r}: cfg_scale must be a finite non-negative number")
        return pair_id, role, scale

    def add_request(self, request: Request) -> None:
        metadata = self._cfg_metadata(request)
        if metadata is not None:
            pair_id, role, scale = metadata
            roles = self._cfg_pairs.get(pair_id, {})
            existing = roles.get(role)
            if existing is not None and existing != request.request_id:
                raise ValueError(f"CFG pair {pair_id!r} already has {role} request {existing!r}")
            existing_scale = self._cfg_pair_scales.get(pair_id)
            if existing_scale is not None and existing_scale != scale:
                raise ValueError(
                    f"CFG pair {pair_id!r} has inconsistent cfg_scale values: {existing_scale} and {scale}"
                )

        super().add_request(request)

        if metadata is not None:
            pair_id, role, scale = metadata
            self._cfg_pairs.setdefault(pair_id, {})[role] = request.request_id
            self._cfg_req_to_pair[request.request_id] = pair_id
            self._cfg_pair_scales[pair_id] = scale
            # Conditional and null prompts need deterministic, identical
            # progress; asymmetric prefix-cache hits violate that contract.
            if hasattr(request, "skip_reading_prefix_cache"):
                request.skip_reading_prefix_cache = True

    def _pair_requests(self, pair_id: str) -> tuple[Request | None, Request | None]:
        roles = self._cfg_pairs.get(pair_id, {})
        return self.requests.get(roles.get(_COND, "")), self.requests.get(roles.get(_UNCOND, ""))

    def _drop_pair(self, pair_id: str) -> None:
        for request_id in self._cfg_pairs.pop(pair_id, {}).values():
            self._cfg_req_to_pair.pop(request_id, None)
        self._cfg_pair_scales.pop(pair_id, None)

    @staticmethod
    def _remove_from_queue(queue: Any, requests: list[Request]) -> None:
        if not requests:
            return
        if hasattr(queue, "remove_requests"):
            queue.remove_requests(requests)
            return
        for request in requests:
            queue.remove(request)

    @staticmethod
    def _prepend_to_queue(queue: Any, requests: list[Request]) -> None:
        if not requests:
            return
        if hasattr(queue, "prepend_requests"):
            queue.prepend_requests(requests)
            return
        if hasattr(queue, "prepend_request"):
            for request in reversed(requests):
                queue.prepend_request(request)
            return
        for request in reversed(requests):
            queue.insert(0, request)

    @classmethod
    def _replace_queue(cls, queue: Any, requests: list[Request]) -> None:
        current = list(queue)
        cls._remove_from_queue(queue, current)
        for request in requests:
            queue.add_request(request) if hasattr(queue, "add_request") else queue.append(request)

    def _available_sequence_slots(self) -> int:
        capacity = getattr(self, "max_num_running_reqs", None)
        if capacity is None:
            capacity = getattr(self.scheduler_config, "max_num_seqs", 0)
        return max(0, int(capacity or 0) - len(self.running))

    def _prepare_cfg_waiting(self) -> list[tuple[Any, list[Request]]]:
        """Hide unsafe pairs and put admissible pairs first and adjacent."""
        skipped_queue = getattr(self, "skipped_waiting", None)
        promoted_held: list[Request] = []

        # Streaming updates are applied by vLLM while traversing
        # ``skipped_waiting``. The first traversal promotes each pair member
        # from WAITING_FOR_STREAMING_REQ to WAITING but deliberately skips
        # scheduling it (see _try_promote_blocked_waiting_request below).
        # Once both members are promoted, move them to the ordinary waiting
        # queue together so the admission logic below sees one atomic pair.
        if skipped_queue is not None:
            skipped_by_id = {
                request.request_id: request
                for request in list(skipped_queue)
            }
            for pair_id, roles in self._cfg_pairs.items():
                if set(roles) != set(_ROLES):
                    continue
                cond = skipped_by_id.get(roles[_COND])
                uncond = skipped_by_id.get(roles[_UNCOND])
                if cond is None or uncond is None:
                    continue
                ready = [
                    request
                    for request in (cond, uncond)
                    if request.status == RequestStatus.WAITING
                ]
                if len(ready) == 2:
                    self._remove_from_queue(
                        skipped_queue, [cond, uncond]
                    )
                    self.waiting.add_request(cond)
                    self.waiting.add_request(uncond)
                elif len(ready) == 1:
                    self._remove_from_queue(skipped_queue, ready)
                    promoted_held.extend(ready)

        waiting_items = list(self.waiting)
        skipped_items = list(skipped_queue) if skipped_queue is not None else []
        waiting_ids = {request.request_id for request in waiting_items}
        skipped_ids = {request.request_id for request in skipped_items}
        running_ids = {request.request_id for request in self.running}

        held: list[tuple[Any, list[Request]]] = []
        if promoted_held:
            held.append((skipped_queue, promoted_held))
        hold_waiting: set[str] = set()
        complete_waiting_pairs: list[str] = []

        for pair_id, roles in self._cfg_pairs.items():
            pair_ids = {request_id for request_id in roles.values()}
            is_complete = set(roles) == set(_ROLES) and all(request_id in self.requests for request_id in pair_ids)
            in_waiting = pair_ids & waiting_ids
            in_skipped = pair_ids & skipped_ids
            in_running = pair_ids & running_ids

            if in_running and (in_waiting or in_skipped):
                raise RuntimeError(f"EarTTS CFG pair {pair_id!r} was split across running and waiting queues")
            if len(in_running) == 1:
                raise RuntimeError(f"EarTTS CFG pair {pair_id!r} has only one running member")

            if not is_complete or len(in_waiting) == 1:
                hold_waiting.update(in_waiting)
            elif len(in_waiting) == 2:
                complete_waiting_pairs.append(pair_id)

        # A pair consumes two sequence slots.  Expose only whole pairs to the
        # upstream scheduler and put them before ordinary requests so another
        # admission cannot consume the second slot between pair members.
        admitted_pair_count = self._available_sequence_slots() // 2
        allowed_pairs = set(complete_waiting_pairs[:admitted_pair_count])
        for pair_id in complete_waiting_pairs[admitted_pair_count:]:
            hold_waiting.update(self._cfg_pairs[pair_id].values())

        held_waiting = [request for request in waiting_items if request.request_id in hold_waiting]
        self._remove_from_queue(self.waiting, held_waiting)
        if held_waiting:
            held.append((self.waiting, held_waiting))

        remaining = list(self.waiting)
        ordinary = [request for request in remaining if request.request_id not in self._cfg_req_to_pair]
        ordered: list[Request] = []
        for pair_id in complete_waiting_pairs:
            if pair_id not in allowed_pairs:
                continue
            cond, uncond = self._pair_requests(pair_id)
            if cond is not None and uncond is not None:
                ordered.extend((cond, uncond))
        ordered.extend(ordinary)
        if ordered != remaining:
            self._replace_queue(self.waiting, ordered)

        actual_ids = [request.request_id for request in self.waiting]
        for pair_id in allowed_pairs:
            roles = self._cfg_pairs[pair_id]
            cond_id, uncond_id = roles[_COND], roles[_UNCOND]
            try:
                cond_index = actual_ids.index(cond_id)
            except ValueError:
                continue
            if cond_index + 1 >= len(actual_ids) or actual_ids[cond_index + 1] != uncond_id:
                raise RuntimeError(
                    "EarTTSCFGScheduler requires an FCFS-compatible waiting "
                    f"queue; CFG pair {pair_id!r} could not be made adjacent"
                )

        return held

    def _try_promote_blocked_waiting_request(
        self, request: Request
    ) -> bool:
        promoted = super()._try_promote_blocked_waiting_request(
            request
        )
        if promoted and request.request_id in self._cfg_req_to_pair:
            # The base scheduler would immediately schedule this first
            # promoted member. Return False once so it is parked back in
            # skipped_waiting; after its peer is promoted,
            # _prepare_cfg_waiting moves both to waiting atomically.
            return False
        return promoted

    def _should_defer_waiting_admission(self) -> bool:
        """Install the pair guard after Omni has processed pending inputs.

        ``OmniARScheduler.schedule`` invokes this hook immediately before the
        stock vLLM scheduler.  Preparing here is important: doing it at the
        start of this class's :meth:`schedule` would hide requests from Omni's
        chunk/input processing and could leave a streaming pair parked in
        ``skipped_waiting`` forever.
        """
        self._cfg_decode_ready_before = {
            request.request_id
            for request in self.running
            if self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens
        }
        self._cfg_held_for_schedule = self._prepare_cfg_waiting()
        self._cfg_waiting_before_schedule = {request.request_id for request in self.waiting}
        return super()._should_defer_waiting_admission()

    def _restore_held(self, held: list[tuple[Any, list[Request]]]) -> None:
        for queue, requests in reversed(held):
            self._prepend_to_queue(queue, requests)

    def _equalize_pair_progress(self, scheduler_output: SchedulerOutput) -> None:
        scheduled = scheduler_output.num_scheduled_tokens
        for pair_id in self._cfg_pairs:
            cond, uncond = self._pair_requests(pair_id)
            if cond is None or uncond is None:
                continue
            cond_scheduled = int(scheduled.get(cond.request_id, 0) or 0)
            uncond_scheduled = int(scheduled.get(uncond.request_id, 0) or 0)
            if not cond_scheduled or not uncond_scheduled:
                continue

            target = min(cond.num_computed_tokens, uncond.num_computed_tokens)
            feasible = all(
                request.num_computed_tokens - target < count
                for request, count in ((cond, cond_scheduled), (uncond, uncond_scheduled))
            )
            if not feasible:
                continue

            for request, count in ((cond, cond_scheduled), (uncond, uncond_scheduled)):
                difference = request.num_computed_tokens - target
                if difference <= 0:
                    continue
                request.num_computed_tokens = target
                if hasattr(request, "num_in_flight_tokens"):
                    request.num_in_flight_tokens = max(0, request.num_in_flight_tokens - difference)
                scheduler_output.num_scheduled_tokens[request.request_id] = count - difference
                scheduler_output.total_num_scheduled_tokens -= difference

    def _assert_atomic_admission(self, scheduler_output: SchedulerOutput, waiting_before: set[str]) -> None:
        scheduled = scheduler_output.num_scheduled_tokens
        for pair_id, roles in self._cfg_pairs.items():
            pair_ids = {roles.get(_COND), roles.get(_UNCOND)}
            pair_ids.discard(None)
            if len(pair_ids & waiting_before) != 2:
                continue
            admitted = {request_id for request_id in pair_ids if scheduled.get(request_id, 0)}
            if admitted and admitted != pair_ids:
                raise RuntimeError(
                    f"EarTTS CFG pair {pair_id!r} was not admitted atomically: scheduled={sorted(admitted)}"
                )

    def _assert_complete_decode_pairs(self, scheduler_output: SchedulerOutput, decode_ready_before: set[str]) -> None:
        scheduled = scheduler_output.num_scheduled_tokens
        for pair_id, roles in self._cfg_pairs.items():
            cond_id = roles.get(_COND)
            uncond_id = roles.get(_UNCOND)
            if cond_id is None or uncond_id is None:
                continue
            cond_count = int(scheduled.get(cond_id, 0) or 0)
            uncond_count = int(scheduled.get(uncond_id, 0) or 0)
            if not cond_count and not uncond_count:
                continue
            if cond_count != uncond_count:
                raise RuntimeError(
                    f"EarTTS scheduler split CFG decode pair {pair_id!r}: "
                    f"{cond_id}={cond_count}, {uncond_id}={uncond_count}"
                )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._cfg_held_for_schedule: list[tuple[Any, list[Request]]] = []
        self._cfg_waiting_before_schedule: set[str] = set()
        self._cfg_decode_ready_before: set[str] = set()

        scheduler_config = getattr(self, "scheduler_config", None)
        original_threshold = getattr(scheduler_config, "long_prefill_token_threshold", None)
        if self._cfg_pairs and original_threshold is not None:
            budget = int(getattr(self, "max_num_scheduled_tokens", 0) or 0)
            if budget:
                scheduler_config.long_prefill_token_threshold = max(1, budget // 2)
        try:
            scheduler_output = super().schedule(throttle_prefills)
        finally:
            if original_threshold is not None:
                scheduler_config.long_prefill_token_threshold = original_threshold
            self._restore_held(self._cfg_held_for_schedule)

        self._equalize_pair_progress(scheduler_output)
        self._assert_atomic_admission(scheduler_output, self._cfg_waiting_before_schedule)
        self._assert_complete_decode_pairs(scheduler_output, self._cfg_decode_ready_before)
        return scheduler_output

    def _assert_matching_sampled_tokens(self, scheduler_output: SchedulerOutput, model_runner_output: Any) -> None:
        sampled = getattr(model_runner_output, "sampled_token_ids", None)
        req_id_to_index = getattr(model_runner_output, "req_id_to_index", None)
        if sampled is None or not isinstance(req_id_to_index, dict):
            return

        scheduled = scheduler_output.num_scheduled_tokens
        for pair_id, roles in self._cfg_pairs.items():
            cond_id = roles.get(_COND)
            uncond_id = roles.get(_UNCOND)
            if (
                cond_id not in scheduled
                or uncond_id not in scheduled
                or cond_id not in req_id_to_index
                or uncond_id not in req_id_to_index
            ):
                continue
            cond_tokens = _normalized_sampled_tokens(sampled[req_id_to_index[cond_id]])
            uncond_tokens = _normalized_sampled_tokens(sampled[req_id_to_index[uncond_id]])
            if cond_tokens != uncond_tokens:
                raise RuntimeError(
                    f"EarTTS CFG pair {pair_id!r} sampled-token mismatch: "
                    f"cond={cond_tokens}, uncond={uncond_tokens}"
                )

    def update_from_output(self, scheduler_output: SchedulerOutput, model_runner_output: Any) -> Any:
        self._assert_matching_sampled_tokens(scheduler_output, model_runner_output)
        # A length stop on a resumable StreamingInput segment is not terminal:
        # Omni parks the request until its next chunk. Do not mirror the
        # segment's internal free/park operation onto its CFG peer here.
        # Explicit abort/final teardown still flows through finish_requests,
        # which expands either member to the complete pair below.
        return super().update_from_output(
            scheduler_output, model_runner_output
        )

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        super()._update_request_as_session(session, update)
        # Stage 0 receives direct StreamingInput updates.  Upstream extends the
        # prompt but does not copy this per-chunk payload.  Downstream stages
        # are intentionally untouched; their connector owns payload delivery.
        if self.vllm_config.model_config.stage_id == 0:
            additional_information = getattr(update, "additional_information", None)
            if additional_information is not None:
                session.additional_information = additional_information


__all__ = ["EarTTSCFGScheduler"]
