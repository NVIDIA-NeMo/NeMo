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

"""Streaming scheduler for the one-stage Nemotron VoiceChat engine.

The only deviation from vLLM-Omni's async AR scheduler is forwarding each
direct ``StreamingInput`` chunk's ``additional_information`` payload onto the
session. This carries the current acoustic embedding into the model runner
without modifying the installed vLLM-Omni package.
"""

from __future__ import annotations

from vllm.v1.request import Request, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler


class NemotronVoicechatSchedulerMixin:
    """Forward direct per-chunk payloads on the one-stage session."""

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        super()._update_request_as_session(session, update)

        # Forward the chunk's payload onto the session, which is the courier
        # that carries it to ``OmniNewRequestData`` and from there into the
        # runner's ``model_intermediate_buffer``. Upstream propagates
        # ``model_intermediate_buffer`` itself but not ``additional_information``,
        # and this pipeline has to use the latter: the per-chunk payload is a
        # tensor (``acoustic_embedding``), and ``model_intermediate_buffer`` is
        # typed ``dict[str, Any]`` on the request, so vLLM's msgpack decoder has
        # no declared type to rebuild a tensor from and would hand the model a
        # ``[dtype, shape, bytes]`` list instead. ``additional_information`` is
        # the transport with an explicit tensor encoding, which is why upstream's
        # own duplex example keeps ``model_intermediate_buffer`` to plain lists.
        #
        # Replace rather than merge: this field is a per-chunk message, and
        # accumulating whole payloads across chunks would keep stale
        # prefill-only keys alive. ``None`` means "this chunk omitted the
        # field" rather than "clear the session", so placeholder chunks do not
        # drop the initial request's state. The runner does the actual merge
        # into the cached buffer, one sub-key at a time. Only stage 0 does this:
        # in a downstream stage the chunk transfer adapter is the sole writer of
        # the payload, so upstream returns early there.
        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info


class NemotronVoicechatARAsyncScheduler(NemotronVoicechatSchedulerMixin, OmniARAsyncScheduler):
    """Default: matches upstream's ``async_scheduling=True`` for LLM_AR stages."""


__all__ = [
    "NemotronVoicechatARAsyncScheduler",
    "NemotronVoicechatSchedulerMixin",
]
