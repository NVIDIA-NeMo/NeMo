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

"""Synchronous per-stream bridge onto the vLLM-Omni engines.

The S2S wrapper drives inference from a synchronous PyTorch loop (perception
-> per-frame text + ASR -> audio codec decode), while ``AsyncOmni.generate``
is an ``async for``. Spawning an event loop per chunk would re-pay the
request-init cost every 80 ms and break vllm-omni's session semantics, so a
:class:`OmniStreamingSession` runs consumer tasks on the runtime's shared loop
and hands frames back across two synchronous queues.

Per-step protocol:

1. **Prefill** -- Nemotron receives the system prompt, EarTTS the speaker
   latent. With CFG enabled EarTTS receives a conditional and an unconditional
   prefill, each with its own KV cache. Nemotron's first token ``t_0`` is fed
   back internally rather than exposed.
2. **Decode step k** (``k >= 1``) -- ``prompt_token_ids = [t_{k-1}]``,
   ``additional_information.acoustic_embedding = ac_emb[k-1]``, producing
   ``t_k`` plus whichever auxiliary channel the checkpoint carries.

The components are stepped independently: :meth:`OmniStreamingSession.step_llm`
returns Nemotron's tokens and :meth:`OmniStreamingSession.step_tts` submits a
text token, so the caller can rewrite that token (forced turn-taking) in
between and both TTS backends see the same value.

The synchronous side is single-threaded: only one step may be in flight at a
time. :meth:`OmniStreamingSession.finish` closes the request cleanly and
:meth:`OmniStreamingSession.abort` drops it.
"""

import asyncio
import threading
import time
import uuid
from queue import Queue
from typing import Any

import torch

from nemo.collections.speechlm2.inference.vllm_omni.outputs import (
    StepTokens,
    _audio_codes,
    _multimodal_output,
    _step_delta,
    _step_tokens,
)
from nemo.collections.speechlm2.inference.vllm_omni.runtime import OmniRuntime
from nemo.collections.speechlm2.parts.logit_boosts import LogitBoosts
from nemo.utils import logging


class _Sentinel:
    """Marker placed on the sync output queues to signal end-of-stream or an error."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException | None = None):
        self.exc = exc


_END_OF_STREAM = _Sentinel()


class OmniStreamingSession:
    """One split Nemotron/EarTTS streaming request.

    The two components are driven independently: :meth:`step_llm` submits an
    acoustic frame and returns Nemotron's tokens, :meth:`step_tts` submits a
    text token and produces one acoustic frame. A session owning both exposes
    both, which lets the caller act on the text token in between rather than
    having EarTTS consume it inside the session. With CFG enabled, conditional
    and unconditional EarTTS requests run in the same engine and the custom
    scheduler keeps their independent KV-cache streams in lockstep.
    """

    def __init__(
        self,
        runtime: OmniRuntime,
        request_id: str,
        system_prompt: str = "",
        speaker_latent: torch.Tensor | None = None,
        t_prefill: int = 0,
        *,
        sampling_params: dict | None = None,
        special_token_ids: set[int] | None = None,
        guidance_enabled: bool = True,
        guidance_scale: float = 0.5,
        step_timeout: float = 60.0,
        profile: bool = False,
        agent_logit_boosts: "LogitBoosts | None" = None,
        text_token_ids: dict[str, int] | None = None,
    ) -> None:
        self._has_llm = runtime.llm_engine is not None
        self._has_tts = runtime.tts_engine is not None
        if not self._has_llm and not self._has_tts:
            raise ValueError("OmniStreamingSession requires at least one vLLM component")
        if self._has_tts and (speaker_latent is None or speaker_latent.numel() == 0):
            raise ValueError("speaker_latent is required for OmniStreamingSession")
        if self._has_llm and t_prefill <= 0:
            raise ValueError(f"t_prefill must be > 0 (got {t_prefill})")

        self._runtime = runtime
        self.request_id = request_id
        self._llm_request_id = f"{request_id}:nemotron"
        self._sampling_history_key = f"{self._llm_request_id}:{uuid.uuid4().hex}"
        self._cfg_pair_id = f"{request_id}:eartts"
        self._tts_cond_request_id = f"{self._cfg_pair_id}:cond"
        self._tts_uncond_request_id = f"{self._cfg_pair_id}:uncond"
        self._step_timeout = step_timeout
        self._system_prompt = system_prompt
        self._speaker_latent = speaker_latent.detach().cpu().contiguous() if speaker_latent is not None else None
        self._t_prefill = int(t_prefill)
        self._sampling_overrides = dict(sampling_params or {})
        self._special_token_ids = tuple(sorted(special_token_ids or ()))
        self._guidance_enabled = bool(guidance_enabled)
        self._guidance_scale = float(guidance_scale)
        self._agent_logit_boosts = agent_logit_boosts or LogitBoosts()
        self._text_token_ids = dict(text_token_ids or {})
        if self._has_llm and self._agent_logit_boosts:
            missing = {"pad_id", "bos_id", "eos_id"} - set(self._text_token_ids)
            if missing:
                raise ValueError(f"agent_logit_boosts requires text_token_ids {sorted(missing)}")

        # Separate completion queues per component: a session may have both,
        # and step_llm()/step_tts() must not consume each other's items.
        self._text_out_q: "Queue[StepTokens | _Sentinel]" = Queue()
        self._tts_done_q: "Queue[StepTokens | _Sentinel]" = Queue()
        self._audio_buf_lock = threading.Lock()
        self._audio_buf: list[torch.Tensor] = []
        self._closed = False
        self._error: BaseException | None = None
        self._loop = runtime._loop
        self._queues_ready = threading.Event()

        self._input_q: asyncio.Queue | None = None
        self._llm_internal_q: asyncio.Queue | None = None
        self._tts_cond_input_q: asyncio.Queue | None = None
        self._tts_uncond_input_q: asyncio.Queue | None = None
        self._pending_tokens_q: asyncio.Queue | None = None
        self._uncond_audio_q: asyncio.Queue | None = None

        self._prof: dict[str, list[float]] | None = {} if profile else None
        self._t_put = 0.0
        self._t_yield = 0.0
        self._t_llm = 0.0
        self._t_done = 0.0

        self._consumer_future = runtime.submit(self._run_consumer())
        if not self._queues_ready.wait(timeout=30):
            self._consumer_future.cancel()
            raise TimeoutError("Timed out creating split vLLM-Omni session queues")

    def _rec(self, name: str, dt_s: float) -> None:
        if self._prof is not None:
            self._prof.setdefault(name, []).append(dt_s * 1000.0)

    def _rec_ms(self, name: str, dt_ms: float) -> None:
        if self._prof is not None:
            self._prof.setdefault(name, []).append(float(dt_ms))

    def log_timing_summary(self) -> None:
        if not self._prof:
            return
        parts = []
        for name, values in self._prof.items():
            mean = sum(values) / len(values)
            parts.append(
                f"{name}: mean={mean:.1f}ms min={min(values):.1f}ms " f"max={max(values):.1f}ms n={len(values)}"
            )
        logging.info(f"OmniStreamingSession {self.request_id} per-frame timing:\n  " + "\n  ".join(parts))

    def _cfg_payload(self, role: str) -> dict[str, Any]:
        return {
            "cfg_enabled": self._guidance_enabled,
            "cfg_role": role,
            "cfg_pair_id": self._cfg_pair_id,
            "cfg_scale": self._guidance_scale,
        }

    def _record_stage_metrics(self, prefix: str, stage_output: Any) -> None:
        if self._prof is None:
            return
        for key, value in (getattr(stage_output, "stage_durations", None) or {}).items():
            self._rec_ms(f"{prefix}.{key}", value)

    async def _run_consumer(self) -> None:
        tasks: list[asyncio.Task] = []
        try:
            self._input_q = asyncio.Queue()
            self._llm_internal_q = asyncio.Queue()
            self._tts_cond_input_q = asyncio.Queue()
            self._tts_uncond_input_q = asyncio.Queue()
            self._pending_tokens_q = asyncio.Queue()
            self._uncond_audio_q = asyncio.Queue()
            self._queues_ready.set()

            if self._has_llm:
                tasks.append(asyncio.create_task(self._consume_llm()))
            if self._has_tts:
                tasks.append(asyncio.create_task(self._consume_tts("cond", self._tts_cond_input_q)))
            if self._has_tts and self._guidance_enabled:
                tasks.append(asyncio.create_task(self._consume_tts("uncond", self._tts_uncond_input_q)))
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._error = exc
            # Both queues, so a caller blocked in either step never hangs.
            self._text_out_q.put(_Sentinel(exc))
            self._tts_done_q.put(_Sentinel(exc))
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._queues_ready.set()
            self._text_out_q.put(_END_OF_STREAM)
            self._tts_done_q.put(_END_OF_STREAM)
            self.log_timing_summary()

    async def _consume_llm(self) -> None:
        from vllm import SamplingParams
        from vllm.engine.protocol import StreamingInput
        from vllm.sampling_params import RequestOutputKind

        from nemo.collections.speechlm2.inference.vllm_omni.nemotron_duplex_h.sampling import SHARED_TEXT_SAMPLING_ARG

        shared_sampling = {
            "temperature": float(self._sampling_overrides.get("temperature", 1.0)),
            "top_p": float(self._sampling_overrides.get("top_p", 1.0)),
            "repetition_penalty": float(self._sampling_overrides.get("repetition_penalty", 1.0)),
            "special_token_ids": list(self._special_token_ids),
            # The first vLLM output is the internal prefill token t_0. The
            # repetition history starts with the first client-visible frame.
            "history_skip": 1,
            "history_key": self._sampling_history_key,
            # Agent-channel boosts, applied before sampling exactly as
            # DuplexSTTModel does. The user-channel ones cannot travel this way
            # because the ASR head's logits never reach vLLM's sampler; the
            # model applies those itself.
            "boosts": self._agent_logit_boosts.as_dict(),
            **self._text_token_ids,
        }
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_tokens=1,
            detokenize=False,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
            extra_args={SHARED_TEXT_SAMPLING_ARG: shared_sampling},
        )

        async def inputs():
            yield StreamingInput(
                prompt={
                    "prompt_token_ids": [0] * self._t_prefill,
                    "additional_information": {
                        "system_prompt": self._system_prompt,
                    },
                },
                sampling_params=params,
            )
            while True:
                submission = await self._input_q.get()
                if submission is None:
                    return
                acoustic, committed_text = submission
                # Always drain the internal queue to stay one output per input,
                # then let the caller's committed token win. That is how the
                # caller's forced-turn-taking rewrite reaches Nemotron's own
                # history, matching what native feedback does through gen_text.
                prev_tokens = await self._llm_internal_q.get()
                if committed_text is not None:
                    prev_tokens = prev_tokens._replace(text=int(committed_text))
                additional_information: dict[str, Any] = {
                    "system_prompt": None,
                    "acoustic_embedding": acoustic,
                }
                if prev_tokens.asr is not None:
                    additional_information["input_asr_ids"] = torch.tensor([prev_tokens.asr], dtype=torch.long)
                if prev_tokens.function is not None:
                    additional_information["input_function_ids"] = torch.tensor(
                        [prev_tokens.function], dtype=torch.long
                    )
                self._t_yield = time.perf_counter()
                yield StreamingInput(
                    prompt={
                        "prompt_token_ids": [int(prev_tokens.text)],
                        "additional_information": additional_information,
                    },
                    sampling_params=params,
                )

        output_count = 0
        try:
            async for stage_output in self._runtime.llm_engine.generate(
                inputs(),
                sampling_params_list=[params],
                request_id=self._llm_request_id,
            ):
                now = time.perf_counter()
                self._record_stage_metrics("llm", stage_output)
                current_tokens = _step_tokens(stage_output)
                await self._llm_internal_q.put(current_tokens)

                output_count += 1
                if output_count <= 1:
                    continue

                self._rec("pull", self._t_yield - self._t_put)
                self._rec("llm_engine", now - self._t_yield)
                self._t_llm = now
                # The token is returned to the caller, never forwarded to
                # EarTTS from here. The caller owns what happens in between
                # (forced turn-taking rewrites the text token) and submits it
                # with step_tts, so both TTS backends see the same token.
                self._t_done = now
                self._text_out_q.put(current_tokens)
        finally:
            # Safety net rather than the normal path: finish() closes the TTS
            # inputs itself. This covers Nemotron ending first, which would
            # otherwise leave the EarTTS consumer waiting and stall the
            # gather() in _run_consumer.
            if self._has_tts:
                await self._tts_cond_input_q.put(None)
                if self._guidance_enabled:
                    await self._tts_uncond_input_q.put(None)

    async def _consume_tts(self, role: str, input_q: asyncio.Queue) -> None:
        from vllm import SamplingParams
        from vllm.engine.protocol import StreamingInput
        from vllm.sampling_params import RequestOutputKind

        extra_args = self._cfg_payload(role) if self._guidance_enabled else {}
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            detokenize=False,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
            extra_args=extra_args,
        )
        request_id = self._tts_cond_request_id if role == "cond" else self._tts_uncond_request_id

        async def inputs():
            prefill_info = {
                "embed": {"voice": self._speaker_latent.clone()},
                **self._cfg_payload(role),
            }
            if not self._guidance_enabled:
                prefill_info["cfg_enabled"] = False
            yield StreamingInput(
                prompt={
                    "prompt_token_ids": [0] * int(self._speaker_latent.shape[0]),
                    "additional_information": prefill_info,
                },
                sampling_params=params,
            )
            while True:
                text_tok = await input_q.get()
                if text_tok is None:
                    return
                yield StreamingInput(
                    prompt={
                        "prompt_token_ids": [0],
                        "additional_information": {
                            "ids": {"output": [int(text_tok)]},
                            **self._cfg_payload(role),
                        },
                    },
                    sampling_params=params,
                )

        output_count = 0
        async for stage_output in self._runtime.tts_engine.generate(
            inputs(),
            sampling_params_list=[params],
            request_id=request_id,
        ):
            now = time.perf_counter()
            req_out = stage_output.request_output
            mm = _multimodal_output(stage_output, req_out)
            finished = bool(getattr(req_out, "finished", False))
            self._record_stage_metrics(f"tts_{role}", stage_output)
            output_count += 1
            if output_count <= 1:
                continue
            audio = _step_delta(_audio_codes(mm), finished, skip_finished=False)
            if audio is None or audio.ndim != 2 or audio.shape[0] < 1:
                continue
            # EarTTS emits exactly one acoustic frame per streaming update, so
            # keep only the newest row: a prefill update covers the whole
            # speaker latent, and a non-drained key would arrive cumulative.
            audio = audio[-1:]
            audio = audio.detach().cpu().to(torch.long)
            if role == "uncond":
                await self._uncond_audio_q.put(audio)
                continue

            # A final empty update can race after all text tokens have been
            # consumed. It has no corresponding wrapper step and must not
            # synthesize another frame.
            if self._pending_tokens_q.empty():
                continue
            tokens = await self._pending_tokens_q.get()
            if self._guidance_enabled:
                uncond_audio = await self._uncond_audio_q.get()
                if not torch.equal(audio, uncond_audio):
                    raise RuntimeError(
                        "EarTTS CFG pair produced divergent client-visible "
                        "codes: "
                        f"cond_shape={tuple(audio.shape)} "
                        f"uncond_shape={tuple(uncond_audio.shape)} "
                        f"cond={audio.tolist()} "
                        f"uncond={uncond_audio.tolist()}"
                    )
            with self._audio_buf_lock:
                self._audio_buf.append(audio)
            if self._has_llm:
                self._rec("tts_after_llm", now - self._t_llm)
            else:
                self._rec("tts_engine", now - self._t_put)
            self._t_done = now
            self._tts_done_q.put(tokens)

    def step_llm(
        self,
        acoustic_embedding: torch.Tensor,
        *,
        prev_text_token: int | None = None,
    ) -> StepTokens:
        """Submit one acoustic frame to Nemotron and return its tokens.

        Returns as soon as Nemotron has produced the frame's tokens. EarTTS is
        not driven from here even when this session owns both components: the
        caller submits the (possibly rewritten) text token with
        :meth:`step_tts`.

        Args:
            acoustic_embedding: This frame's encoded audio.
            prev_text_token: Text token to feed back as the previous step's
                output, letting a caller that rewrote it (forced turn-taking)
                keep Nemotron's history consistent with its own. ``None``
                keeps whatever Nemotron last produced, which is what the first
                frame after prefill needs since its predecessor is the
                engine-internal prefill token.
        """
        if not self._has_llm:
            raise RuntimeError("This vLLM-Omni session has no Nemotron component")
        if self._closed:
            raise RuntimeError(f"OmniStreamingSession {self.request_id} is closed")
        ac_emb = acoustic_embedding.detach().cpu().contiguous()
        if ac_emb.dim() == 1:
            ac_emb = ac_emb.unsqueeze(0)
        elif ac_emb.dim() == 3:
            ac_emb = ac_emb.reshape(-1, ac_emb.shape[-1])
        if ac_emb.dim() != 2:
            raise ValueError(
                "acoustic_embedding must be shapeable to 2D [n, hidden], " f"got {tuple(acoustic_embedding.shape)}"
            )
        ac_emb = ac_emb.to(torch.float32)

        self._t_put = time.perf_counter()
        asyncio.run_coroutine_threadsafe(
            self._input_q.put((ac_emb, prev_text_token)),
            self._loop,
        ).result()
        self._rec("put", time.perf_counter() - self._t_put)

        item = self._text_out_q.get(timeout=self._step_timeout)
        returned = time.perf_counter()
        if not isinstance(item, _Sentinel):
            self._rec("deliver", returned - self._t_done)
            self._rec("step_total", returned - self._t_put)
        if isinstance(item, _Sentinel):
            if item.exc is not None:
                raise RuntimeError(f"OmniStreamingSession {self.request_id} consumer raised") from item.exc
            raise RuntimeError(f"OmniStreamingSession {self.request_id} ended before producing a token")
        return item

    def step_tts(self, text_token: int) -> None:
        """Submit one text token to EarTTS and wait for its acoustic frame.

        Valid whether or not this session also owns Nemotron; the resulting
        codes are collected by :meth:`drain_audio_codes`.
        """
        if not self._has_tts:
            raise RuntimeError("This vLLM-Omni session has no EarTTS component")
        if self._closed:
            raise RuntimeError(f"OmniStreamingSession {self.request_id} is closed")

        tokens = StepTokens(int(text_token))
        self._t_put = time.perf_counter()

        async def _put_tts_input() -> None:
            await self._pending_tokens_q.put(tokens)
            await self._tts_cond_input_q.put(tokens.text)
            if self._guidance_enabled:
                await self._tts_uncond_input_q.put(tokens.text)

        asyncio.run_coroutine_threadsafe(_put_tts_input(), self._loop).result()
        item = self._tts_done_q.get(timeout=self._step_timeout)
        if isinstance(item, _Sentinel):
            if item.exc is not None:
                raise RuntimeError(f"OmniStreamingSession {self.request_id} consumer raised") from item.exc
            raise RuntimeError(f"OmniStreamingSession {self.request_id} ended before producing audio")

    def drain_audio_codes(self) -> list[torch.Tensor]:
        with self._audio_buf_lock:
            out = self._audio_buf
            self._audio_buf = []
        return out

    def _abort_engine_requests(self) -> None:
        requests = (
            (self._runtime.llm_engine, self._llm_request_id),
            (self._runtime.tts_engine, self._tts_cond_request_id),
            (self._runtime.tts_engine, self._tts_uncond_request_id),
        )
        for engine, request_id in requests:
            if engine is None:
                continue
            if request_id == self._tts_uncond_request_id and not self._guidance_enabled:
                continue
            try:
                abort_result = engine.abort(request_id)
                if asyncio.iscoroutine(abort_result):
                    asyncio.run_coroutine_threadsafe(abort_result, self._loop).result(timeout=5)
            except Exception as exc:
                logging.debug(f"AsyncOmni.abort({request_id}) raised: {exc!r}")

    def finish(self, *, drain_remaining_audio_s: float = 0.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:

            async def _close_inputs() -> None:
                # Close every input this session owns. A session with both
                # components drives them independently, so closing only one
                # would leave the other consumer waiting forever.
                if self._has_llm:
                    await self._input_q.put(None)
                if self._has_tts:
                    await self._tts_cond_input_q.put(None)
                    if self._guidance_enabled:
                        await self._tts_uncond_input_q.put(None)

            asyncio.run_coroutine_threadsafe(_close_inputs(), self._loop).result(timeout=5)
        except Exception:
            pass
        try:
            self._consumer_future.result(timeout=max(drain_remaining_audio_s, 1.0))
        except Exception as exc:
            logging.debug(f"OmniStreamingSession {self.request_id} consumer ended with: {exc!r}")
            self._abort_engine_requests()
            self._consumer_future.cancel()
            try:
                self._consumer_future.result(timeout=5)
            except Exception:
                pass
        for queue in (self._text_out_q, self._tts_done_q):
            try:
                while True:
                    queue.get_nowait()
            except Exception:
                pass

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abort_engine_requests()
        self._consumer_future.cancel()
