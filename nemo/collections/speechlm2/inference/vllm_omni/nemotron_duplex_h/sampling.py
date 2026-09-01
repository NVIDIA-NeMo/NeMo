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

"""VoiceChat text sampling, shared with the PyTorch backend.

Both backends decode the text head with
:func:`~nemo.collections.speechlm2.inference.model_wrappers.text_sampling.sample_text_token`.
The PyTorch backend calls it directly; vLLM reaches it through the
logits-processor hook implemented here, so the two cannot drift.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from nemo.collections.speechlm2.inference.model_wrappers.text_sampling import sample_text_token
from nemo.collections.speechlm2.parts.logit_boosts import LogitBoosts, apply_logit_boosts

SHARED_TEXT_SAMPLING_ARG = "nemo_shared_text_sampling"


def _sampling_config(params: SamplingParams) -> dict[str, Any] | None:
    extra_args = params.extra_args
    if not isinstance(extra_args, dict):
        return None
    value = extra_args.get(SHARED_TEXT_SAMPLING_ARG)
    return value if isinstance(value, dict) else None


@dataclass
class SharedTextSamplingState:
    """Sampling history that survives vLLM streaming segment re-admission."""

    sample_count: int = 0
    tokens: list[int] = field(default_factory=list)


class SharedTextRequestSampler:
    """Select with NeMo's sampler, then force vLLM greedy to that token."""

    def __init__(
        self,
        *,
        top_p: float,
        repetition_penalty: float,
        temperature: float,
        special_token_ids: set[int],
        history_skip: int,
        state: SharedTextSamplingState | None = None,
        boosts: LogitBoosts | None = None,
        pad_id: int | None = None,
        bos_id: int | None = None,
        eos_id: int | None = None,
    ) -> None:
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.special_token_ids = special_token_ids
        self.history_skip = history_skip
        self.state = state or SharedTextSamplingState()
        self.boosts = boosts or LogitBoosts()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        if self.boosts and None in (pad_id, bos_id, eos_id):
            raise ValueError("Agent logit boosts require pad_id, bos_id and eos_id")
        self._special_ids_tensor = (
            torch.tensor(sorted(special_token_ids), dtype=torch.long) if special_token_ids else None
        )

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        del output_ids
        history = self.state.tokens
        generated_tokens = torch.tensor(
            history,
            device=logits.device,
            dtype=torch.long,
        ).unsqueeze(0)
        if self._special_ids_tensor is not None and self._special_ids_tensor.device != logits.device:
            self._special_ids_tensor = self._special_ids_tensor.to(logits.device)

        # Same order as DuplexSTTModel: boost the special tokens, then sample.
        apply_logit_boosts(
            logits,
            self.boosts,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
        )

        sampled = sample_text_token(
            logits.unsqueeze(0),
            generated_tokens,
            len(history),
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            special_token_ids=self.special_token_ids,
            special_ids_tensor=self._special_ids_tensor,
        )
        selected_token = int(sampled[0].item())
        if self.state.sample_count >= self.history_skip:
            self.state.tokens.append(selected_token)
        self.state.sample_count += 1
        logits.fill_(float("-inf"))
        logits[selected_token] = 0.0
        return logits


class SharedTextSamplingLogitsProcessor(AdapterLogitsProcessor):
    """Batch adapter enabling shared VoiceChat text sampling per request."""

    def __init__(
        self,
        vllm_config: Any,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1) or 1)
        self._max_history_states = max(1, max_num_seqs)
        self._history_states: OrderedDict[str, SharedTextSamplingState] = OrderedDict()

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra_args = sampling_params.extra_args
        if not isinstance(extra_args, dict):
            return
        raw_config = extra_args.get(SHARED_TEXT_SAMPLING_ARG)
        if raw_config is None:
            return
        if not isinstance(raw_config, dict):
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG} must be a mapping")

        top_p = raw_config.get("top_p")
        temperature = raw_config.get("temperature")
        repetition_penalty = raw_config.get("repetition_penalty")
        for name, value in (
            ("top_p", top_p),
            ("temperature", temperature),
            ("repetition_penalty", repetition_penalty),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.{name} must be finite")
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.top_p must be in (0, 1]")
        if float(temperature) < 0.0:
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.temperature must be >= 0")
        if float(repetition_penalty) <= 0.0:
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.repetition_penalty must be > 0")

        special_ids = raw_config.get("special_token_ids")
        if not isinstance(special_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in special_ids
        ):
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.special_token_ids " "must be a list of integers")
        history_skip = raw_config.get("history_skip")
        if isinstance(history_skip, bool) or not isinstance(history_skip, int) or history_skip < 0:
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.history_skip " "must be a non-negative integer")
        history_key = raw_config.get("history_key")
        if not isinstance(history_key, str) or not history_key:
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.history_key " "must be a non-empty string")

        boosts = raw_config.get("boosts")
        if boosts is None:
            return
        if not isinstance(boosts, dict):
            raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.boosts must be a mapping")
        for name in ("pad", "bos", "eos"):
            value = boosts.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.boosts.{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{SHARED_TEXT_SAMPLING_ARG}.boosts.{name} must be finite")
        if not any(boosts.get(name) for name in ("pad", "bos", "eos")):
            return
        for name in ("pad_id", "bos_id", "eos_id"):
            token_id = raw_config.get(name)
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(
                    f"{SHARED_TEXT_SAMPLING_ARG}.{name} must be a non-negative " "integer when boosts are set"
                )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> SharedTextRequestSampler | None:
        config = _sampling_config(params)
        if config is None:
            return None
        history_key = str(config["history_key"])
        state = self._history_states.get(history_key)
        if state is None:
            while len(self._history_states) >= self._max_history_states:
                self._history_states.popitem(last=False)
            state = SharedTextSamplingState()
            self._history_states[history_key] = state
        else:
            self._history_states.move_to_end(history_key)
        boosts = LogitBoosts.from_dict(config.get("boosts"))
        return SharedTextRequestSampler(
            top_p=float(config["top_p"]),
            repetition_penalty=float(config["repetition_penalty"]),
            temperature=float(config["temperature"]),
            special_token_ids=set(config["special_token_ids"]),
            history_skip=int(config["history_skip"]),
            state=state,
            boosts=boosts,
            pad_id=config.get("pad_id"),
            bos_id=config.get("bos_id"),
            eos_id=config.get("eos_id"),
        )


__all__ = [
    "SHARED_TEXT_SAMPLING_ARG",
    "SharedTextRequestSampler",
    "SharedTextSamplingState",
    "SharedTextSamplingLogitsProcessor",
]
