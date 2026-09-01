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

import pytest
import torch

from nemo.collections.speechlm2.inference.model_wrappers.text_sampling import sample_text_token


def test_special_tokens_bypass_sampling_and_never_penalise_history(monkeypatch):
    """Special tokens are chosen greedily and excluded from the repetition
    penalty, so a pad-heavy history cannot bias the next real token."""
    # All-ones parameters reduce to greedy.
    assert sample_text_token(
        torch.tensor([[0.1, 0.8, 0.3]]),
        torch.empty((1, 0), dtype=torch.long),
        0,
        top_p=1.0,
        repetition_penalty=1.0,
        temperature=1.0,
        special_token_ids=set(),
    ).tolist() == [1]

    penalty_kwargs = {
        "top_p": 1.0,
        "repetition_penalty": 1.5,
        "temperature": 0.8,
        "special_token_ids": {0},
        "special_ids_tensor": torch.tensor([0]),
    }
    logits = torch.tensor([[0.2, 1.1, 0.9, 0.4]])
    torch.manual_seed(9)
    with_special_history = sample_text_token(logits, torch.tensor([[0]], dtype=torch.long), 1, **penalty_kwargs)
    torch.manual_seed(9)
    without_history = sample_text_token(logits, torch.empty((1, 0), dtype=torch.long), 0, **penalty_kwargs)
    assert torch.equal(with_special_history, without_history)

    # A special token wins outright rather than going through multinomial.
    def fail_multinomial(*_args, **_kwargs):
        raise AssertionError("special-token bypass called torch.multinomial")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    assert sample_text_token(
        torch.tensor([[0.1, 1.5, 0.7]]),
        torch.tensor([[2]], dtype=torch.long),
        1,
        top_p=0.8,
        repetition_penalty=1.2,
        temperature=0.7,
        special_token_ids={1},
        special_ids_tensor=torch.tensor([1]),
    ).tolist() == [1]


def test_vllm_processor_reuses_shared_sampler_and_skips_prefill_history():
    pytest.importorskip("vllm")
    from nemo.collections.speechlm2.inference.vllm_omni.nemotron_duplex_h.sampling import (
        SharedTextRequestSampler,
        SharedTextSamplingState,
    )

    logits = torch.tensor([0.2, 1.3, 0.9, 0.5])
    params = {
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "temperature": 0.75,
        "special_token_ids": {0},
    }

    torch.manual_seed(17)
    expected = sample_text_token(
        logits.unsqueeze(0),
        torch.tensor([[2, 2]], dtype=torch.long),
        2,
        special_ids_tensor=torch.tensor([0]),
        **params,
    )

    processor = SharedTextRequestSampler(
        history_skip=1,
        state=SharedTextSamplingState(
            sample_count=3,
            tokens=[2, 2],
        ),
        **params,
    )
    torch.manual_seed(17)
    forced_logits = processor([], logits.clone())

    assert int(forced_logits.argmax().item()) == int(expected[0].item())
    assert torch.isfinite(forced_logits).sum().item() == 1


def test_vllm_processor_history_survives_segment_readmission():
    """A vLLM request is re-admitted per streaming segment; the sampling
    history has to outlive that or the repetition penalty resets mid-stream."""
    pytest.importorskip("vllm")
    from types import SimpleNamespace

    from vllm import SamplingParams

    from nemo.collections.speechlm2.inference.vllm_omni.nemotron_duplex_h.sampling import (
        SHARED_TEXT_SAMPLING_ARG,
        SharedTextSamplingLogitsProcessor,
    )

    adapter = SharedTextSamplingLogitsProcessor(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1)),
        torch.device("cpu"),
        False,
    )
    params = SamplingParams(
        temperature=0.0,
        extra_args={
            SHARED_TEXT_SAMPLING_ARG: {
                "top_p": 1.0,
                "temperature": 1.0,
                "repetition_penalty": 1.0,
                "special_token_ids": [0],
                "history_skip": 1,
                "history_key": "stream-1",
            }
        },
    )

    first_segment = adapter.new_req_logits_processor(params)
    assert first_segment is not None
    first_segment([], torch.tensor([0.0, 2.0, 1.0]))
    first_segment([], torch.tensor([0.0, 1.0, 2.0]))

    resumed_segment = adapter.new_req_logits_processor(params)
    assert resumed_segment is not None
    assert resumed_segment.state is first_segment.state
    assert resumed_segment.state.sample_count == 2
    assert resumed_segment.state.tokens == [2]
