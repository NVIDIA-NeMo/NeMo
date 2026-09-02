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
