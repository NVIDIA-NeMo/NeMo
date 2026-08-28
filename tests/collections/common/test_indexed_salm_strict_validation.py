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

"""Coverage for strict validation-time audio loading."""

import pytest
import torch
from lhotse import CutSet

from nemo.collections.common.data.lhotse.text_adapters import NeMoMultimodalConversation, TextTurn
from nemo.collections.speechlm2.data import salm_dataset
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset


class _Tokenizer:
    pad = 0
    unk_id = 0


def _conversation(id_):
    conversation = NeMoMultimodalConversation(id_, [TextTurn("not serialized", "user")])
    conversation.input_ids = torch.tensor([1, 2])
    conversation.mask = torch.tensor([0, 1])
    return conversation


def test_strict_audio_loading_rejects_partial_conversation_drop(monkeypatch):
    first = _conversation("first")
    second = _conversation("second")
    requested = CutSet([first, second])

    def partial(conversations, load_audio):
        return torch.empty(0), torch.empty(0, dtype=torch.long), CutSet([first])

    monkeypatch.setattr(salm_dataset, "collate_conversation_audio_fault_tolerant", partial)
    strict = SALMDataset(_Tokenizer(), strict_audio_loading=True)

    with pytest.raises(RuntimeError, match="dropped or reordered conversations"):
        strict[requested]


def test_default_training_fault_tolerance_still_accepts_partial_drop(monkeypatch):
    first = _conversation("first")
    second = _conversation("second")
    requested = CutSet([first, second])

    def partial(conversations, load_audio):
        return torch.empty(0), torch.empty(0, dtype=torch.long), CutSet([first])

    monkeypatch.setattr(salm_dataset, "collate_conversation_audio_fault_tolerant", partial)
    training = SALMDataset(_Tokenizer())

    batch = training[requested]

    assert [conversation.id for conversation in batch["conversations"]] == ["first"]


def test_strict_audio_loading_reraises_collation_errors(monkeypatch):
    requested = CutSet([_conversation("first")])

    def fail(conversations, load_audio):
        raise OSError("decode failed")

    monkeypatch.setattr(salm_dataset, "collate_conversation_audio_fault_tolerant", fail)

    assert SALMDataset(_Tokenizer())[requested] is None
    with pytest.raises(OSError, match="decode failed"):
        SALMDataset(_Tokenizer(), strict_audio_loading=True)[requested]
