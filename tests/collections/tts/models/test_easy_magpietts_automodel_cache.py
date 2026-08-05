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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel, TrainingMode


class _FakeAutomodelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()
        self.weight = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return SimpleNamespace(past_key_values=kwargs['past_key_values'])


class _FakeNemotronHybridCache:
    def __init__(self, config, batch_size, dtype, device):
        self.config = config
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.has_previous_state = False
        self.conv_states = {0: torch.ones(1)}
        self.ssm_states = {0: torch.ones(1)}
        self.key_cache = [torch.ones(1)]
        self.value_cache = [torch.ones(1)]


@pytest.mark.unit
def test_automodel_forward_creates_hybrid_cache_for_prefill(monkeypatch):
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    nn.Module.__init__(model)
    model.decoder_type = 'nemo_automodel'
    model.decoder = _FakeAutomodelDecoder()
    monkeypatch.setattr(model, '_get_automodel_cache_class', lambda: _FakeNemotronHybridCache)

    inputs_embeds = torch.randn(2, 4, 8)
    output = model.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        use_cache=True,
        past_key_values=None,
        cache_position=torch.arange(4),
    )

    cache = output.past_key_values
    assert isinstance(cache, _FakeNemotronHybridCache)
    assert cache.config is model.decoder.config
    assert cache.batch_size == inputs_embeds.shape[0]
    assert cache.dtype == model.decoder.weight.dtype
    assert cache.device == inputs_embeds.device
    assert model.decoder.forward_kwargs['past_key_values'] is cache
    assert torch.equal(model.decoder.forward_kwargs['cache_position'], torch.arange(4))


@pytest.mark.unit
def test_automodel_forward_reuses_existing_hybrid_cache(monkeypatch):
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    nn.Module.__init__(model)
    model.decoder_type = 'nemo_automodel'
    model.decoder = _FakeAutomodelDecoder()
    monkeypatch.setattr(
        model,
        '_get_automodel_cache_class',
        lambda: pytest.fail('cache class should not be loaded when a cache already exists'),
    )
    existing_cache = object()

    output = model.forward(
        inputs_embeds=torch.randn(2, 1, 8),
        attention_mask=None,
        use_cache=True,
        past_key_values=existing_cache,
        cache_position=torch.tensor([4]),
    )

    assert output.past_key_values is existing_cache
    assert model.decoder.forward_kwargs['past_key_values'] is existing_cache
    assert torch.equal(model.decoder.forward_kwargs['cache_position'], torch.tensor([4]))


@pytest.mark.unit
def test_automodel_streaming_cache_lifecycle(monkeypatch):
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    nn.Module.__init__(model)
    model.decoder_type = "nemo_automodel"
    model.decoder = _FakeAutomodelDecoder()
    model.default_inference_mode = "streaming_1_2"
    model.mode_name_to_mode = {"streaming_1_2": TrainingMode("streaming", 1, 2, 0)}
    model.phoneme_tokenizer = None
    model.num_audio_codebooks = 2

    context_embedding = torch.randn(2, 4, 8)
    context_lens = torch.tensor([4, 4])
    context_audio_codes = torch.zeros(2, 2, 3, dtype=torch.long)
    context_audio_codes_lens = torch.tensor([3, 3])
    monkeypatch.setattr(
        model,
        "prepare_context_tensors",
        lambda **kwargs: (context_embedding, context_lens, context_audio_codes, context_audio_codes_lens),
    )

    created_from = []

    def create_cache(inputs_embeds):
        created_from.append(inputs_embeds)
        return _FakeNemotronHybridCache(
            model.decoder.config,
            inputs_embeds.shape[0],
            model.decoder.weight.dtype,
            inputs_embeds.device,
        )

    forward_calls = []

    def forward(**kwargs):
        forward_calls.append(kwargs)
        return SimpleNamespace(
            hidden_states=(torch.zeros_like(kwargs["inputs_embeds"]),),
            past_key_values=kwargs["past_key_values"],
        )

    monkeypatch.setattr(model, "_create_automodel_cache", create_cache)
    monkeypatch.setattr(model, "forward", forward)

    state = model.streaming_init(
        context_audio_codes=context_audio_codes,
        context_audio_codes_lens=context_audio_codes_lens,
        context_text_tokens=torch.zeros(2, 2, dtype=torch.long),
        context_text_tokens_lens=torch.tensor([2, 2]),
    )

    cache = state.past_key_values
    assert isinstance(cache, _FakeNemotronHybridCache)
    assert created_from[0].shape == (2, 4, 8)
    assert forward_calls[0]["past_key_values"] is cache
    assert state.cache_seq_len == 4

    empty_phase_mask = torch.zeros(2, dtype=torch.bool)
    monkeypatch.setattr(
        model,
        "_prepare_streaming_input",
        lambda *args, **kwargs: (torch.randn(2, 1, 8), empty_phase_mask, empty_phase_mask, empty_phase_mask),
    )
    monkeypatch.setattr(model, "_process_predictions", lambda *args, **kwargs: (None, None))

    state, _, _ = model.streaming_step(state)

    assert forward_calls[-1]["past_key_values"] is cache
    assert state.past_key_values is cache
    assert state.cache_seq_len == 5

    cache.has_previous_state = True
    model.streaming_finalize(state)

    assert state.past_key_values is None
    assert state.cache_seq_len == 0
    assert cache.has_previous_state is False
    assert cache.conv_states == {}
    assert cache.ssm_states == {}
    assert cache.key_cache == []
    assert cache.value_cache == []
