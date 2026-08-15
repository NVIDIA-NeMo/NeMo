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

import copy

import pytest
import torch

from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, unpack_encoder_output
from tests.collections.asr.test_packed_transformer_encoder import _make_moe_encoder
from tests.collections.asr.test_parallel_expert_encoder import (
    _MEL_FEATURES,
    _N_SPK,
    build_toy_pe_encoder,
)


def test_previous_moe_state_loads_strictly_before_and_after_packed_use():
    previous = _make_moe_encoder()
    previous_state = copy.deepcopy(previous.state_dict())
    restored = _make_moe_encoder()

    restored.load_state_dict(previous_state, strict=True)
    with torch.no_grad():
        restored.forward_sequence_packed(torch.randn(2, 8, 12), torch.tensor([12, 5]))

    assert set(restored.state_dict()) == set(previous_state)


def test_sequence_packed_training_dropout_is_finite_and_reproducible_within_path():
    encoder = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.2,
        dropout_pre_encoder=0.2,
        dropout_emb=0.2,
        self_attention_model='rope',
        sync_max_audio_length=False,
    ).train()
    audio = torch.randn(3, 8, 12)
    lengths = torch.tensor([12, 7, 3])

    with torch.no_grad():
        torch.manual_seed(17)
        first = encoder.forward_sequence_packed(audio, lengths).data
        torch.manual_seed(17)
        second = encoder.forward_sequence_packed(audio, lengths).data

    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize('target_mode', ['none', 'mixed', 'external'])
def test_pee_packed_fallback_matches_padded_routing_modes(target_mode):
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(3, _MEL_FEATURES, 40)
    lengths = torch.tensor([40, 23, 9])
    mels[1, :, 23:] = 0.0
    mels[2, :, 9:] = 0.0
    targets = None
    if target_mode != 'none':
        targets = torch.zeros(3, 5, _N_SPK)
        targets[0, :, 0] = 1.0
        targets[2, :, 2] = 1.0
        if target_mode == 'mixed':
            targets[1] = -1.0

    with torch.no_grad():
        padded, output_lengths = encoder(mels, lengths, spk_targets=targets)
        packed = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)

    restored = unpack_encoder_output(packed, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-5, atol=1e-6)


def test_pee_packed_input_metadata_is_preserved_through_fallback():
    encoder = build_toy_pe_encoder().eval()
    lengths = torch.tensor([32, 17])
    mels = torch.randn(2, _MEL_FEATURES, 32)
    mels[1, :, 17:] = 0.0
    packed_input = pack_encoder_output(mels.transpose(1, 2), lengths)

    with torch.no_grad():
        dense = encoder.forward_sequence_packed(mels, lengths)
        packed = encoder.forward_sequence_packed(packed_input)

    torch.testing.assert_close(packed.data, dense.data, rtol=1e-5, atol=1e-6)
    assert torch.equal(packed.lengths, dense.lengths)
    assert torch.equal(packed.cu_seqlens, dense.cu_seqlens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='PEE packed gradient parity requires CUDA')
def test_pee_packed_fallback_matches_padded_input_and_parameter_gradients():
    torch.manual_seed(0)
    padded_encoder = build_toy_pe_encoder().cuda().eval()
    packed_encoder = copy.deepcopy(padded_encoder)
    padded_mels = torch.randn(2, _MEL_FEATURES, 32, device='cuda', requires_grad=True)
    packed_mels = padded_mels.detach().clone().requires_grad_()
    lengths = torch.tensor([32, 17], device='cuda')
    targets = torch.zeros(2, 4, _N_SPK, device='cuda')
    targets[0, :, 0] = 1.0

    padded, output_lengths = padded_encoder(padded_mels, lengths, spk_targets=targets)
    packed = packed_encoder.forward_sequence_packed(packed_mels, lengths, spk_targets=targets)
    valid = torch.arange(padded.shape[-1], device='cuda')[None, :] < output_lengths[:, None]
    padded.transpose(1, 2)[valid].float().square().mean().backward()
    packed.data.float().square().mean().backward()

    torch.testing.assert_close(packed_mels.grad, padded_mels.grad, rtol=2e-3, atol=2e-4)
    for name in ('asr_norm.weight', 'asr_encoder.layers.0.self_attn.linear_q.weight'):
        padded_grad = dict(padded_encoder.named_parameters())[name].grad
        packed_grad = dict(packed_encoder.named_parameters())[name].grad
        torch.testing.assert_close(packed_grad, padded_grad, rtol=2e-3, atol=2e-4)
