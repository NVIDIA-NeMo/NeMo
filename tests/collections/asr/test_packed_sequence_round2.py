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

import pytest
import torch

from nemo.collections.asr.modules.moe_transformer_encoder import MoEFeedForward, MoETransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output


def test_packed_output_with_data_reuses_validated_metadata_and_preserves_gradients():
    packed = pack_encoder_output(torch.randn(2, 4, 3), torch.tensor([4, 2]))
    replacement = torch.randn(6, 5, requires_grad=True)

    updated = packed.with_data(replacement)

    assert updated.lengths is packed.lengths
    assert updated.cu_seqlens is packed.cu_seqlens
    assert updated.max_seqlen == packed.max_seqlen
    updated.data.square().sum().backward()
    assert replacement.grad is not None
    with pytest.raises(ValueError, match="replacement data"):
        packed.with_data(torch.randn(5, 3))


@pytest.mark.parametrize(("top_k", "router_type"), [(1, "switch"), (2, "omni")])
def test_moe_packed_routing_statistics_auxiliary_loss_and_reset(top_k, router_type):
    torch.manual_seed(0)
    encoder = MoETransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        self_attention_model="rope",
        moe_num_experts=4,
        moe_top_k=top_k,
        moe_router_type=router_type,
        sync_max_audio_length=False,
    ).train()
    lengths = torch.tensor([7, 3, 1])

    with torch.no_grad():
        encoder.forward_sequence_packed(torch.randn(3, 7, 32), lengths, bypass_pre_encode=True)

    ffn = encoder.layers[0].ffn
    assert isinstance(ffn, MoEFeedForward)
    num_tokens = int(lengths.sum())
    assert ffn._num_tokens == num_tokens
    assert int(ffn._expert_counts.sum()) == num_tokens * top_k
    torch.testing.assert_close(ffn._gate_prob_sum.sum(), torch.tensor(float(num_tokens)))
    expected_aux = (
        ffn.num_experts * (ffn._expert_counts.float() / num_tokens * (ffn._gate_prob_sum / num_tokens)).sum()
    )
    torch.testing.assert_close(ffn._aux_loss.float(), expected_aux.float())
    torch.testing.assert_close(encoder._cum_counts[0], ffn._expert_counts)
    torch.testing.assert_close(encoder._cum_prob_sum[0], ffn._gate_prob_sum)
    assert int(encoder._cum_tokens[0]) == num_tokens

    metrics = encoder.get_moe_metrics(distributed=False, reset=True)
    assert metrics is not None
    assert not encoder._cum_counts.any()
    assert not encoder._cum_prob_sum.any()
    assert not encoder._cum_tokens.any()


def test_moe_packed_auxiliary_loss_is_padding_neutral_while_legacy_contract_is_unchanged():
    torch.manual_seed(0)
    encoder = MoETransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        self_attention_model="rope",
        moe_num_experts=4,
        moe_top_k=2,
        sync_max_audio_length=False,
    ).train()
    inputs = torch.randn(2, 6, 32)
    lengths = torch.tensor([6, 2])

    with torch.no_grad():
        encoder(inputs, lengths, bypass_pre_encode=True)
        legacy_tokens = encoder.layers[0].ffn._num_tokens
        legacy_auxiliary_loss = encoder.get_moe_auxiliary_loss()
        encoder.forward_sequence_packed(inputs, lengths, bypass_pre_encode=True)
        packed_tokens = encoder.layers[0].ffn._num_tokens
        packed_auxiliary_loss = encoder.get_moe_auxiliary_loss()

    assert legacy_tokens == inputs.shape[0] * inputs.shape[1]
    assert packed_tokens == int(lengths.sum())
    assert torch.isfinite(legacy_auxiliary_loss)
    assert torch.isfinite(packed_auxiliary_loss)
