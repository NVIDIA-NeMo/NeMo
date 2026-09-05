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

from nemo.collections.asr.modules.ggemm_transformer_encoder import GGEMMTransformerEncoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder


def _tiny_transformer() -> TransformerEncoder:
    return TransformerEncoder(
        feat_in=8,
        d_model=16,
        n_heads=1,
        n_layers=1,
        subsampling='feature_stacking',
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        qkv_bias=False,
        qk_norm=False,
        ff_expansion=1.0,
        pre_block_norm=True,
        self_attention_model='rope',
        sync_max_audio_length=False,
    )


@pytest.mark.unit
def test_ggemm_transformer_encoder_requires_an_expert():
    with pytest.raises(ValueError, match="requires at least one expert"):
        GGEMMTransformerEncoder({})


@pytest.mark.unit
def test_ggemm_transformer_encoder_packed_inference_matches_serial_experts():
    encoder = GGEMMTransformerEncoder({'asr': _tiny_transformer(), 'aux': _tiny_transformer()}).eval()
    features = torch.randn(2, 8, 20)
    lengths = torch.tensor([20, 14])

    with torch.no_grad():
        reference = encoder.forward_all(features, lengths)
        packed = encoder.forward_packed(features, lengths)

    assert set(packed) == set(reference) == {'asr', 'aux'}
    for name in reference:
        reference_states, reference_lengths = reference[name]
        packed_states, packed_lengths = packed[name]
        torch.testing.assert_close(packed_states, reference_states, atol=1e-5, rtol=1e-4)
        assert torch.equal(packed_lengths, reference_lengths)
