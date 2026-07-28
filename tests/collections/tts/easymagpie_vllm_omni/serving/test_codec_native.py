# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from easymagpie_vllm_omni.codec.codec import EasyMagpieCodec, FiniteScalarDequantizer
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packing import unstack_acoustic_codes
from easymagpie_vllm_omni.codec.weight_conversion import fold_weight_norm


def tiny_config() -> EasyMagpieCodecConfig:
    return EasyMagpieCodecConfig(
        input_dim=4,
        input_filters=8,
        hidden_filters=16,
        num_hidden_layers=2,
        pre_upsample_rates=[2],
        pre_upsample_filters=[8],
        resblock_upsample_rates=[2],
        resblock_upsample_filters=[4],
        num_codebooks=2,
        codebook_size=4,
        num_levels_per_group=[2, 2],
        frame_stacking_factor=2,
    )


def test_fsq_decode() -> None:
    decode = FiniteScalarDequantizer(2, [2, 2])
    indices = torch.tensor([[[0, 3]]])
    actual = decode(indices)
    expected = torch.tensor([[[-1.0, -1.0, 0.0, 0.0]]])
    torch.testing.assert_close(actual, expected)


def test_unstack_predictor_codes_preserves_codebook_and_time_order() -> None:
    # Each predictor row is [c0_t0, c0_t1, c1_t0, c1_t1, ...].
    stacked = torch.tensor(
        [
            [0, 1, 10, 11, 20, 21],
            [100, 101, 110, 111, 120, 121],
        ]
    )
    actual = unstack_acoustic_codes(stacked, num_codebooks=3, frame_stacking_factor=2)
    expected = torch.tensor(
        [
            [0, 10, 20],
            [1, 11, 21],
            [100, 110, 120],
            [101, 111, 121],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_fold_weight_norm_matches_torch() -> None:
    conv = torch.nn.Conv1d(3, 5, 3)
    torch.nn.utils.parametrizations.weight_norm(conv)
    g = conv.parametrizations.weight.original0.detach()
    v = conv.parametrizations.weight.original1.detach()
    torch.testing.assert_close(fold_weight_norm(g, v), conv.weight.detach())


def test_streaming_matches_full_decode() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = EasyMagpieCodec(config).eval()
    codes = torch.randint(0, config.codebook_size, (1, 7, config.num_stacked_codebooks))

    expected = model(codes)
    state = None
    chunks = []
    for start, end in ((0, 1), (1, 4), (4, 6), (6, 7)):
        audio, state = model.stream(codes[:, start:end], state)
        chunks.append(audio)
    actual = torch.cat(chunks, dim=-1)

    assert actual.shape == (1, 7 * config.samples_per_frame)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"activation": "snake"}, "half_snake"),
        ({"codebook_size": 8}, "product of num_levels_per_group"),
        ({"pre_upsample_filters": [3]}, "divisible"),
    ],
)
def test_codec_config_rejects_unsupported_topologies(kwargs, message) -> None:
    config_kwargs = {
        "input_dim": 4,
        "input_filters": 8,
        "hidden_filters": 16,
        "num_hidden_layers": 2,
        "pre_upsample_rates": [2],
        "pre_upsample_filters": [8],
        "resblock_upsample_rates": [2],
        "resblock_upsample_filters": [4],
        "num_codebooks": 2,
        "codebook_size": 4,
        "num_levels_per_group": [2, 2],
        "frame_stacking_factor": 2,
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        EasyMagpieCodecConfig(**config_kwargs)
