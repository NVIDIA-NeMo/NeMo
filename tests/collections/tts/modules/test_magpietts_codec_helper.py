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

from nemo.collections.tts.models.audio_codec import AudioCodecModel
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from tests.collections.tts.models.test_audio_codec import create_codec_config


pytestmark = pytest.mark.unit


def _semantic_acoustic_codec():
    semantic_cfg = create_codec_config()
    semantic_cfg.vector_quantizer.params.num_groups = 1
    semantic_cfg.audio_encoder.params.out_dim = 5
    semantic_cfg.audio_decoder.params.input_dim = 5

    codec_cfg = create_codec_config()
    codec_cfg.semantic_codec = semantic_cfg
    codec_cfg.audio_encoder.params.out_dim = 35
    return AudioCodecModel(cfg=codec_cfg).eval()


def test_codec_helper_returns_and_splits_prequantized_embedding():
    torch.manual_seed(42)
    codec = _semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(2, 2400)
    audio_lens = torch.tensor([2400, 1920])

    codes, codes_lens, embedding = helper.audio_to_codes_and_embedding(audio, audio_lens)
    semantic, acoustic = helper.split_prequantized_embedding(embedding)
    reconstructed = helper.acoustic_embedding_to_codes(codes[:, :1], acoustic, codes_lens)
    dequantized = codec.dequantize(tokens=codes, tokens_len=codes_lens)

    assert semantic.shape == (2, 5, embedding.size(2))
    assert acoustic.shape == (2, 35, embedding.size(2))
    assert not torch.allclose(embedding, dequantized)
    for batch_idx, length in enumerate(codes_lens.tolist()):
        assert torch.equal(reconstructed[batch_idx, :, :length], codes[batch_idx, :, :length])
