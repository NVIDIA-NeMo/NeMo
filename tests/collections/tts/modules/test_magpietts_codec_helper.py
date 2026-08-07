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

from unittest.mock import patch

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


def test_codec_helper_decodes_semantic_tokens_with_continuous_acoustics():
    torch.manual_seed(42)
    codec = _semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(1, 2400)
    audio_lens = torch.tensor([2400])
    codes, codes_lens, embedding = helper.audio_to_codes_and_embedding(audio, audio_lens)
    _, acoustic = helper.split_prequantized_embedding(embedding)

    with (
        patch.object(
            codec,
            "quantize",
            side_effect=AssertionError("continuous acoustic embeddings must not be quantized"),
        ),
        patch.object(
            codec,
            "dequantize",
            side_effect=AssertionError("semantic decode must not construct placeholder acoustic tokens"),
        ),
        patch.object(
            codec.vector_quantizer.fsqs[1],
            "decode",
            side_effect=AssertionError("semantic decode must not touch acoustic quantizer groups"),
        ),
    ):
        decoded_audio, decoded_lens, decoder_input = helper.semantic_and_acoustic_embedding_to_audio(
            semantic_codes=codes[:, :1],
            acoustic_embedding=acoustic,
            codes_len=codes_lens,
        )

    assert decoded_audio.shape[0] == 1
    assert decoded_lens.shape == (1,)
    assert decoder_input.shape == embedding.shape
    torch.testing.assert_close(decoder_input[:, 5:], acoustic)


def test_codec_helper_returns_and_splits_prequantized_embedding():
    torch.manual_seed(42)
    codec = _semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(2, 2400)
    audio_lens = torch.tensor([2400, 1920])

    codes, codes_lens, embedding = helper.audio_to_codes_and_embedding(audio, audio_lens)
    with patch.object(
        codec.vector_quantizer.fsqs[1],
        "encode",
        side_effect=AssertionError("semantic encode must not touch acoustic quantizer groups"),
    ):
        semantic_codes, semantic_lens, semantic_embedding = helper.audio_to_semantic_codes_and_embedding(
            audio, audio_lens, num_semantic_codebooks=1
        )
    semantic, acoustic = helper.split_prequantized_embedding(embedding)
    dequantized = codec.dequantize(tokens=codes, tokens_len=codes_lens)

    assert semantic.shape == (2, 5, embedding.size(2))
    assert acoustic.shape == (2, 35, embedding.size(2))
    torch.testing.assert_close(semantic_codes, codes[:, :1])
    torch.testing.assert_close(semantic_lens, codes_lens)
    torch.testing.assert_close(semantic_embedding, embedding)
    assert not torch.allclose(embedding, dequantized)
    assert not hasattr(helper, "acoustic_embedding_to_codes")
