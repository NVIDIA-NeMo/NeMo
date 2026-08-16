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


def _semantic_acoustic_codec(*, dithered_acoustic_fsq: bool = False):
    semantic_cfg = create_codec_config()
    semantic_cfg.vector_quantizer.params.num_groups = 1
    semantic_cfg.audio_encoder.params.out_dim = 5
    semantic_cfg.audio_decoder.params.input_dim = 5

    codec_cfg = create_codec_config()
    codec_cfg.semantic_codec = semantic_cfg
    codec_cfg.audio_encoder.params.out_dim = 35
    if dithered_acoustic_fsq:
        codec_cfg.vector_quantizer.params.dithered_acoustic_fsq = True
        codec_cfg.vector_quantizer.params.num_semantic_groups = 1
    return AudioCodecModel(cfg=codec_cfg).eval()


def _hybrid_semantic_acoustic_codec():
    semantic_cfg = create_codec_config()
    semantic_cfg.vector_quantizer.params.num_groups = 1
    semantic_cfg.audio_encoder.params.out_dim = 5
    semantic_cfg.audio_decoder.params.input_dim = 5

    codec_cfg = create_codec_config()
    codec_cfg.semantic_codec = semantic_cfg
    codec_cfg.audio_encoder.params.out_dim = 35
    codec_cfg.hybrid_codec = {
        "continuous_dim": 35,
        "residual_dropout_rate": 0.5,
        "kl_loss_scale": 0.1,
    }
    return AudioCodecModel(cfg=codec_cfg).eval()


def test_codec_helper_decodes_semantic_tokens_with_continuous_acoustics():
    torch.manual_seed(42)
    codec = _semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(1, 2400)
    audio_lens = torch.tensor([2400])
    codes, codes_lens, embedding = helper.audio_to_codes_and_embedding(audio, audio_lens)
    _, acoustic = helper.split_continuous_embedding(embedding)

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


def test_codec_helper_returns_and_splits_continuous_fsq_embedding():
    torch.manual_seed(42)
    codec = _semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(2, 2400)
    audio_lens = torch.tensor([2400, 1920])

    raw_embedding, raw_embedding_lens = helper.audio_to_prequantized_embedding(audio, audio_lens)
    codes, codes_lens, embedding = helper.audio_to_codes_and_embedding(audio, audio_lens)
    with patch.object(
        codec.vector_quantizer.fsqs[1],
        "encode",
        side_effect=AssertionError("semantic encode must not touch acoustic quantizer groups"),
    ):
        semantic_codes, semantic_lens, acoustic_embedding = helper.audio_to_semantic_codes_and_acoustic_embedding(
            audio, audio_lens, num_semantic_codebooks=1
        )
    semantic, acoustic = helper.split_continuous_embedding(embedding)
    dequantized = codec.dequantize(tokens=codes, tokens_len=codes_lens)

    expected_groups = []
    for raw_group, fsq_group in zip(raw_embedding.chunk(codec.num_codebooks, dim=1), codec.vector_quantizer.fsqs):
        scale = (fsq_group.num_levels // 2).to(raw_group.dtype)
        expected_groups.append(fsq_group.compress(inputs=raw_group, input_len=raw_embedding_lens) / scale)
    expected_embedding = torch.cat(expected_groups, dim=1)

    assert semantic.shape == (2, 5, embedding.size(2))
    assert acoustic.shape == (2, 35, embedding.size(2))
    torch.testing.assert_close(semantic_codes, codes[:, :1])
    torch.testing.assert_close(semantic_lens, codes_lens)
    torch.testing.assert_close(acoustic_embedding, acoustic)
    torch.testing.assert_close(embedding, expected_embedding)
    assert not torch.allclose(raw_embedding, embedding)
    assert not torch.allclose(embedding, dequantized)


def test_codec_helper_quantizes_bounded_acoustic_fsq_values_without_recompression():
    torch.manual_seed(42)
    codec = _semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(2, 2400)
    audio_lens = torch.tensor([2400, 1920])

    semantic_from_codec, teacher_lens, target_from_codec, acoustic_from_codec = (
        helper.audio_to_semantic_codes_acoustic_embedding_and_codes(
            audio,
            audio_lens,
            num_semantic_codebooks=1,
        )
    )
    full_codes, codes_lens, full_embedding = helper.audio_to_codes_and_embedding(audio, audio_lens)
    _, acoustic_embedding = helper.split_continuous_embedding(full_embedding)
    acoustic_codes = helper.acoustic_embedding_to_codes(
        acoustic_embedding,
        num_semantic_codebooks=1,
        codes_len=codes_lens,
    )

    torch.testing.assert_close(semantic_from_codec, full_codes[:, :1])
    torch.testing.assert_close(acoustic_from_codec, full_codes[:, 1:])
    torch.testing.assert_close(target_from_codec, acoustic_embedding)
    torch.testing.assert_close(teacher_lens, codes_lens)
    torch.testing.assert_close(acoustic_codes, full_codes[:, 1:].long())

    out_of_range = acoustic_embedding.clone()
    out_of_range[:, 0::2] = 100.0
    out_of_range[:, 1::2] = -100.0
    clipped_embedding = helper.clamp_acoustic_embedding(out_of_range, num_semantic_codebooks=1)
    clipped_codes = helper.acoustic_embedding_to_codes(out_of_range, num_semantic_codebooks=1)
    assert clipped_embedding.max().item() == pytest.approx(0.49925)
    assert clipped_embedding.min().item() == pytest.approx(-0.99925)
    assert clipped_codes.min() >= 0
    assert clipped_codes.max() < codec.codebook_size


def test_codec_helper_uses_symmetric_dithered_acoustic_fsq_geometry():
    codec = _semantic_acoustic_codec(dithered_acoustic_fsq=True)
    helper = CodecHelper(codec)
    embedding_lens = torch.tensor([2])
    prequantized = torch.linspace(-1.5, 1.5, steps=80).reshape(1, 40, 2)

    continuous = helper._continuous_fsq_embedding(prequantized, embedding_lens)
    group_dim = codec.vector_quantizer.codebook_dim_per_group
    semantic_group = codec.vector_quantizer.fsqs[0]
    semantic_scale = (semantic_group.num_levels // 2).to(prequantized.dtype)
    expected_semantic = (
        semantic_group.compress(inputs=prequantized[:, :group_dim], input_len=embedding_lens) / semantic_scale
    )
    expected_acoustic = prequantized[:, group_dim:].clamp(-1.0, 1.0)
    torch.testing.assert_close(continuous, torch.cat([expected_semantic, expected_acoustic], dim=1))

    full_codes = codec.quantize(encoded=prequantized, encoded_len=embedding_lens)
    acoustic_codes = helper.acoustic_embedding_to_codes(
        continuous[:, group_dim:],
        num_semantic_codebooks=1,
        codes_len=embedding_lens,
    )
    torch.testing.assert_close(acoustic_codes, full_codes[:, 1:].long())

    out_of_range = continuous[:, group_dim:].clone()
    out_of_range[:, 0::2] = 2.0
    out_of_range[:, 1::2] = -2.0
    clipped = helper.clamp_acoustic_embedding(out_of_range, num_semantic_codebooks=1)
    assert clipped.max().item() == 1.0
    assert clipped.min().item() == -1.0


def test_codec_helper_rejects_acoustic_token_quantization_for_hybrid_codec():
    codec = _hybrid_semantic_acoustic_codec()
    helper = CodecHelper(codec)
    with pytest.raises(ValueError, match="do not have acoustic token groups"):
        helper.acoustic_embedding_to_codes(torch.zeros(1, 35, 2), num_semantic_codebooks=1)


def test_codec_helper_uses_hybrid_posterior_mean_as_acoustic_embedding():
    torch.manual_seed(42)
    codec = _hybrid_semantic_acoustic_codec()
    helper = CodecHelper(codec)
    audio = torch.randn(2, 2400)
    audio_lens = torch.tensor([2400, 1920])

    semantic_codes, codes_lens, acoustic_embedding = helper.audio_to_semantic_codes_and_acoustic_embedding(
        audio,
        audio_lens,
        num_semantic_codebooks=1,
    )
    expected_codes, residual_mu, _, expected_lens = codec.encode_hybrid(audio, audio_lens)
    semantic_embedding = helper.semantic_codes_to_embedding(semantic_codes, codes_lens)

    torch.testing.assert_close(semantic_codes, expected_codes)
    torch.testing.assert_close(codes_lens, expected_lens)
    torch.testing.assert_close(acoustic_embedding, residual_mu)
    assert semantic_embedding.size(1) == 5
    assert acoustic_embedding.size(1) == 35

    with patch.object(codec, "decode_hybrid", wraps=codec.decode_hybrid) as decode_hybrid:
        decoded_audio, decoded_lens, combined_embedding = helper.semantic_and_acoustic_embedding_to_audio(
            semantic_codes=semantic_codes,
            acoustic_embedding=acoustic_embedding,
            codes_len=codes_lens,
        )

    assert decoded_audio.shape[0] == audio.shape[0]
    assert decoded_lens.shape == audio_lens.shape
    assert combined_embedding.shape[1] == semantic_embedding.size(1) + acoustic_embedding.size(1)
    torch.testing.assert_close(combined_embedding[:, : semantic_embedding.size(1)], semantic_embedding)
    torch.testing.assert_close(combined_embedding[:, semantic_embedding.size(1) :], acoustic_embedding)
    torch.testing.assert_close(decode_hybrid.call_args.kwargs["residual"], residual_mu)
