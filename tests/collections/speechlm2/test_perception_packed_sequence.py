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

from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import unpack_encoder_output
from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule, IdentityConnector
from tests.collections.asr.test_parallel_expert_encoder import build_toy_pe_encoder


class _FeaturePassthrough(torch.nn.Module):
    def forward(self, input_signal, length):
        return input_signal, length


def _make_perception() -> AudioPerceptionModule:
    encoder = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
        self_attention_model='rope',
        sync_max_audio_length=False,
    ).eval()
    perception = AudioPerceptionModule.__new__(AudioPerceptionModule)
    torch.nn.Module.__init__(perception)
    perception.preprocessor = _FeaturePassthrough()
    perception._modules['encoder'] = encoder
    perception.modality_adapter = IdentityConnector()
    perception.proj = torch.nn.Linear(32, 24)
    perception.spec_augmentation = None
    perception.rote = None
    return perception.eval()


def test_perception_sequence_packed_matches_legacy_and_preserves_state_dict():
    torch.manual_seed(0)
    perception = _make_perception()
    features = torch.randn(3, 8, 12)
    lengths = torch.tensor([12, 7, 4])
    state_keys = set(perception.state_dict())

    with torch.no_grad():
        legacy, output_lengths = perception(input_signal=features, input_signal_length=lengths)
        packed = perception.forward_sequence_packed(input_signal=features, input_signal_length=lengths)

    restored = unpack_encoder_output(packed, total_length=legacy.shape[1])
    valid = torch.arange(legacy.shape[1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], legacy[valid], rtol=1e-5, atol=1e-6)
    assert perception.supports_sequence_packed_output
    assert set(perception.state_dict()) == state_keys


def test_perception_sequence_packed_rejects_adapter_that_cannot_preserve_thd():
    perception = _make_perception()
    perception.modality_adapter = torch.nn.Identity()

    assert not perception.supports_sequence_packed_output
    with pytest.raises(ValueError, match="IdentityConnector"):
        perception.forward_sequence_packed(
            input_signal=torch.randn(1, 8, 8),
            input_signal_length=torch.tensor([8]),
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("encoder_kind", ["transformer", "moe", "pee"])
def test_perception_packed_waveform_matches_padded_waveform_for_all_new_encoders(encoder_kind, device):
    torch.manual_seed(17)
    perception = _make_waveform_perception(encoder_kind).to(device)
    lengths = torch.tensor([4096, 2600, 1200], dtype=torch.long, device=device)
    audios = torch.randn(3, int(lengths.max()), device=device)
    for row, length in zip(audios, lengths):
        row[int(length) :] = 0.0
    packed_audio_samples = torch.cat([row[: int(length)] for row, length in zip(audios, lengths)])
    audio_cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    state_keys = set(perception.state_dict())

    with torch.no_grad():
        expected = perception.forward_sequence_packed(input_signal=audios, input_signal_length=lengths)
        actual = perception.forward_sequence_packed(
            input_signal=packed_audio_samples,
            input_signal_length=lengths,
            input_signal_cu_seqlens=audio_cu_seqlens,
        )

    assert torch.equal(actual.lengths, expected.lengths)
    assert torch.equal(actual.cu_seqlens, expected.cu_seqlens)
    if device == "cuda":
        torch.testing.assert_close(actual.data, expected.data, rtol=2e-3, atol=1e-3)
    else:
        torch.testing.assert_close(actual.data, expected.data, rtol=2e-4, atol=2e-5)
    assert set(perception.state_dict()) == state_keys


def _make_waveform_perception(encoder_kind: str) -> AudioPerceptionModule:
    if encoder_kind == "transformer":
        encoder = TransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            subsampling_factor=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            sync_max_audio_length=False,
        )
        features = 8
    elif encoder_kind == "moe":
        encoder = MoETransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            subsampling_factor=2,
            drop_rate=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            self_attention_model="rope",
            moe_num_experts=4,
            moe_top_k=2,
            sync_max_audio_length=False,
        )
        features = 8
    else:
        encoder = build_toy_pe_encoder(always_run_diarization=True)
        features = 128

    perception = AudioPerceptionModule.__new__(AudioPerceptionModule)
    torch.nn.Module.__init__(perception)
    perception.preprocessor = AudioToMelSpectrogramPreprocessor(
        features=features,
        normalize="per_feature",
        dither=0,
        pad_to=0,
    )
    perception._modules["encoder"] = encoder
    perception.modality_adapter = IdentityConnector()
    perception.proj = torch.nn.Linear(encoder.d_model, 24)
    perception.spec_augmentation = None
    perception.rote = None
    return perception.eval()
