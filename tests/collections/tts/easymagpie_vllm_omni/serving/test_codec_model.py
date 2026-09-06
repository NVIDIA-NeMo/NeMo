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

from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.codec.model import EasyMagpieCodecForConditionalGeneration


def _payload(frames: int, codebooks: int = 16) -> dict:
    return {"codes": {"audio": torch.arange(frames * codebooks).view(frames, codebooks)}}


def _model(**overrides):
    config = dict(
        num_stacked_codebooks=16,
        frame_stacking_factor=2,
        codebook_size=1024,
        samples_per_codec_frame=3,
        samples_per_frame=6,
    )
    config.update(overrides)
    model = EasyMagpieCodecForConditionalGeneration.__new__(EasyMagpieCodecForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(**config)
    return model


def test_payload_codes_skips_active_but_unscheduled_requests():
    model = _model()
    infos = [_payload(2), _payload(4), _payload(1)]
    spans = [(0, 2), (2, 2), (2, 3)]

    codes, frame_counts, valid_sample_counts = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, infos, torch.device("cpu"), spans
    )

    assert codes.shape == (3, 16)
    assert frame_counts == [2, 1]
    assert valid_sample_counts == [12, 6]
    assert torch.equal(codes[:2], infos[0]["codes"]["audio"])
    assert torch.equal(codes[2:], infos[2]["codes"]["audio"])


def test_payload_codes_rejects_scheduled_frame_mismatch():
    model = _model()

    with pytest.raises(ValueError, match="scheduled 1 placeholders.*2 frames"):
        EasyMagpieCodecForConditionalGeneration._payload_codes(model, [_payload(2)], torch.device("cpu"), [(0, 1)])


def test_payload_codes_computes_valid_lengths_before_device_transfer():
    model = _model(num_stacked_codebooks=4)
    infos = [
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [3, 1025, 103, 1025]])}},
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [1025, 3, 1025, 103]])}},
    ]

    codes, frame_counts, valid_sample_counts = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, infos, torch.device("meta")
    )

    assert codes.device.type == "meta"
    assert frame_counts == [2, 2]
    assert valid_sample_counts == [9, 6]


def test_payload_codes_trim_multiple_padded_terminal_frames():
    model = _model(num_stacked_codebooks=4)
    info = {
        "codes": {
            "audio": torch.tensor(
                [
                    [1, 2, 101, 102],
                    [3, 1025, 103, 1025],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ]
            )
        }
    }

    _, frame_counts, valid_sample_counts = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, [info], torch.device("cpu"), [(0, 4)]
    )

    assert frame_counts == [4]
    assert valid_sample_counts == [9]


def test_payload_codes_only_trims_control_from_the_terminal_row():
    model = _model(num_stacked_codebooks=4)
    info = {
        "codes": {
            "audio": torch.tensor(
                [
                    [1, 2, 101, 102],
                    [3, 1025, 103, 1025],
                    [4, 5, 104, 105],
                ]
            )
        }
    }

    _, _, valid_sample_counts = EasyMagpieCodecForConditionalGeneration._payload_codes(
        model, [info], torch.device("cpu"), [(0, 3)]
    )

    assert valid_sample_counts == [18]


class _FakeCodec(torch.nn.Module):
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.arange(codes.shape[0] * 6, dtype=torch.float32, device=codes.device)


@pytest.mark.parametrize(
    ("terminal_row", "expected_samples"),
    [
        ([3, 1025, 103, 1025], 9),
        ([1025, 3, 1025, 103], 6),
    ],
)
def test_forward_trims_terminal_control_subframes(terminal_row, expected_samples):
    model = EasyMagpieCodecForConditionalGeneration.__new__(EasyMagpieCodecForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_stacked_codebooks=4,
        frame_stacking_factor=2,
        codebook_size=1024,
        samples_per_codec_frame=3,
        samples_per_frame=6,
        output_sample_rate=22050,
    )
    model.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    model.codec = _FakeCodec()
    info = {"codes": {"audio": torch.tensor([[1, 2, 101, 102], terminal_row])}}

    output = model.forward(
        input_ids=torch.zeros(2, dtype=torch.long),
        runtime_additional_information=[info],
        request_token_spans=[(0, 2)],
    )

    audio = output.multimodal_outputs["model_outputs"][0]
    torch.testing.assert_close(audio, torch.arange(expected_samples, dtype=torch.float32))


def test_forward_keeps_request_offsets_when_terminal_audio_is_trimmed():
    model = _model(num_stacked_codebooks=4)
    model.config.output_sample_rate = 22050
    model.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    model.codec = _FakeCodec()
    infos = [
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [3, 1025, 103, 1025]])}},
        {"codes": {"audio": torch.tensor([[1, 2, 101, 102], [1025, 3, 1025, 103]])}},
    ]

    output = model.forward(
        input_ids=torch.zeros(4, dtype=torch.long),
        runtime_additional_information=infos,
        request_token_spans=[(0, 2), (2, 4)],
    )

    first, second = output.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(first, torch.arange(9, dtype=torch.float32))
    torch.testing.assert_close(second, torch.arange(12, 18, dtype=torch.float32))
