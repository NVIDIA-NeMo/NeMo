# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from unittest import mock

import pytest
import torch
from easymagpie_vllm_omni import watermark


class _FakeAudioProcessor:
    def signal_to_magphase(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return signal, torch.zeros_like(signal)

    def magphase_to_signal(self, magnitude: torch.Tensor, _phase: torch.Tensor) -> torch.Tensor:
        return magnitude


class _FakeEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, magnitude: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.batch_sizes.append(int(magnitude.shape[0]))
        return magnitude + 0.125, None


class _FakePerthNet:
    def __init__(self) -> None:
        self.hp = types.SimpleNamespace(sample_rate=22_050, n_fft=8)
        self.device = torch.device("cpu")
        self.ap = _FakeAudioProcessor()
        self.encoder = _FakeEncoder()


def setup_function() -> None:
    watermark._WATERMARKER = None
    watermark._INITIALIZED_DEVICE = None
    watermark._SHORT_AUDIO_WARNING_EMITTED = False


def test_unindexed_cuda_device_is_normalized() -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=2),
    ):
        assert watermark._normalized_device("cuda") == "cuda:2"


def test_equal_length_waveforms_are_batched_and_length_is_preserved() -> None:
    perth_net = _FakePerthNet()
    watermarker = types.SimpleNamespace(perth_net=perth_net)
    waveforms = [torch.zeros(16), torch.full((16,), 0.5), torch.zeros(12)]

    with mock.patch.object(watermark, "initialize_watermarker", return_value=watermarker):
        result = watermark.watermark_waveforms(
            waveforms,
            sample_rate=22_050,
            device="cpu",
        )

    assert perth_net.encoder.batch_sizes == [2, 1]
    assert [tensor.numel() for tensor in result] == [16, 16, 12]
    torch.testing.assert_close(result[0], torch.full((16,), 0.125))
    torch.testing.assert_close(result[1], torch.full((16,), 0.625))


def test_short_waveform_is_left_unchanged() -> None:
    perth_net = _FakePerthNet()
    watermarker = types.SimpleNamespace(perth_net=perth_net)
    original = torch.tensor([0.1, -0.2, 0.3, -0.4])

    with mock.patch.object(watermark, "initialize_watermarker", return_value=watermarker):
        result = watermark.watermark_waveforms(
            [original.clone()],
            sample_rate=22_050,
            device="cpu",
        )

    assert perth_net.encoder.batch_sizes == []
    torch.testing.assert_close(result[0], original)


def test_explicit_disable_bypasses_initialization() -> None:
    waveforms = [torch.zeros(16)]
    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "off"}),
        mock.patch.object(watermark, "initialize_watermarker") as initialize,
    ):
        result = watermark.watermark_waveforms(
            waveforms,
            sample_rate=22_050,
            device="cpu",
        )

    initialize.assert_not_called()
    assert result is waveforms


def test_enabled_initialization_failure_is_fatal() -> None:
    broken_perth = types.SimpleNamespace(
        PerthImplicitWatermarker=mock.Mock(side_effect=ValueError("bad checkpoint"))
    )
    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.dict("sys.modules", {"perth": broken_perth}),
        pytest.raises(RuntimeError, match="could not be initialized"),
    ):
        watermark.initialize_watermarker("cpu")


def test_encoder_is_compiled_on_cuda() -> None:
    encoder = object()
    perth_net = types.SimpleNamespace(
        hp=types.SimpleNamespace(sample_rate=32_000, n_fft=8),
        encoder=encoder,
        device="cuda:0",
        ap=_FakeAudioProcessor(),
    )
    watermarker = types.SimpleNamespace(perth_net=perth_net)
    compiled = mock.Mock(name="compiled_encoder", side_effect=lambda mag: (mag, None))
    fake_perth = types.SimpleNamespace(
        PerthImplicitWatermarker=mock.Mock(return_value=watermarker)
    )

    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.dict("sys.modules", {"perth": fake_perth}),
        mock.patch.object(watermark, "_normalized_device", return_value="cuda:0"),
        mock.patch.object(torch, "compile", return_value=compiled) as compile_fn,
        mock.patch.object(torch.cuda, "is_available", return_value=False),
    ):
        result = watermark.initialize_watermarker("cuda:0")

    compile_fn.assert_called_once_with(encoder, mode="default", dynamic=True)
    assert result.perth_net.encoder is compiled


def test_encoder_is_not_compiled_on_cpu() -> None:
    encoder = object()
    perth_net = types.SimpleNamespace(
        hp=types.SimpleNamespace(sample_rate=32_000, n_fft=8),
        encoder=encoder,
        device="cpu",
    )
    watermarker = types.SimpleNamespace(perth_net=perth_net)
    fake_perth = types.SimpleNamespace(
        PerthImplicitWatermarker=mock.Mock(return_value=watermarker)
    )

    with (
        mock.patch.dict("os.environ", {"NEMOTRON_TTS_PERTH_WATERMARK": "1"}),
        mock.patch.dict("sys.modules", {"perth": fake_perth}),
        mock.patch.object(torch, "compile") as compile_fn,
    ):
        result = watermark.initialize_watermarker("cpu")

    compile_fn.assert_not_called()
    assert result.perth_net.encoder is encoder
