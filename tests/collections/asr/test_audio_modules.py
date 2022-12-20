# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
from typing import Optional

import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.audio_modules import MaskReferenceChannel, SpectrogramToMultichannelFeatures
from nemo.collections.asr.modules.audio_preprocessing import AudioToSpectrogram

try:
    importlib.import_module('torchaudio')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


class TestSpectrogramToMultichannelFeatures:
    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [256])
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('mag_reduction', [None, 'rms', 'abs_mean', 'mean_abs'])
    def test_magnitude(self, fft_length: int, num_channels: int, mag_reduction: Optional[str]):
        """Test calculation of spatial features for multi-channel audio.
        """
        atol = 1e-6
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 25
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

        spec2feat = SpectrogramToMultichannelFeatures(
            num_subbands=audio2spec.num_subbands, mag_reduction=mag_reduction, use_ipd=False, mag_normalization=None,
        )

        for n in range(num_examples):
            x = _rng.normal(size=(batch_size, num_channels, num_samples))

            spec, spec_len = audio2spec(input=torch.Tensor(x), input_length=torch.Tensor([num_samples] * batch_size))

            # UUT output
            feat, _ = spec2feat(input=spec, input_length=spec_len)
            feat_np = feat.cpu().detach().numpy()

            # Golden output
            spec_np = spec.cpu().detach().numpy()
            if mag_reduction is None:
                feat_golden = np.abs(spec_np)
            elif mag_reduction == 'rms':
                feat_golden = np.sqrt(np.mean(np.abs(spec_np) ** 2, axis=1, keepdims=True))
            elif mag_reduction == 'mean_abs':
                feat_golden = np.mean(np.abs(spec_np), axis=1, keepdims=True)
            elif mag_reduction == 'abs_mean':
                feat_golden = np.abs(np.mean(spec_np, axis=1, keepdims=True))
            else:
                raise NotImplementedError()

            # Compare shape
            assert feat_np.shape == feat_golden.shape, f'Feature shape not matching for example {n}'

            # Compare values
            assert np.allclose(feat_np, feat_golden, atol=atol), f'Features not matching for example {n}'

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [256])
    @pytest.mark.parametrize('num_channels', [1, 4])
    def test_ipd(self, fft_length: int, num_channels: int):
        """Test calculation of IPD spatial features for multi-channel audio.
        """
        atol = 1e-5
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 10
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

        spec2feat = SpectrogramToMultichannelFeatures(
            num_subbands=audio2spec.num_subbands,
            mag_reduction='rms',
            use_ipd=True,
            mag_normalization=None,
            ipd_normalization=None,
        )

        for n in range(num_examples):
            x = _rng.normal(size=(batch_size, num_channels, num_samples))

            spec, spec_len = audio2spec(input=torch.Tensor(x), input_length=torch.Tensor([num_samples] * batch_size))

            # UUT output
            feat, _ = spec2feat(input=spec, input_length=spec_len)
            feat_np = feat.cpu().detach().numpy()
            ipd = feat_np[..., audio2spec.num_subbands :, :]

            # Golden output
            spec_np = spec.cpu().detach().numpy()
            spec_mean = np.mean(spec_np, axis=1, keepdims=True)
            ipd_golden = np.angle(spec_np) - np.angle(spec_mean)
            ipd_golden = np.remainder(ipd_golden + np.pi, 2 * np.pi) - np.pi

            # Compare shape
            assert ipd.shape == ipd_golden.shape, f'Feature shape not matching for example {n}'

            # Compare values
            assert np.allclose(ipd, ipd_golden, atol=atol), f'Features not matching for example {n}'


class TestMaskBasedProcessor:
    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize('fft_length', [256])
    @pytest.mark.parametrize('num_channels', [1, 4])
    @pytest.mark.parametrize('num_masks', [1, 2])
    def test_mask_reference_channel(self, fft_length: int, num_channels: int, num_masks: int):
        """Test masking of the reference channel.
        """
        if num_channels == 1:
            # Only one channel available
            ref_channels = [0]
        else:
            # Use first or last channel for MC signals
            ref_channels = [0, num_channels - 1]

        atol = 1e-6
        batch_size = 8
        num_samples = fft_length * 50
        num_examples = 10
        random_seed = 42

        _rng = np.random.default_rng(seed=random_seed)

        hop_length = fft_length // 4
        audio2spec = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length)

        for ref_channel in ref_channels:

            mask_processor = MaskReferenceChannel(ref_channel=ref_channel)

            for n in range(num_examples):
                x = _rng.normal(size=(batch_size, num_channels, num_samples))

                spec, spec_len = audio2spec(
                    input=torch.Tensor(x), input_length=torch.Tensor([num_samples] * batch_size)
                )

                # Randomly-generated mask
                mask = _rng.uniform(
                    low=0.0, high=1.0, size=(batch_size, num_masks, audio2spec.num_subbands, spec.shape[-1])
                )

                # UUT output
                out, _ = mask_processor(input=spec, input_length=spec_len, mask=torch.tensor(mask))
                out_np = out.cpu().detach().numpy()

                # Golden output
                spec_np = spec.cpu().detach().numpy()
                out_golden = np.zeros_like(mask, dtype=spec_np.dtype)
                for m in range(num_masks):
                    out_golden[:, m, ...] = spec_np[:, ref_channel, ...] * mask[:, m, ...]

                # Compare shape
                assert out_np.shape == out_golden.shape, f'Output shape not matching for example {n}'

                # Compare values
                assert np.allclose(out_np, out_golden, atol=atol), f'Output not matching for example {n}'
