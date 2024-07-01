# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from torchmetrics.audio.snr import SignalNoiseRatio

from nemo.collections.audio.metrics.audio import AudioMetricWrapper


class TestAudioMetricWrapper:
    def test_metric_full_batch(self):
        """Test metric on batches where all examples have equal length."""
        ref_metric = SignalNoiseRatio()
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio())

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target)
                batch_value_ref = ref_metric(preds=preds, target=target)

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()

    def test_input_length(self):
        """Test metric on batches where examples have different length."""
        ref_metric = SignalNoiseRatio()
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio())

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                input_length = torch.randint(low=num_samples // 2, high=num_samples, size=(batch_size,))

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target, input_length=input_length)

                # compute reference value, assuming batch reduction using averaging
                batch_value_ref = 0
                for b_idx, b_len in enumerate(input_length):
                    batch_value_ref += ref_metric(preds=preds[b_idx, ..., :b_len], target=target[b_idx, ..., :b_len])
                batch_value_ref /= batch_size  # average

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()

    @pytest.mark.unit
    @pytest.mark.parametrize('channel', [0, 1])
    def test_channel(self, channel):
        """Test metric on a single channel from a batch."""
        ref_metric = SignalNoiseRatio()
        # select only a single channel
        wrapped_metric = AudioMetricWrapper(metric=SignalNoiseRatio(), channel=channel)

        num_resets = 5
        num_batches = 10
        batch_size = 8
        num_channels = 2
        num_samples = 200

        batch_shape = (batch_size, num_channels, num_samples)

        for nr in range(num_resets):
            for nb in range(num_batches):
                target = torch.rand(*batch_shape)
                preds = target + torch.rand(1) * torch.rand(*batch_shape)

                # varying length
                input_length = torch.randint(low=num_samples // 2, high=num_samples, size=(batch_size,))

                # test forward for a single batch
                batch_value_wrapped = wrapped_metric(preds=preds, target=target, input_length=input_length)

                # compute reference value, assuming batch reduction using averaging
                batch_value_ref = 0
                for b_idx, b_len in enumerate(input_length):
                    batch_value_ref += ref_metric(
                        preds=preds[b_idx, channel, :b_len], target=target[b_idx, channel, :b_len]
                    )
                batch_value_ref /= batch_size  # average

                assert torch.allclose(
                    batch_value_wrapped, batch_value_ref
                ), f'Metric forward not matching for batch {nb}, reset {nr}'

            # test compute (over num_batches)
            assert torch.allclose(
                wrapped_metric.compute(), ref_metric.compute()
            ), f'Metric compute not matching for batch {nb}, reset {nr}'

            ref_metric.reset()
            wrapped_metric.reset()
