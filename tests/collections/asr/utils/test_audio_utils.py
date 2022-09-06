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

from typing import List, Type, Union

import numpy as np
import pytest

from nemo.collections.asr.parts.utils.audio_utils import select_channels


class TestSelectChannels:
    num_samples = 1000
    max_diff_tol = 1e-9

    @pytest.mark.unit
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, 1, [0, 1]])
    def test_single_channel_input(self, channel_selector: Type[Union[str, int, List[int]]]):
        """Cover the case with single-channel input signal.
        Channel selector should not do anything in this case.
        """
        golden_out = signal_in = np.random.rand(self.num_samples)

        if channel_selector not in [None, 0, 'average']:
            # Expect a failure if looking for a different channel when input is 1D
            with pytest.raises(ValueError):
                # UUT
                signal_out = select_channels(signal_in, channel_selector)
        else:
            # UUT
            signal_out = select_channels(signal_in, channel_selector)

            # Check difference
            max_diff = np.max(np.abs(signal_out - golden_out))
            assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [2, 4])
    @pytest.mark.parametrize("channel_selector", [None, 'average', 0, [1], [0, 1]])
    def test_multi_channel_input(self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]):
        """Cover the case with multi-channel input signal and single-
        or multi-channel output.
        """
        num_samples = 1000
        signal_in = np.random.rand(self.num_samples, num_channels)

        # calculate golden output
        if channel_selector is None:
            golden_out = signal_in
        elif channel_selector == 'average':
            golden_out = np.mean(signal_in, axis=1)
        else:
            golden_out = signal_in[:, channel_selector].squeeze()

        # UUT
        signal_out = select_channels(signal_in, channel_selector)

        # Check difference
        max_diff = np.max(np.abs(signal_out - golden_out))
        assert max_diff < self.max_diff_tol

    @pytest.mark.unit
    @pytest.mark.parametrize("num_channels", [1, 2])
    @pytest.mark.parametrize("channel_selector", [2, [1, 2]])
    def test_select_more_channels_than_available(
        self, num_channels: int, channel_selector: Type[Union[str, int, List[int]]]
    ):
        """This test is expecting the UUT to fail because we ask for more channels
        than available in the input signal.
        """
        num_samples = 1000
        signal_in = np.random.rand(self.num_samples, num_channels)

        # expect failure since we ask for more channels than available
        with pytest.raises(ValueError):
            # UUT
            signal_out = select_channels(signal_in, channel_selector)
