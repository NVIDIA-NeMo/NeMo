# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Tests for exp_manager timer callback defaults.
"""

import inspect
from unittest.mock import patch

from nemo.utils.exp_manager import DeltaTimingCallback, TimingCallback


class TestTimerCallbackDefaults:
    """Tests that timer callbacks avoid mutable default arguments."""

    def test_timing_callback_passes_fresh_empty_kwargs_by_default(self):
        """Each TimingCallback should receive its own empty timer_kwargs dict."""
        with patch('nemo.utils.exp_manager.timers.NamedTimer') as mock_timer:
            TimingCallback()
            TimingCallback()

        kwargs1 = mock_timer.call_args_list[0].kwargs
        kwargs2 = mock_timer.call_args_list[1].kwargs
        assert kwargs1 == {}
        assert kwargs2 == {}
        assert kwargs1 is not kwargs2

    def test_delta_timing_callback_default_timer_kwargs_is_none(self):
        """DeltaTimingCallback should use None as the default timer_kwargs to avoid shared mutable state."""
        sig = inspect.signature(DeltaTimingCallback.__init__)
        assert sig.parameters['timer_kwargs'].default is None
