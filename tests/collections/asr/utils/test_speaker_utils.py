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
Tests for speaker utility functions.
"""

import pytest
import torch

from nemo.collections.asr.parts.utils.speaker_utils import check_ranges


class TestCheckRanges:
    """Tests for check_ranges."""

    def test_invalid_range_error_message_includes_range(self):
        """An invalid range should be reported in the error message."""
        range_tensor = torch.tensor([[1.0, 0.5]])
        with pytest.raises(ValueError, match="Range start time should be preceding the end time but we got:"):
            check_ranges(range_tensor)
