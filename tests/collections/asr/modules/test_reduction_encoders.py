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
Tests for transformer reduction encoders.
"""

import pytest

from nemo.collections.asr.modules.transformer.reduction_encoders import PoolingEncoder


class TestPoolingEncoder:
    """Tests for PoolingEncoder."""

    def test_hidden_steps_less_than_two_error_message(self):
        """An invalid hidden_steps value should be reported in the error message."""
        with pytest.raises(ValueError, match="Expected hidden_steps >= 2 but received hidden_steps = 1"):
            PoolingEncoder(num_layers=1, hidden_size=4, inner_size=8, hidden_steps=1)
