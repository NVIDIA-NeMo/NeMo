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
Tests for neural type shape mismatch error messages.
"""

import pytest
import torch
import torch.nn as nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LogitsType, NeuralType


class SimpleModule(nn.Module, Serialization, Typing):
    """A minimal module with fixed input neural types for testing."""

    @property
    def input_types(self):
        return {"x": NeuralType(('B', 'D'), LogitsType())}

    @property
    def output_types(self):
        return {"y": NeuralType(('B', 'D'), LogitsType())}

    @typecheck()
    def forward(self, x):
        return x


class TestNeuralTypeShapeMismatch:
    """Tests that neural type shape mismatch errors are spelled correctly."""

    def test_input_shape_mismatch_error_message(self):
        """Input shape mismatch error should say 'occurred', not 'occured'."""
        model = SimpleModule()
        # Input is 1D but the module expects 2D (B, D)
        x = torch.randn(4)
        with pytest.raises(TypeError, match="Input shape mismatch occurred"):
            model(x=x)

    def test_output_shape_mismatch_error_message(self):
        """Output shape mismatch error should say 'occurred', not 'occured'."""

        class OutputMismatchModule(nn.Module, Serialization, Typing):
            @property
            def input_types(self):
                return {"x": NeuralType(('B', 'D'), LogitsType())}

            @property
            def output_types(self):
                return {"y": NeuralType(('B', 'D'), LogitsType())}

            @typecheck()
            def forward(self, x):
                return x.flatten()  # wrong shape

        model = OutputMismatchModule()
        x = torch.randn(4, 8)
        with pytest.raises(TypeError, match="Output shape mismatch occurred"):
            model(x=x)
