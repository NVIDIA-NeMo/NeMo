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

import numpy as np
import pytest
import torch

from nemo.collections.llm.fn.activation import openai_gelu, quick_gelu, squared_relu


class TestActivationFunctions:
    """Test suite for activation functions in nemo.collections.llm.fn.activation."""

    @pytest.fixture
    def input_tensor(self):
        """Returns a tensor with test values for activation functions."""
        return torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    @pytest.fixture
    def larger_input_tensor(self):
        """Returns a larger tensor for testing activation functions."""
        return torch.randn(5, 10, dtype=torch.float32)

    def test_openai_gelu(self, input_tensor):
        """Test the openai_gelu activation function."""
        result = openai_gelu(input_tensor)

        # Expected values calculated based on the GELU formula
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        expected = torch.tensor(
            [-0.04540231, -0.15880801, -0.15428599, 0.0, 0.34571401, 0.84119199, 1.95459769], dtype=torch.float32
        )

        assert torch.allclose(result, expected, atol=1e-5)

        # Also verify specific properties of GELU
        assert torch.allclose(result[input_tensor == 0], torch.zeros_like(result[input_tensor == 0]))
        assert torch.all(result[input_tensor > 0] > 0)
        assert torch.all(result[input_tensor < 0] < 0)

    def test_quick_gelu(self, input_tensor):
        """Test the quick_gelu activation function."""
        result = quick_gelu(input_tensor)

        # Expected values calculated based on quick_gelu formula
        # quick_gelu(x) = x * sigmoid(1.702 * x)
        expected = torch.tensor(
            [-0.06434138, -0.15420423, -0.14961156, 0.0, 0.35038844, 0.84579577, 1.93565862], dtype=torch.float32
        )

        assert torch.allclose(result, expected, atol=1e-5)

        # Also verify properties of quick_gelu
        assert torch.allclose(result[input_tensor == 0], torch.zeros_like(result[input_tensor == 0]))
        assert torch.all(result[input_tensor > 0] > 0)
        assert torch.all(result[input_tensor < 0] < 0)

    def test_squared_relu(self, input_tensor):
        """Test the squared_relu activation function."""
        result = squared_relu(input_tensor)

        # Expected values: for x <= 0, output is 0; for x > 0, output is x^2
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 4.0], dtype=torch.float32)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_activation_shapes(self, larger_input_tensor):
        """Test that all activation functions preserve input tensor shape."""
        gelu_output = openai_gelu(larger_input_tensor)
        quick_gelu_output = quick_gelu(larger_input_tensor)
        squared_relu_output = squared_relu(larger_input_tensor)

        assert gelu_output.shape == larger_input_tensor.shape
        assert quick_gelu_output.shape == larger_input_tensor.shape
        assert squared_relu_output.shape == larger_input_tensor.shape

    def test_gelu_implementation_equivalence(self):
        """Test that openai_gelu is close to the mathematical definition of GELU."""
        x = torch.linspace(-3, 3, 100)

        # Implementation directly from the paper
        def paper_gelu(x):
            return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

        expected = paper_gelu(x)
        result = openai_gelu(x)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_squared_relu_0(self):
        """Test that squared_relu of 0 is 0."""
        x = torch.tensor([0.0], dtype=torch.float32)
        result = squared_relu(x)
        expected = torch.tensor([0.0], dtype=torch.float32)

        assert torch.allclose(result, expected)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the activation functions."""
        # Explicitly enable gradient computation
        with torch.enable_grad():
            x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, requires_grad=True)

            # Test openai_gelu
            y_gelu = openai_gelu(x)
            loss_gelu = y_gelu.sum()
            loss_gelu.backward()
            assert x.grad is not None and not torch.allclose(x.grad, torch.zeros_like(x))

            # Reset gradients
            x.grad.zero_()

            # Test quick_gelu
            y_quick = quick_gelu(x)
            loss_quick = y_quick.sum()
            loss_quick.backward()
            assert x.grad is not None and not torch.allclose(x.grad, torch.zeros_like(x))

            # Reset gradients
            x.grad.zero_()

            # Test squared_relu
            y_squared = squared_relu(x)
            loss_squared = y_squared.sum()
            loss_squared.backward()
            assert x.grad is not None and not torch.allclose(x.grad, torch.zeros_like(x))
