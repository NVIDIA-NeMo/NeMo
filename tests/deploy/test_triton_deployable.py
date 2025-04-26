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
from nemo.deploy.triton_deployable import ITritonDeployable


class MockTritonDeployable(ITritonDeployable):
    def __init__(self):
        self.input_shape = (1, 10)
        self.output_shape = (1, 5)

    def get_triton_input(self):
        return {"input": {"shape": self.input_shape, "dtype": np.float32}}

    def get_triton_output(self):
        return {"output": {"shape": self.output_shape, "dtype": np.float32}}

    def triton_infer_fn(self, **inputs: np.ndarray):
        input_data = inputs["input"]
        return {"output": np.ones(self.output_shape) * np.mean(input_data)}


@pytest.fixture
def mock_deployable():
    return MockTritonDeployable()


def test_get_triton_input(mock_deployable):
    """Test that get_triton_input returns the correct input specification."""
    input_spec = mock_deployable.get_triton_input()

    assert "input" in input_spec
    assert input_spec["input"]["shape"] == (1, 10)
    assert input_spec["input"]["dtype"] == np.float32


def test_get_triton_output(mock_deployable):
    """Test that get_triton_output returns the correct output specification."""
    output_spec = mock_deployable.get_triton_output()

    assert "output" in output_spec
    assert output_spec["output"]["shape"] == (1, 5)
    assert output_spec["output"]["dtype"] == np.float32


def test_triton_infer_fn(mock_deployable):
    """Test that triton_infer_fn processes inputs correctly."""
    # Create test input
    test_input = np.random.rand(1, 10).astype(np.float32)
    input_mean = np.mean(test_input)

    # Run inference
    result = mock_deployable.triton_infer_fn(input=test_input)

    # Check output
    assert "output" in result
    assert result["output"].shape == (1, 5)
    assert np.allclose(result["output"], input_mean)


def test_abstract_class_instantiation():
    """Test that ITritonDeployable cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ITritonDeployable()
