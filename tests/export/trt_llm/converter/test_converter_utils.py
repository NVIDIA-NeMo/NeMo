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


@pytest.mark.run_only_on('GPU')
def test_any_word_in_key():
    # Test positive cases
    from nemo.export.trt_llm.converter.utils import any_word_in_key

    assert any_word_in_key("model.layer1.attention.dense.weight", ["attention", "mlp"]) == True
    assert any_word_in_key("model.layer1.mlp.weight", ["attention", "mlp"]) == True

    # Test negative cases
    assert any_word_in_key("model.layer1.other.weight", ["attention", "mlp"]) == False
    assert any_word_in_key("", ["attention", "mlp"]) == False


@pytest.mark.run_only_on('GPU')
def test_get_trt_llm_keyname():
    # Test final layernorm case
    from nemo.export.trt_llm.converter.utils import get_trt_llm_keyname

    assert get_trt_llm_keyname("final_layernorm.weight") == "transformer.ln_f.weight"

    # Test layer cases
    assert get_trt_llm_keyname("layers.1.attention.dense.weight") == "transformer.layers.1.attention.dense.weight"
    assert get_trt_llm_keyname("layers.2.mlp.linear_fc2.weight") == "transformer.layers.2.mlp.proj.weight"


@pytest.mark.run_only_on('GPU')
def test_is_scaling_factor():
    from nemo.export.trt_llm.converter.utils import is_scaling_factor

    assert is_scaling_factor("model.layer1.scale_fwd.weight") == True
    assert is_scaling_factor("model.layer1.weight") == False
    assert is_scaling_factor("") == False


@pytest.mark.run_only_on('GPU')
def test_get_scaling_factor_keys():
    from nemo.export.trt_llm.converter.utils import get_scaling_factor_keys

    key = "layers.1.mlp.dense_h_to_4h.scale_fwd"
    keys, gate_keys = get_scaling_factor_keys(key)

    # Check main keys
    assert keys[0].endswith(".weights_scaling_factor")
    assert keys[1].endswith(".activation_scaling_factor")

    # Check gate keys
    assert gate_keys[0].endswith(".activation_scaling_factor")
    assert gate_keys[1].endswith(".weights_scaling_factor")


@pytest.mark.run_only_on('GPU')
def test_split():
    # Test numpy array splitting
    from nemo.export.trt_llm.converter.utils import split

    arr = np.array([1, 2, 3, 4])
    assert np.array_equal(split(arr, tp_size=2, idx=0), np.array([1, 2]))
    assert np.array_equal(split(arr, tp_size=2, idx=1), np.array([3, 4]))

    # Test torch tensor splitting
    tensor = torch.tensor([1, 2, 3, 4])
    assert torch.equal(split(tensor, tp_size=2, idx=0), torch.tensor([1, 2]))
    assert torch.equal(split(tensor, tp_size=2, idx=1), torch.tensor([3, 4]))

    # Test no split case
    assert np.array_equal(split(arr, tp_size=1, idx=0), arr)


@pytest.mark.run_only_on('GPU')
def test_generate_int8():
    # Create test weights and activation ranges
    from nemo.export.trt_llm.converter.utils import generate_int8

    weights = np.random.randn(4, 4).astype(np.float32)
    act_range = {"w": torch.tensor(2.0), "x": torch.tensor(3.0), "y": torch.tensor(4.0)}

    result = generate_int8(weights, act_range)

    # Check that all expected keys are present
    expected_keys = [
        "weight.int8",
        "weight.int8.col",
        "scale_x_orig_quant",
        "scale_w_quant_orig",
        "scale_w_quant_orig.col",
        "scale_y_accum_quant",
        "scale_y_accum_quant.col",
        "scale_y_quant_orig",
    ]
    assert all(key in result for key in expected_keys)

    # Check that int8 weights are in correct range
    assert np.all(result["weight.int8"] >= -127)
    assert np.all(result["weight.int8"] <= 127)
    assert np.all(result["weight.int8.col"] >= -127)
    assert np.all(result["weight.int8.col"] <= 127)


if __name__ == "__main__":
    pytest.main([__file__])
