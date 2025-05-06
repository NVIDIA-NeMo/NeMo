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


import pytest
import torch


@pytest.mark.run_only_on('GPU')
def test_replace_number_add_offset():
    from nemo.export.utils.lora_converter import replace_number_add_offset

    # Test with no offset
    key = "layers.0.self_attention.lora_kqv_adapter.linear_in.weight"
    assert replace_number_add_offset(key, 0) == key

    # Test with positive offset
    assert replace_number_add_offset(key, 1) == "layers.1.self_attention.lora_kqv_adapter.linear_in.weight"

    # Test with negative offset
    assert replace_number_add_offset(key, -1) == "layers.-1.self_attention.lora_kqv_adapter.linear_in.weight"

    # Test with key that doesn't contain layer number
    key = "embedding.word_embeddings.weight"
    assert replace_number_add_offset(key, 1) == key


@pytest.mark.run_only_on('GPU')
def test_rename_qkv_keys():
    from nemo.export.utils.lora_converter import rename_qkv_keys

    key = "layers.0.self_attention.lora_kqv_adapter.linear_in.weight"
    new_keys = rename_qkv_keys(key)

    assert len(new_keys) == 3
    assert new_keys[0] == "layers.0.self_attention.lora_unfused_kqv_adapter.q_adapter.linear_in.weight"
    assert new_keys[1] == "layers.0.self_attention.lora_unfused_kqv_adapter.k_adapter.linear_in.weight"
    assert new_keys[2] == "layers.0.self_attention.lora_unfused_kqv_adapter.v_adapter.linear_in.weight"


@pytest.mark.run_only_on('GPU')
def test_reformat_module_names_to_hf():
    from nemo.export.utils.lora_converter import reformat_module_names_to_hf

    # Create sample tensors with NeMo-style names
    tensors = {
        "q_adapter.linear_in.weight": torch.randn(10, 10),
        "k_adapter.linear_out.weight": torch.randn(10, 10),
        "v_adapter.linear_in.weight": torch.randn(10, 10),
        "lora_dense_attention_adapter.linear_out.weight": torch.randn(10, 10),
        "lora_4htoh_adapter.linear_in.weight": torch.randn(10, 10),
        "gate_adapter.linear_out.weight": torch.randn(10, 10),
        "up_adapter.linear_in.weight": torch.randn(10, 10),
    }

    new_tensors, module_names = reformat_module_names_to_hf(tensors)

    # Check that all tensors were converted
    assert len(new_tensors) == len(tensors)

    # Check that module names were correctly identified
    expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"]
    assert set(module_names) == set(expected_modules)

    # Check some specific conversions
    assert "base_model.q_proj.lora_A.weight" in new_tensors
    assert "base_model.k_proj.lora_B.weight" in new_tensors
    assert "base_model.v_proj.lora_A.weight" in new_tensors


@pytest.mark.run_only_on('GPU')
def test_convert_lora_weights_to_canonical():
    from nemo.export.utils.lora_converter import convert_lora_weights_to_canonical

    # Create a sample config
    config = {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_query_groups": 4,
        "peft": {"lora_tuning": {"adapter_dim": 16}},
    }

    # Create sample fused QKV weights
    lora_weights = {
        "layers.0.self_attention.lora_kqv_adapter.linear_in.weight": torch.randn(16, 1024),
        "layers.0.self_attention.lora_kqv_adapter.linear_out.weight": torch.randn(1024, 16),
        "layers.0.lora_hto4h_adapter.linear_in.weight": torch.randn(16, 1024),
        "layers.0.lora_hto4h_adapter.linear_out.weight": torch.randn(2048, 16),
    }

    converted_weights = convert_lora_weights_to_canonical(config, lora_weights)

    # Check that QKV weights were unfused
    assert "layers.0.self_attention.lora_unfused_kqv_adapter.q_adapter.linear_in.weight" in converted_weights
    assert "layers.0.self_attention.lora_unfused_kqv_adapter.k_adapter.linear_in.weight" in converted_weights
    assert "layers.0.self_attention.lora_unfused_kqv_adapter.v_adapter.linear_in.weight" in converted_weights

    # Check that H-to-4H weights were unfused
    assert "layers.0.lora_unfused_hto4h_adapter.gate_adapter.linear_in.weight" in converted_weights
    assert "layers.0.lora_unfused_hto4h_adapter.up_adapter.linear_in.weight" in converted_weights
