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


@pytest.mark.run_only_on('GPU')
def test_rename_key():
    # Test basic self_attention replacement
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import rename_key

    assert rename_key("self_attention.weight") == "attention.weight"

    # Test layernorm replacements
    assert rename_key("attention.linear_qkv.layer_norm_weight") == "input_layernorm.weight"
    assert rename_key("attention.linear_qkv.layer_norm_bias") == "input_layernorm.bias"
    assert rename_key("mlp.linear_fc1.layer_norm_weight") == "post_attention_layernorm.weight"
    assert rename_key("mlp.linear_fc1.layer_norm_bias") == "post_attention_layernorm.bias"

    # Test key with no replacements needed
    assert rename_key("some_other_key") == "some_other_key"


@pytest.mark.run_only_on('GPU')
def test_rename_key_dist_ckpt():
    # Test key with layers
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import rename_key_dist_ckpt

    assert rename_key_dist_ckpt("layers.linear_qkv.weight", 0) == "layers.0.linear_qkv.weight"
    assert rename_key_dist_ckpt("layers.self_attention.weight", 1) == "layers.1.attention.weight"

    # Test key without layers
    assert rename_key_dist_ckpt("embedding.weight", 0) == "embedding.weight"


@pytest.mark.run_only_on('GPU')
def test_get_layer_prefix():
    # Test for mcore model
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import get_layer_prefix

    layer_names_mcore = [
        "model.decoder.layers.0.self_attention.weight",
        "optimizer.state",
        "model.decoder.layers.1.self_attention.bias",
    ]
    model_prefix, transformer_prefix = get_layer_prefix(layer_names_mcore, is_mcore=True)
    assert model_prefix == "model."
    assert transformer_prefix == "model.decoder."

    # Test for non-mcore model
    layer_names_non_mcore = [
        "model.encoder.layers.0.self_attention.weight",
        "optimizer.state",
        "model.encoder.layers.1.self_attention.bias",
    ]
    model_prefix, transformer_prefix = get_layer_prefix(layer_names_non_mcore, is_mcore=False)
    assert model_prefix == "model."
    assert transformer_prefix == "model.encoder."


@pytest.mark.run_only_on('GPU')
def test_rename_layer_num():
    # Test basic layer number replacement
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import rename_layer_num

    assert rename_layer_num("model.layers.0.attention.weight", 1) == "model.layers.1.attention.weight"
    assert rename_layer_num("decoder.layers.5.mlp.weight", 2) == "decoder.layers.2.mlp.weight"

    # Test with multiple numeric components
    assert rename_layer_num("model.layers.0.attention.head.8.weight", 3) == "model.layers.3.attention.head.8.weight"


@pytest.mark.run_only_on('GPU')
def test_get_layer_num():
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import get_layer_num

    assert get_layer_num("model.layers.0.attention.weight") == 0
    assert get_layer_num("decoder.layers.5.mlp.weight") == 5

    with pytest.raises(ValueError):
        get_layer_num("model.attention.weight")  # No layers component


@pytest.mark.run_only_on('GPU')
def test_is_scaling_factor():
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import is_scaling_factor

    assert is_scaling_factor("layer.extra_state.weight") == True
    assert is_scaling_factor("layer.weight") == False
    assert is_scaling_factor("extra_state") == True


@pytest.mark.run_only_on('GPU')
def test_create_export_dir(tmp_path):
    from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import create_export_dir

    # Test creating new directory
    export_dir = tmp_path / "new_export_dir"
    created_dir = create_export_dir(export_dir)
    assert created_dir.exists()
    assert created_dir.is_dir()

    # Test with existing directory
    existing_dir = create_export_dir(export_dir)
    assert existing_dir == export_dir
