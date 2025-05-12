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

from nemo.collections.llm.fn.activation import squared_relu
from nemo.collections.llm.gpt.model.nemotron import (
    Nemotron3Config4B,
    Nemotron3Config8B,
    Nemotron3Config22B,
    Nemotron4Config15B,
    Nemotron4Config340B,
    NemotronConfig,
)


def test_nemotron_config():
    config = NemotronConfig()
    assert config.normalization == "LayerNorm"
    assert config.activation_func == squared_relu
    assert config.position_embedding_type == "rope"
    assert config.share_embeddings_and_output_weights is False
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.rotary_percent == 0.5
    assert config.masked_softmax_fusion is True
    assert config.persist_layer_norm is True
    assert config.bias_dropout_add_fusion is False
    assert config.layernorm_zero_centered_gamma is True

    assert config.num_layers == 32
    assert config.seq_length == 4096
    assert config.hidden_size == 3072
    assert config.ffn_hidden_size == 9216
    assert config.num_attention_heads == 24
    assert config.num_query_groups == 8
    assert config.kv_channels == 128
    assert config.init_method_std == 0.0134


def test_nemotron3_config_4b():
    config = Nemotron3Config4B()
    assert config.num_layers == 32
    assert config.seq_length == 4096
    assert config.hidden_size == 3072
    assert config.ffn_hidden_size == 9216
    assert config.num_attention_heads == 24
    assert config.num_query_groups == 8
    assert config.kv_channels == 128
    assert config.init_method_std == 0.0134


def test_nemotron3_config_8b():
    config = Nemotron3Config8B()
    assert config.num_layers == 32
    assert config.seq_length == 4096
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 16384
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 32
    assert config.kv_channels == 4096 // 32
    assert config.init_method_std == 0.010


def test_nemotron3_config_22b():
    config = Nemotron3Config22B()
    assert config.num_layers == 40
    assert config.seq_length == 4096
    assert config.hidden_size == 6144
    assert config.ffn_hidden_size == 24576
    assert config.num_attention_heads == 48
    assert config.num_query_groups == 48
    assert config.kv_channels == 6144 // 48
    assert config.init_method_std == 0.008


def test_nemotron4_config_15b():
    config = Nemotron4Config15B()
    assert config.num_layers == 32
    assert config.seq_length == 4096
    assert config.hidden_size == 6144
    assert config.ffn_hidden_size == 24576
    assert config.num_attention_heads == 48
    assert config.num_query_groups == 8
    assert config.kv_channels == 6144 // 48
    assert config.init_method_std == 0.0134


def test_nemotron4_config_340b():
    config = Nemotron4Config340B()
    assert config.num_layers == 96
    assert config.seq_length == 4096
    assert config.hidden_size == 18432
    assert config.ffn_hidden_size == 73728
    assert config.num_attention_heads == 96
    assert config.num_query_groups == 8
    assert config.kv_channels == 18432 // 96
    assert config.init_method_std == 0.0063
