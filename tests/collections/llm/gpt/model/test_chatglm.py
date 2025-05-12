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

import torch.nn.functional as F

from nemo.collections.llm.gpt.model.chatglm import ChatGLM2Config6B, ChatGLM3Config6B, ChatGLMConfig


def test_chatglm_config():
    config = ChatGLMConfig()
    assert config.num_layers == 28
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 13696
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 2
    assert config.init_method_std == 0.02
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.normalization == "RMSNorm"
    assert config.add_bias_linear is False
    assert config.add_qkv_bias is True
    assert config.rotary_percent == 0.5
    assert config.rotary_interleaved is True
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.share_embeddings_and_output_weights is False
    assert config.make_vocab_size_divisible_by == 65024


def test_chatglm2_config_6b():
    config = ChatGLM2Config6B()
    assert config.seq_length == 32768


def test_chatglm3_config_6b():
    config = ChatGLM3Config6B()
    assert config.seq_length == 8192
