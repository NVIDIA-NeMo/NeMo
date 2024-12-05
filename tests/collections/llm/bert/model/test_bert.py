# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.bert.model.bert import (
    HuggingFaceBertBaseConfig,
    HuggingFaceBertLargeConfig,
    MegatronBertBaseConfig,
    MegatronBertLargeConfig,
)


def test_huggingface_bert_base_config():
    config = HuggingFaceBertBaseConfig()
    assert config.bert_type == 'huggingface'
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12


def test_huggingface_bert_large_config():
    config = HuggingFaceBertLargeConfig()
    assert config.bert_type == 'huggingface'
    assert config.num_layers == 24
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 4096
    assert config.num_attention_heads == 16


def test_megatron_bert_base_config():
    config = MegatronBertBaseConfig()
    assert config.bert_type == 'megatron'
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12


def test_megatron_bert_large_config():
    config = MegatronBertLargeConfig()
    assert config.bert_type == 'megatron'
    assert config.num_layers == 24
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 4096
    assert config.num_attention_heads == 16
