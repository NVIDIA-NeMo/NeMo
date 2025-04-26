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

from nemo.export.multimodal.converter import split_gate_weight, split_kv_weight, split_qkv_weight


class TestMultimodalConverter:
    @pytest.fixture
    def model_config(self):
        # Create a simple test config
        config = type(
            'TestConfig',
            (),
            {'hidden_size': 128, 'num_attention_heads': 4, 'num_query_groups': 2, 'kv_channels': None},
        )()
        return config

    def test_split_qkv_weight(self, model_config):
        # Create a test QKV weight tensor
        batch_size = model_config.num_attention_heads + 2 * model_config.num_query_groups
        qkv_weight = torch.randn(
            batch_size, model_config.hidden_size // model_config.num_attention_heads, model_config.hidden_size
        )

        result = split_qkv_weight(qkv_weight, model_config)

        assert len(result) == 3
        assert result[0][0] == 'q_proj'
        assert result[1][0] == 'k_proj'
        assert result[2][0] == 'v_proj'

        # Check shapes
        assert result[0][1].shape == (
            model_config.num_attention_heads,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )
        assert result[1][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )
        assert result[2][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )

    def test_split_kv_weight(self, model_config):
        # Create a test KV weight tensor
        batch_size = 2 * model_config.num_query_groups
        kv_weight = torch.randn(
            batch_size, model_config.hidden_size // model_config.num_attention_heads, model_config.hidden_size
        )

        result = split_kv_weight(kv_weight, model_config)

        assert len(result) == 2
        assert result[0][0] == 'k_proj'
        assert result[1][0] == 'v_proj'

        # Check shapes
        assert result[0][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )
        assert result[1][1].shape == (
            model_config.num_query_groups,
            model_config.hidden_size // model_config.num_attention_heads,
            model_config.hidden_size,
        )

    def test_split_gate_weight(self):
        # Create a test gate weight tensor
        gate_weight = torch.randn(200, 100)  # Example dimensions

        result = split_gate_weight(gate_weight)

        assert len(result) == 2
        assert result[0][0] == 'gate_proj'
        assert result[1][0] == 'up_proj'

        # Check shapes
        assert result[0][1].shape == (100, 100)
        assert result[1][1].shape == (100, 100)
