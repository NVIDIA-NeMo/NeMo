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
def test_determine_quantization_settings():
    # Test with default NeMo config (no fp8)
    from nemo.export.trt_llm.converter.model_converter import determine_quantization_settings

    nemo_config = {'fp8': False}
    fp8_quant, fp8_kv = determine_quantization_settings(nemo_config)
    assert not fp8_quant
    assert not fp8_kv

    # Test with NeMo config having fp8=True
    nemo_config = {'fp8': True}
    fp8_quant, fp8_kv = determine_quantization_settings(nemo_config)
    assert fp8_quant
    assert fp8_kv

    # Test with override parameters
    fp8_quant, fp8_kv = determine_quantization_settings(nemo_config, fp8_quantized=False, fp8_kvcache=True)
    assert not fp8_quant
    assert fp8_kv


@pytest.mark.run_only_on('GPU')
def test_prompt_convert_task_templates():
    # Test with task templates
    from nemo.export.trt_llm.converter.model_converter import prompt_convert

    prompt_config = {
        'task_templates': [
            {'taskname': 'task1'},
            {'taskname': 'task2'},
        ]
    }

    # Create mock weights
    prompt_weights = {
        'prompt_table': {
            'prompt_table.task1.prompt_embeddings.weight': torch.ones(2, 4),
            'prompt_table.task2.prompt_embeddings.weight': torch.ones(3, 4),
        }
    }

    result = prompt_convert(prompt_config, prompt_weights)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3, 4)  # (num_tasks, max_length, embedding_dim)


@pytest.mark.run_only_on('GPU')
def test_prompt_convert_direct_embeddings():
    # Test with direct embeddings
    from nemo.export.trt_llm.converter.model_converter import prompt_convert

    prompt_config = {}
    prompt_weights = {'prompt_embeddings_weights': torch.ones(2, 3, 4)}

    result = prompt_convert(prompt_config, prompt_weights)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3, 4)
