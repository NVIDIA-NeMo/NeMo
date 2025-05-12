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

import os
import re

import pytest
import torch


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_get_nemo_to_trtllm_conversion_dict_on_nemo_model():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    dummy_state = object()
    model_state_dict = {
        'model.embedding.word_embeddings.weight': dummy_state,
        'model.decoder.layers.0.self_attention.linear_proj.weight': dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(model_state_dict)

    # Check that every key starts with 'model.' and not 'model..' by using a regex
    # This pattern ensures:
    #   - The key starts with 'model.'
    #   - Immediately after 'model.', there must be at least one character that is NOT a '.'
    #     (preventing the 'model..' scenario)
    pattern = re.compile(r'^model\.[^.].*')
    for key in nemo_model_conversion_dict.keys():
        assert pattern.match(key), f"Key '{key}' does not properly start with 'model.'"


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_get_nemo_to_trtllm_conversion_dict_on_mcore_model():
    try:
        from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import DEFAULT_CONVERSION_DICT

        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    dummy_state = object()
    model_state_dict = {
        'embedding.word_embeddings.weight': dummy_state,
        'decoder.layers.0.self_attention.linear_proj.weight': dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(model_state_dict)

    # This is essentially a no-op
    assert nemo_model_conversion_dict == DEFAULT_CONVERSION_DICT


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_tensorrt_llm_initialization():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    # Test basic initialization
    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)
    assert trt_llm.model_dir == model_dir
    assert trt_llm.engine_dir == os.path.join(model_dir, "trtllm_engine")
    assert trt_llm.model is None
    assert trt_llm.tokenizer is None
    assert trt_llm.config is None

    # Test initialization with lora checkpoints
    lora_ckpt_list = ["/path/to/lora1", "/path/to/lora2"]
    trt_llm = TensorRTLLM(model_dir=model_dir, lora_ckpt_list=lora_ckpt_list, load_model=False)
    assert trt_llm.lora_ckpt_list == lora_ckpt_list

    # Test initialization with python runtime options
    trt_llm = TensorRTLLM(
        model_dir=model_dir,
        use_python_runtime=False,
        enable_chunked_context=False,
        max_tokens_in_paged_kv_cache=None,
        load_model=False,
    )
    assert trt_llm.use_python_runtime is False
    assert trt_llm.enable_chunked_context is False
    assert trt_llm.max_tokens_in_paged_kv_cache is None


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_tensorrt_llm_supported_models():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Test supported models list
    supported_models = trt_llm.get_supported_models_list
    assert isinstance(supported_models, list)
    assert len(supported_models) > 0
    assert all(isinstance(model, str) for model in supported_models)

    # Test HF model mapping
    hf_mapping = trt_llm.get_supported_hf_model_mapping
    assert isinstance(hf_mapping, dict)
    assert len(hf_mapping) > 0


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_tensorrt_llm_input_dtype():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    from megatron.core.export.data_type import DataType

    # Test different storage dtypes
    test_cases = [
        (torch.float32, DataType.float32),
        (torch.float16, DataType.float16),
        (torch.bfloat16, DataType.bfloat16),
    ]

    for storage_dtype, expected_dtype in test_cases:
        input_dtype = trt_llm.get_input_dtype(storage_dtype)
        assert input_dtype == expected_dtype, f"Expected {expected_dtype} for {storage_dtype}, got {input_dtype}"


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_tensorrt_llm_hidden_size():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Test hidden size property
    hidden_size = trt_llm.get_hidden_size
    if hidden_size is not None:
        assert isinstance(hidden_size, int)
        assert hidden_size > 0
    else:
        assert hidden_size is None


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_tensorrt_llm_triton_io():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Test Triton input configuration
    triton_input = trt_llm.get_triton_input
    assert isinstance(triton_input, tuple)
    assert triton_input[0].name == "prompts"
    assert triton_input[1].name == "max_output_len"
    assert triton_input[2].name == "top_k"
    assert triton_input[3].name == "top_p"
    assert triton_input[4].name == "temperature"
    assert triton_input[5].name == "random_seed"
    assert triton_input[6].name == "stop_words_list"
    assert triton_input[7].name == "bad_words_list"
    assert triton_input[8].name == "no_repeat_ngram_size"

    # Test Triton output configuration
    triton_output = trt_llm.get_triton_output
    assert isinstance(triton_output, tuple)
    assert triton_output[0].name == "outputs"
    assert triton_output[1].name == "generation_logits"
    assert triton_output[2].name == "context_logits"


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_tensorrt_llm_pad_logits():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Create a sample logits tensor
    batch_size = 2
    seq_len = 3
    vocab_size = 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test padding logits
    padded_logits = trt_llm._pad_logits(logits)
    assert isinstance(padded_logits, torch.Tensor)
    assert padded_logits.shape[0] == batch_size
    assert padded_logits.shape[1] == seq_len
    assert padded_logits.shape[2] >= vocab_size  # Should be padded to a multiple of 8
