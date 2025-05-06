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


from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def exporter():
    from nemo.export.vllm_hf_exporter import vLLMHFExporter

    return vLLMHFExporter()


@pytest.fixture
def mock_llm():
    with patch('nemo.export.vllm_hf_exporter.LLM') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_init(exporter):
    """Test initialization of vLLMHFExporter"""
    assert exporter.model is None
    assert exporter.lora_models is None


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_export(exporter, mock_llm):
    """Test export method"""
    model_path = "/path/to/model"
    exporter.export(model=model_path)

    assert exporter.model is not None
    mock_llm.assert_called_once_with(model=model_path, enable_lora=False)


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_export_with_lora(exporter, mock_llm):
    """Test export method with LoRA enabled"""
    model_path = "/path/to/model"
    exporter.export(model=model_path, enable_lora=True)

    assert exporter.model is not None
    mock_llm.assert_called_once_with(model=model_path, enable_lora=True)


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_add_lora_models(exporter):
    """Test adding LoRA models"""
    lora_name = "test_lora"
    lora_model = "path/to/lora"

    exporter.add_lora_models(lora_name, lora_model)

    assert exporter.lora_models is not None
    assert lora_name in exporter.lora_models
    assert exporter.lora_models[lora_name] == lora_model


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_get_triton_input(exporter):
    """Test triton input configuration"""
    inputs = exporter.get_triton_input

    # Check that we have all expected inputs
    input_names = [tensor.name for tensor in inputs]
    assert "prompts" in input_names
    assert "max_output_len" in input_names
    assert "top_k" in input_names
    assert "top_p" in input_names
    assert "temperature" in input_names

    # Check data types
    for tensor in inputs:
        if tensor.name == "prompts":
            assert tensor.dtype == bytes
        elif tensor.name == "max_output_len":
            assert tensor.dtype == np.int_
        elif tensor.name in ["top_k"]:
            assert tensor.dtype == np.int_
        elif tensor.name in ["top_p", "temperature"]:
            assert tensor.dtype == np.single


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_get_triton_output(exporter):
    """Test triton output configuration"""
    outputs = exporter.get_triton_output

    assert len(outputs) == 1
    assert outputs[0].name == "outputs"
    assert outputs[0].dtype == bytes


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_forward_without_model(exporter):
    """Test forward method without initialized model"""
    with pytest.raises(AssertionError, match="Model is not initialized"):
        exporter.forward(["test prompt"])


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_forward_with_lora_not_added(exporter, mock_llm):
    """Test forward method with non-existent LoRA model"""
    exporter.export(model="/path/to/model")

    with pytest.raises(Exception, match="No lora models are available"):
        exporter.forward(["test prompt"], lora_model_name="non_existent_lora")


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_forward_with_invalid_lora(exporter, mock_llm):
    """Test forward method with invalid LoRA model name"""
    exporter.export(model="/path/to/model")
    exporter.add_lora_models("valid_lora", "path/to/lora")

    with pytest.raises(AssertionError, match="Lora model was not added before"):
        exporter.forward(["test prompt"], lora_model_name="invalid_lora")


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_triton_infer_fn(exporter, mock_llm):
    """Test triton inference function"""
    exporter.export(model="/path/to/model")
    mock_llm.generate.return_value = [MagicMock(outputs=[MagicMock(text="test output")])]

    inputs = {
        "prompts": np.array([b"test prompt"]),
        "max_output_len": np.array([64]),
        "top_k": np.array([1]),
        "top_p": np.array([0.1]),
        "temperature": np.array([1.0]),
    }

    result = exporter.triton_infer_fn(**inputs)

    assert "outputs" in result
    assert isinstance(result["outputs"], np.ndarray)
    assert result["outputs"].dtype == np.bytes_


@pytest.mark.skip(reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on('GPU')
def test_triton_infer_fn_error_handling(exporter):
    """Test triton inference function error handling"""
    inputs = {"prompts": np.array([b"test prompt"])}

    result = exporter.triton_infer_fn(**inputs)

    assert "outputs" in result
    assert isinstance(result["outputs"], np.ndarray)
    assert result["outputs"].dtype == np.bytes_
    assert b"An error occurred" in result["outputs"][0]
