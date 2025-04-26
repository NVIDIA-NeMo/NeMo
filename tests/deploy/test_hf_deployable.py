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

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo.deploy.nlp.hf_deployable import HuggingFaceLLMDeploy


@pytest.fixture
def mock_model():
    model = MagicMock(spec=AutoModelForCausalLM)
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.batch_decode.return_value = ["Generated text"]
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    return tokenizer


@pytest.fixture
def mock_peft_model():
    with patch("nemo.deploy.nlp.hf_deployable.PeftModel") as mock:
        mock.from_pretrained.return_value = MagicMock()
        yield mock


class TestHuggingFaceLLMDeploy:

    def test_initialization_invalid_task(self):
        with pytest.raises(AssertionError):
            HuggingFaceLLMDeploy(hf_model_id_path="test/model", task="invalid-task")

    def test_initialization_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceLLMDeploy(task="text-generation")

    def test_triton_input_output_config(self):
        deployer = HuggingFaceLLMDeploy(model=MagicMock(), tokenizer=MagicMock(), task="text-generation")

        inputs = deployer.get_triton_input
        outputs = deployer.get_triton_output

        assert len(inputs) == 10  # Verify number of input tensors
        assert len(outputs) == 3  # Verify number of output tensors

        # Verify required input tensor names
        assert any(tensor.name == "prompts" for tensor in inputs)
        assert any(tensor.name == "max_length" for tensor in inputs)

        # Verify output tensor names
        assert any(tensor.name == "sentences" for tensor in outputs)
        assert any(tensor.name == "logits" for tensor in outputs)
        assert any(tensor.name == "scores" for tensor in outputs)
