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


from unittest.mock import MagicMock

import pytest
import torch

from nemo.export.onnx_llm_exporter import OnnxLLMExporter


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, inputs):
        return self.linear(inputs['input_ids'])


class TestOnnxLLMExporter:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "onnx_model")

    @pytest.fixture
    def dummy_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()
        return tokenizer

    @pytest.fixture
    def dummy_model(self):
        return DummyModel()

    def test_init_with_model_and_tokenizer(self, temp_dir, dummy_model, dummy_tokenizer):
        exporter = OnnxLLMExporter(
            onnx_model_dir=temp_dir, model=dummy_model, tokenizer=dummy_tokenizer, load_runtime=False
        )
        assert exporter.model == dummy_model
        assert exporter.tokenizer == dummy_tokenizer
        assert exporter.onnx_model_dir == temp_dir

    def test_init_with_model_and_model_path_raises_error(self, temp_dir, dummy_model):
        with pytest.raises(ValueError, match="A model was also passed but it will be overridden"):
            OnnxLLMExporter(
                onnx_model_dir=temp_dir, model=dummy_model, model_name_or_path="some/path", load_runtime=False
            )
