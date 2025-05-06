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
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.mark.run_only_on('GPU')
class TestBuild(unittest.TestCase):

    @pytest.mark.run_only_on('GPU')
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config = {
            "mm_cfg": {
                "vision_encoder": {
                    "from_pretrained": "test_model",
                    "hidden_size": 768,
                },
                "mm_mlp_adapter_type": "linear",
                "hidden_size": 4096,
            }
        }
        self.mock_weights = {
            "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.weight": torch.randn(
                4096, 768
            ),
            "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.bias": torch.randn(4096),
        }

    @pytest.mark.run_only_on('GPU')
    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)

    @pytest.mark.run_only_on('GPU')
    @patch('nemo.export.multimodal.build.TensorRTLLM')
    def test_build_trtllm_engine(self, mock_trtllm):
        # Test basic functionality
        mock_exporter = MagicMock()
        mock_trtllm.return_value = mock_exporter

        from nemo.export.multimodal.build import build_trtllm_engine

        build_trtllm_engine(
            model_dir=self.temp_dir,
            visual_checkpoint_path="test_path",
            model_type="neva",
            tensor_parallelism_size=1,
            max_input_len=256,
            max_output_len=256,
            max_batch_size=1,
            max_multimodal_len=1024,
            dtype="bfloat16",
        )

        mock_exporter.export.assert_called_once()

    @pytest.mark.run_only_on('GPU')
    @patch('nemo.export.multimodal.build.MLLaMAForCausalLM')
    @patch('nemo.export.multimodal.build.build_trtllm')
    def test_build_mllama_trtllm_engine(self, mock_build_trtllm, mock_mllama):
        # Test basic functionality
        mock_model = MagicMock()
        mock_mllama.from_hugging_face.return_value = mock_model
        mock_build_trtllm.return_value = MagicMock()

        from nemo.export.multimodal.build import build_mllama_trtllm_engine

        build_mllama_trtllm_engine(
            model_dir=self.temp_dir,
            hf_model_path="test_path",
            tensor_parallelism_size=1,
            max_input_len=256,
            max_output_len=256,
            max_batch_size=1,
            max_multimodal_len=1024,
            dtype="bfloat16",
        )

        mock_mllama.from_hugging_face.assert_called_once()
        mock_build_trtllm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
