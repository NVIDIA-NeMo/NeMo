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


from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def model_dir(tmp_path):
    return str(tmp_path / "model_dir")


@pytest.fixture
def mock_runner():
    runner = Mock()
    runner.model_type = "neva"
    runner.load_test_media = Mock(return_value=np.zeros((1, 224, 224, 3)))
    runner.run = Mock(return_value="Test response")
    return runner


class TestTensorRTMMExporter:

    @pytest.mark.run_only_on('GPU')
    def test_init(self, model_dir):
        # Test basic initialization
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        assert exporter.model_dir == model_dir
        assert exporter.runner is None
        assert exporter.modality == "vision"

    @pytest.mark.run_only_on('GPU')
    def test_init_invalid_modality(self, model_dir):
        # Test initialization with invalid modality
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        with pytest.raises(AssertionError):
            TensorRTMMExporter(model_dir, modality="invalid")

    @pytest.mark.run_only_on('GPU')
    @patch("nemo.export.tensorrt_mm_exporter.build_mllama_engine")
    def test_export_mllama(self, mock_build, model_dir):
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path", model_type="mllama", tensor_parallel_size=1, load_model=False
        )
        mock_build.assert_called_once()

    @pytest.mark.run_only_on('GPU')
    @patch("nemo.export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo.export.tensorrt_mm_exporter.build_visual_engine")
    def test_export_neva(self, mock_visual, mock_trtllm, model_dir):
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path", model_type="neva", tensor_parallel_size=1, load_model=False
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()

    @pytest.mark.run_only_on('GPU')
    def test_forward_without_loading(self, model_dir):
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        with pytest.raises(Exception) as exc_info:
            exporter.forward("test prompt", "test_image.jpg")
        assert "should be exported and" in str(exc_info.value)

    @pytest.mark.run_only_on('GPU')
    def test_forward(self, model_dir, mock_runner):
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.runner = mock_runner

        result = exporter.forward(
            input_text="What's in this image?", input_media="test_image.jpg", batch_size=1, max_output_len=30
        )

        assert result == "Test response"
        mock_runner.load_test_media.assert_called_once()
        mock_runner.run.assert_called_once()

    @pytest.mark.run_only_on('GPU')
    def test_get_triton_input(self, model_dir):
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        inputs = exporter.get_triton_input

        # Verify we have the expected number of inputs
        assert len(inputs) == 10  # 1 text input + 1 media input + 8 optional parameters

        # Verify the first input is for text
        assert inputs[0].name == "input_text"
        assert inputs[0].dtype == bytes

    @pytest.mark.run_only_on('GPU')
    def test_get_triton_output(self, model_dir):
        from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        outputs = exporter.get_triton_output

        assert len(outputs) == 1
        assert outputs[0].name == "outputs"
        assert outputs[0].dtype == bytes
