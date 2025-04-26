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
from omegaconf import DictConfig

from nemo.export.quantize.quantizer import QUANT_CFG_CHOICES, Quantizer


@pytest.fixture
def basic_quantization_config():
    return DictConfig(
        {'algorithm': 'int8', 'decoder_type': 'llama', 'awq_block_size': 128, 'sq_alpha': 0.5, 'enable_kv_cache': True}
    )


@pytest.fixture
def basic_export_config():
    return DictConfig(
        {
            'dtype': '16',
            'decoder_type': 'llama',
            'inference_tensor_parallel': 1,
            'inference_pipeline_parallel': 1,
            'save_path': '/tmp/model.qnemo',
        }
    )


class TestQuantizer:
    def test_init_valid_configs(self, basic_quantization_config, basic_export_config):
        quantizer = Quantizer(basic_quantization_config, basic_export_config)
        assert quantizer.quantization_config == basic_quantization_config
        assert quantizer.export_config == basic_export_config
        assert quantizer.quant_cfg == QUANT_CFG_CHOICES['int8']

    def test_init_invalid_algorithm(self, basic_quantization_config, basic_export_config):
        basic_quantization_config.algorithm = 'invalid_algo'
        with pytest.raises(AssertionError):
            Quantizer(basic_quantization_config, basic_export_config)

    def test_init_invalid_dtype(self, basic_quantization_config, basic_export_config):
        basic_export_config.dtype = '32'
        with pytest.raises(AssertionError):
            Quantizer(basic_quantization_config, basic_export_config)

    def test_null_algorithm(self, basic_quantization_config, basic_export_config):
        basic_quantization_config.algorithm = None
        quantizer = Quantizer(basic_quantization_config, basic_export_config)
        assert quantizer.quant_cfg is None

    @patch('nemo.export.quantize.quantizer.dist')
    def test_quantize_method(self, mock_dist, basic_quantization_config, basic_export_config):
        mock_dist.get_rank.return_value = 0

        # Create mock model and forward loop
        mock_model = MagicMock()
        mock_forward_loop = MagicMock()

        quantizer = Quantizer(basic_quantization_config, basic_export_config)

        with patch('modelopt.torch.quantization.quantize') as mock_quantize:
            with patch('modelopt.torch.quantization.print_quant_summary'):
                quantizer.quantize(mock_model, mock_forward_loop)

                # Verify quantize was called with correct arguments
                mock_quantize.assert_called_once_with(mock_model, QUANT_CFG_CHOICES['int8'], mock_forward_loop)

    @patch('nemo.export.quantize.quantizer.dist')
    def test_modify_model_config(self, mock_dist):
        mock_config = DictConfig({'sequence_parallel': True})
        modified_config = Quantizer.modify_model_config(mock_config)

        assert modified_config.sequence_parallel is False
        assert modified_config.name == 'modelopt'
        assert modified_config.apply_rope_fusion is False

    @patch('nemo.export.quantize.quantizer.dist')
    @patch('nemo.export.quantize.quantizer.export_tensorrt_llm_checkpoint')
    def test_export_method(self, mock_export, mock_dist, basic_quantization_config, basic_export_config):
        mock_dist.get_rank.return_value = 0
        mock_model = MagicMock()
        mock_model.cfg.megatron_amp_O2 = False
        mock_model.trainer.num_nodes = 1

        quantizer = Quantizer(basic_quantization_config, basic_export_config)

        with patch('nemo.export.quantize.quantizer.save_artifacts'):
            quantizer.export(mock_model)

            # Verify export was called with correct arguments
            mock_export.assert_called_once()
            call_args = mock_export.call_args[1]
            assert call_args['decoder_type'] == 'llama'
            assert call_args['inference_tensor_parallel'] == 1
            assert call_args['inference_pipeline_parallel'] == 1
