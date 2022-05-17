# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import ASRModel, EncDecCTCModel
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin, get_registered_adapter
from nemo.utils import config_utils


@pytest.fixture()
def model():
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoderAdapter',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 50,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 50,
        'num_classes': 28,
        'vocabulary': [
            ' ',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
            "'",
        ],
    }
    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


def get_adapter_cfg(in_features=50, dim=100, norm_pos='pre'):
    cfg = adapter_modules.LinearAdapterConfig(in_features=in_features, dim=dim, norm_position=norm_pos)
    cfg = OmegaConf.structured(cfg)
    return cfg


class TestASRAdapterMixin:
    @pytest.mark.unit
    def test_asr_model_constructor(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_asr_model_constructor_encoder_module(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='encoder:adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_asr_multiple_adapter(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        original_num_params = new_num_params
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_asr_forward_linear_pre(self, model):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0'])
    def test_asr_forward_linear_post(self, model, name):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name, cfg=get_adapter_cfg(norm_pos='post'))
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name1', ['adapter_0', 'encoder:adapter_0'])
    @pytest.mark.parametrize('name2', ['adapter_1', 'encoder:adapter_1'])
    def test_asr_multi_adapter_forward(self, model, name1, name2):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name1, cfg=get_adapter_cfg())
        model.add_adapter(name=name2, cfg=get_adapter_cfg())
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        resolved_name1 = model.resolve_adapter_module_name_(name1)[-1]
        resolved_name2 = model.resolve_adapter_module_name_(name2)[-1]
        assert model.get_enabled_adapters() == [resolved_name1, resolved_name2]
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name1', ['adapter_0', 'encoder:adapter_0'])
    @pytest.mark.parametrize('name2', ['adapter_1', 'encoder:adapter_1'])
    def test_asr_multi_adapter_partial_forward(self, model, name1, name2):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name1, cfg=get_adapter_cfg())
        model.add_adapter(name=name2, cfg=get_adapter_cfg())

        model.set_enabled_adapters(name=name1, enabled=False)
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        resolved_name2 = model.resolve_adapter_module_name_(name2)[-1]
        assert model.get_enabled_adapters() == [resolved_name2]
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0'])
    def test_asr_forward_unfrozen_adapters(self, model, name):
        model.eval()
        original_num_params = model.num_weights

        dim = 10
        model.add_adapter(name=name, cfg=get_adapter_cfg(dim=dim))
        model.freeze()
        model.unfreeze_enabled_adapters()

        assert original_num_params == 5443

        original_params = 0
        adapter_params = 0
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                assert param.requires_grad is False
                original_params += param.numel()
            else:
                assert param.requires_grad is True
                adapter_params += param.numel()

        for mname, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                assert module.track_running_stats is False

        assert original_params > adapter_params

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor_pretrained(self):
        # Check to/from config_dict:
        cfg = ASRModel.from_pretrained('stt_en_citrinet_256', map_location='cpu', return_config=True)
        adapter_metadata = get_registered_adapter(cfg.encoder._target_)
        if adapter_metadata is not None:
            cfg.encoder._target_ = adapter_metadata.adapter_class_path
        model = ASRModel.from_pretrained('stt_en_citrinet_256', override_config_path=cfg)

        assert isinstance(model, AdapterModuleMixin)
        assert hasattr(model, 'encoder')
        assert isinstance(model.encoder, AdapterModuleMixin)

        model.add_adapter('adapter_0', cfg=get_adapter_cfg(in_features=cfg.encoder.jasper[0].filters, dim=5))
        assert model.is_adapter_available()

        model.freeze()
        model.unfreeze_enabled_adapters()
        assert model.num_weights < 1e5
