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

import copy

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import SpeechEncDecSelfSupervisedModel


@pytest.fixture()
def ssl_model():
    preprocessor = {
        'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'params': dict({'pad_to': 16, 'dither': 0}),
    }

    model_defaults = {'enc_hidden': 32, 'dec_out': 128}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
            ],
        },
    }

    spec_augment = {
        '_target_': 'nemo.collections.asr.modules.MaskedPatchAugmentation',
        'freq_masks': 3,
        'freq_width': 20,
        'patch_size': 16,
        'mask_patches': 0.5,
    }

    loss_list_contr_mlm = {
        'contr': {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoderReconstruction',
                'feat_in': model_defaults['enc_hidden'],
                'feat_hidden': 128,
                'feat_out': model_defaults['dec_out'],
                'stride_layers': 0,
                'non_stride_layers': 0,
                'stride_transpose': False,
            },
            'loss': {
                '_target_': 'nemo.collections.asr.losses.ContrastiveLoss',
                'in_dim': 64,
                'proj_dim': model_defaults['dec_out'],
                'combine_time_steps': 1,
                'quantized_targets': True,
                'codebook_size': 64,
                'sample_from_same_utterance_only': True,
                'sample_from_non_masked': False,
                'num_negatives': 3,
            },
        },
        'mlm': {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 4096,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 1},
            'targets_from_loss': "contr",
        },
    }

    modelConfig_contr_mlm = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'spec_augment': DictConfig(spec_augment),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'loss_list': DictConfig(loss_list_contr_mlm),
        }
    )
    ssl_model = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_mlm)
    return ssl_model


class TestSSLModel:
    @pytest.mark.unit
    def test_constructor(self, ssl_model):
        confdict = ssl_model.to_config_dict()
        instance2 = SpeechEncDecSelfSupervisedModel.from_config_dict(confdict)
        assert isinstance(instance2, SpeechEncDecSelfSupervisedModel)

    @pytest.mark.unit
    def test_contr_nonquant(self, ssl_model):
        modelConfig_contr_nonquant = ssl_model.to_config_dict()

        loss_list_contr_nonquant = dict(modelConfig_contr_nonquant['loss_list'])
        del loss_list_contr_nonquant['mlm']

        loss_list_contr_nonquant['contr']['loss']['quantized_targets'] = False

        modelConfig_contr_nonquant['loss_list'] = DictConfig(loss_list_contr_nonquant)

        ssl_model = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_nonquant)

        input_signal = torch.randn(size=(4, 64000))
        length = torch.randint(low=48000, high=64000, size=[4])

        with torch.no_grad():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

            loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

        assert len(loss_val_dict) == 1

    @pytest.mark.unit
    def test_contr_mlm(self, ssl_model):

        input_signal = torch.randn(size=(4, 64000))
        length = torch.randint(low=48000, high=64000, size=[4])

        with torch.no_grad():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

        loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

        assert len(loss_val_dict) == 2

    @pytest.mark.unit
    def test_contr_mlm_multi(self, ssl_model):
        modelConfig_contr_mlm_multi = ssl_model.to_config_dict()

        model_defaults = modelConfig_contr_mlm_multi['model_defaults']

        loss_list_contr_mlm_multi = dict(modelConfig_contr_mlm_multi['loss_list'])
        loss_list_contr_mlm_multi['mlm_2'] = {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 4096,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 1},
            'output_from_layer': "encoder.0",
            'targets_from_loss': "contr",
        }
        loss_list_contr_mlm_multi['mlm_3'] = {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 4096,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 1},
            'output_from_layer': "encoder.1",
            'targets_from_loss': "contr",
        }
        modelConfig_contr_mlm_multi['loss_list'] = DictConfig(loss_list_contr_mlm_multi)

        ssl_model = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_mlm_multi)

        input_signal = torch.randn(size=(4, 64000))
        length = torch.randint(low=48000, high=64000, size=[4])

        with torch.no_grad():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

            loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

        assert len(loss_val_dict) == 4
