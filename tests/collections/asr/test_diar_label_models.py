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
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecDiarLabelModel


@pytest.fixture()
def msdd_model():

    preprocessor = {
        'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'params': {"features": 80, "window_size": 0.025, "window_stride": 0.01, "sample_rate": 16000,},
    }

    speaker_model_encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 80,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 512,
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': False,
                }
            ],
        },
    }

    speaker_model_decoder = {
        'cls': 'nemo.collections.asr.modules.SpeakerDecoder',
        'params': {'feat_in': 512, 'num_classes': 2, 'pool_mode': 'xvector', 'emb_sizes': [1024]},
    }

    speaker_model_cfg = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(speaker_model_encoder),
            'decoder': DictConfig(speaker_model_decoder),
        }
    )

    msdd_module = {
        'cls': 'nemo.collections.asr.modules.MSDD_module',
        'params': {
            "num_spks": 2,
            "hidden_size": 256,
            "num_lstm_layers": 3,
            "dropout_rate": 0.5,
            "cnn_output_ch": 32,
            "conv_repeat": 2,
            "emb_dim": 192,
            "scale_n": 5,
            "weighting_scheme": 'conv_scale_weight',
            "context_vector_type": 'cos_sim',
        },
    }

    loss = {'cls': 'nemo.collections.asr.losses.bce_loss.BCELoss', 'params': {"weight": None}}

    diarizer = {
        'out_dir': None,
        'oracle_vad': True,
        "speaker_embeddings": {
            "model_path": None,
            "parameters": {
                "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                "multiscale_weights": [1, 1, 1, 1, 1],
                "save_embeddings": True,
            },
        },
    }

    modelConfig = DictConfig(
        {
            'msdd_module': DictConfig(msdd_module),
            'preprocessor': DictConfig(preprocessor),
            'diarizer': DictConfig(diarizer),
            'loss': DictConfig(loss),
            'max_num_of_spks': 2,
            'num_workers': 5,
            'emb_batch_size': 0,
            'soft_label_thres': 0.5,
            'scale_n': 5,
            'speaker_model_cfg': speaker_model_cfg,
        }
    )
    model = EncDecDiarLabelModel(cfg=modelConfig)
    return model


class TestEncDecDiarLabelModel:
    @pytest.mark.unit
    def test_constructor(self, msdd_model):
        diar_model = msdd_model.train()
        assert diar_model.cfg.scale_n == len(
            diar_model.cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec
        )
        assert diar_model.cfg.scale_n == len(diar_model.cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec)
        assert diar_model.cfg.scale_n == len(diar_model.cfg.diarizer.speaker_embeddings.parameters.multiscale_weights)
        assert diar_model.cfg.msdd_module.num_spks == diar_model.cfg.max_num_of_spks
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = diar_model.to_config_dict()
        instance2 = EncDecDiarLabelModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecDiarLabelModel)

    @pytest.mark.unit
    def test_forward_infer(self, msdd_model):
        diar_model = msdd_model.eval()

        # batch_size 4, scale_n 5, length 25, emb_dim 192
        input_signal = torch.randn(size=(4, 25, 5, 192))
        input_signal_length = 25 * torch.ones(4, dtype=torch.int)
        emb_vectors = torch.randn(size=(4, 5, 192, 2))
        targets = torch.randint(2, size=(4, 25, 2), dtype=torch.int)

        with torch.no_grad():
            # batch size 1
            preds_list, scale_weights_list = [], []
            for i in range(input_signal.size(0)):
                preds, scale_weights = diar_model.forward_infer(
                    input_signal[i : i + 1], input_signal_length[i : i + 1], emb_vectors[i : i + 1], targets[i : i + 1]
                )
                preds_list.append(preds)
                scale_weights_list.append(scale_weights)
            preds_instance = torch.cat(preds_list, 0)
            scale_weights_instance = torch.cat(scale_weights_list, 0)

            # batch size 4
            preds_batch, scale_weights_batch = diar_model.forward_infer(
                input_signal, input_signal_length, emb_vectors, targets
            )

        assert preds_instance.shape == preds_batch.shape
        assert scale_weights_instance.shape == scale_weights_batch.shape

        diff = torch.mean(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(preds_instance - preds_batch))
        assert diff <= 1e-6
        diff = torch.mean(torch.abs(scale_weights_instance - scale_weights_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(scale_weights_instance - scale_weights_batch))
        assert diff <= 1e-6
