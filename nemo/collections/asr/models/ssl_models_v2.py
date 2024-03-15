# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from math import ceil
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.modules.ssl_modules.masking import ConvFeatureMaksingWrapper
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    BoolType,
    LabelsType,
    LengthsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging


class EncDecSpeechSSLModel(SpeechEncDecSelfSupervisedModel):
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        self.quantizer = self.from_config_dict(cfg.quantizer)
        self.mask_processor = self.from_config_dict(cfg.masking)
        self.encoder = self.from_config_dict(cfg.encoder)
        self.decoder = self.from_config_dict(cfg.decoder)
        self.loss = self.from_config_dict(cfg.loss)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "targets": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "apply_mask": NeuralType(optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self.cfg.num_books == 1 and self.cfg.squeeze_single:
            logprobs = NeuralType(('B', 'T', 'C'), LogprobsType())
            tokens = NeuralType(('B', 'T'), LabelsType())
        else:
            logprobs = NeuralType(('B', 'T', 'C', 'H'), LogprobsType())
            tokens = NeuralType(('B', 'T', 'H'), LabelsType())
        return {
            "logprobs": logprobs,
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "tokens": tokens,
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        apply_mask=False,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        _, tokens = self.quantizer(input_signal=processed_signal)

        if apply_mask:
            masked_signal, masks = self.mask_processor(
                input_feats=processed_signal, input_lengths=processed_signal_length
            )
        else:
            masked_signal = processed_signal
            masks = torch.zeros_like(processed_signal)

        encoded, encoded_len = self.encoder(audio_signal=masked_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded)

        return log_probs, encoded_len, masks.detach(), tokens.detach()

    def training_step(self, batch, batch_idx):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length, apply_mask=True
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length, apply_mask=True
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value,
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def inference_pass(self, batch, batch_idx, dataloader_idx=0, mode='val'):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length, apply_mask=True
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length, apply_mask=True
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        return {f'{mode}_loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx, eval_mode="test")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}


class SelfSupervisedConvMLMModel(SpeechEncDecSelfSupervisedModel):
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        if "mask_embedding" in cfg:
            self.mask_embedding = nn.Parameter(torch.FloatTensor(cfg.encoder.feat_in))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.1)
            if cfg.mask_embedding.get("freeze", False):
                self.mask_embedding.requires_grad = False
        else:
            self.mask_embedding = None

        self.quantizer = self.from_config_dict(cfg.quantizer)
        self.mask_processor = self.from_config_dict(cfg.masking)
        self.encoder = self.from_config_dict(cfg.encoder)
        self.decoder = self.from_config_dict(cfg.decoder)
        self.loss = self.from_config_dict(cfg.loss)

        # hacked to mask features after convolutional sub-sampling
        self.pre_encoder = ConvFeatureMaksingWrapper(self.encoder.pre_encode, self.mask_processor)
        self.encoder.pre_encode = self.pre_encoder

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "targets": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "apply_mask": NeuralType(optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self.cfg.num_books == 1 and self.cfg.squeeze_single:
            logprobs = NeuralType(('B', 'T', 'C'), LogprobsType())
            tokens = NeuralType(('B', 'T'), LabelsType())
        else:
            logprobs = NeuralType(('B', 'T', 'C', 'H'), LogprobsType())
            tokens = NeuralType(('B', 'T', 'H'), LabelsType())
        return {
            "logprobs": logprobs,
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "tokens": tokens,
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        apply_mask=False,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        _, tokens = self.quantizer(input_signal=processed_signal)

        self.encoder.pre_encode.set_masking(apply_mask=apply_mask)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        masks = self.encoder.pre_encode.get_current_mask()

        log_probs = self.decoder(encoder_output=encoded)

        return log_probs, encoded_len, masks, tokens

    def training_step(self, batch, batch_idx):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length, apply_mask=True
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length, apply_mask=True
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value,
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def inference_pass(self, batch, batch_idx, dataloader_idx=0, mode='val'):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length, apply_mask=True
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length, apply_mask=True
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        return {f'{mode}_loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_pass(batch, batch_idx, dataloader_idx, eval_mode="test")
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': test_loss_mean}
        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}
