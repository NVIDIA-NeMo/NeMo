from math import ceil
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.modules.audio_preprocessing import RandomBlockMaskingAugmentation
from nemo.collections.asr.parts.submodules.ssl_quantizers import RandomProjectionVectorQuantizer
from nemo.core.classes.common import PretrainedModelInfo, typecheck


class SelfSupervisedRandomQuantizationModel(SpeechEncDecSelfSupervisedModel):
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

    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
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

        if self.training:
            masked_signal, masks = self.mask_processor(
                input_feats=processed_signal, input_lengths=processed_signal_length, mask_value=self.mask_embedding
            )
        else:
            masked_signal = processed_signal
            masks = torch.zeros_like(processed_signal)

        encoded, encoded_len = self.encoder(audio_signal=masked_signal, length=processed_signal_length)
        log_probs = self.decoder(encoder_output=encoded)

        return log_probs, encoded_len, masks, tokens

    def training_step(self, batch, batch_idx):
        input_signal, input_signal_length, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, masks, tokens = self.forward(
                processed_signal=input_signal, processed_signal_length=input_signal_length
            )
        else:
            log_probs, encoded_len, masks, tokens = self.forward(
                input_signal=input_signal, input_signal_length=input_signal_length
            )

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value.item(),
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_signal, input_signal_length, _, _ = batch
        log_probs, encoded_len, masks, tokens = self.forward(input_signal, input_signal_length)

        loss_value = self.loss(masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len)

        return {'val_loss': loss_value.item()}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        return test_logs
