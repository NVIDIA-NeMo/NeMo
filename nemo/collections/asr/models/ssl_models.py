# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, SpectrogramType, VoidType
from nemo.utils import logging

__all__ = ['SpeechEncDecSelfSupervisedModel']


class SpeechEncDecSelfSupervisedModel(ModelPT, ASRModuleMixin):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return []

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.encoder)

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

        self.decoder_ssl = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.decoder)
        self.loss = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.loss)

        self.spec_augmentation = SpeechEncDecSelfSupervisedModel.from_config_dict(self._cfg.spec_augment)

        # dropout for features/spectrograms (applied before masking)
        self.dropout_features = (
            torch.nn.Dropout(self._cfg.dropout_features) if "dropout_features" in self._cfg else None
        )

        # dropout for targets (applied before quantization)
        self.dropout_features_q = (
            torch.nn.Dropout(self._cfg.dropout_features_q) if "dropout_features_q" in self._cfg else None
        )

        # Feature penalty for preprocessor encodings (for Wav2Vec training)
        if "feature_penalty" in self._cfg:
            self.feat_pen, self.pen_factor = 0.0, self._cfg.feature_penalty
        else:
            self.feat_pen, self.pen_factor = None, None

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in val_data_config and val_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches
                    * ceil((len(self._validation_dl.dataset) / self.world_size) / val_data_config['batch_size'])
                )

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
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "spectrograms": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "spec_masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "outputs": NeuralType(('B', 'T', 'D'), VoidType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) Processed spectrograms of shape [B, D, T].
            2) Masks applied to spectrograms of shape [B, D, T].
            3) Decoder outputs of shape [B, T, D].
        """
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

        if self.pen_factor:
            self.feat_pen = processed_signal.float().pow(2).mean() * self.pen_factor
        spectrograms = processed_signal.detach().clone()

        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)
        if self.dropout_features_q:
            spectrograms = self.dropout_features_q(spectrograms)

        processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        masked_spectrograms = processed_signal.detach()
        spec_masks = torch.logical_and(masked_spectrograms < 1e-5, masked_spectrograms > -1e-5).float()
        for idx, proc_len in enumerate(processed_signal_length):
            spec_masks[idx, :, proc_len:] = 0.0

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        outputs = self.decoder_ssl(encoder_output=encoded)

        return spectrograms, spec_masks, outputs

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        spectrograms, spec_masks, outputs = self.forward(input_signal=signal, input_signal_length=signal_len)

        self.loss.set_num_updates(self.trainer.global_step)
        loss_value = self.loss(spectrograms=spectrograms, spec_masks=spec_masks, decoder_outputs=outputs)
        if self.feat_pen:
            loss_value += self.feat_pen

        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch
        spectrograms, spec_masks, outputs = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(spectrograms=spectrograms, spec_masks=spec_masks, decoder_outputs=outputs)
        if self.feat_pen:
            loss_value += self.feat_pen
        return {
            'val_loss': loss_value,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}
