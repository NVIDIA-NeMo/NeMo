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

import copy
import itertools
from math import ceil
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning import Trainer
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset
from nemo.collections.asr.data.audio_to_label_dataset import get_tarred_speech_label_dataset
from nemo.collections.asr.data.audio_to_text_dataset import convert_to_config_list
from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.losses import CrossEntropyLoss as CELoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.collections.common.parts.preprocessing.collections import ASRSpeechLabel
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import *
from nemo.utils import logging

__all__ = ['EncDecSpeakerLabelModel']


class EncDecSpeakerLabelModel(ModelPT, ExportableEncDecModel):
    """
    Encoder decoder class for speaker label models.
    Model class creates training, validation methods for setting up data
    performing model forward pass.
    Expects config dict for
        * preprocessor
        * Jasper/Quartznet Encoder
        * Speaker Decoder
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []

        model = PretrainedModelInfo(
            pretrained_model_name="speakerverification_speakernet",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/speakerverification_speakernet/versions/1.0.0rc1/files/speakerverification_speakernet.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:speakerverification_speakernet",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="ecapa_tdnn",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/ecapa_tdnn/versions/v1/files/ecapa_tdnn.nemo",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:ecapa_tdnn",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="titanet_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/titanet_large/versions/v0/files/titanet-l.nemo",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_large",
        )
        result.append(model)

        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = EncDecSpeakerLabelModel.from_config_dict(cfg.preprocessor)
        self.encoder = EncDecSpeakerLabelModel.from_config_dict(cfg.encoder)
        self.decoder = EncDecSpeakerLabelModel.from_config_dict(cfg.decoder)
        if 'angular' in cfg.decoder and cfg.decoder['angular']:
            logging.info("loss is Angular Softmax")
            scale = cfg.loss.scale
            margin = cfg.loss.margin
            self.loss = AngularSoftmaxLoss(scale=scale, margin=margin)
        else:
            logging.info("loss is Softmax-CrossEntropy")
            self.loss = CELoss()
        self.task = None
        self._accuracy = TopKClassificationAccuracy(top_k=[1])
        self.labels = None
        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecSpeakerLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

    @staticmethod
    def extract_labels(data_layer_config):
        labels = set()
        manifest_filepath = data_layer_config.get('manifest_filepath', None)
        if manifest_filepath is None:
            logging.warning("No manifest_filepath was provided, no labels got extracted!")
            return None
        manifest_filepaths = convert_to_config_list(data_layer_config['manifest_filepath'])

        for manifest_filepath in itertools.chain.from_iterable(manifest_filepaths):
            collection = ASRSpeechLabel(
                manifests_files=manifest_filepath,
                min_duration=data_layer_config.get("min_duration", None),
                max_duration=data_layer_config.get("max_duration", None),
                index_by_file_id=True,
            )
            labels.update(collection.uniq_labels)
        labels = list(sorted(labels))
        logging.warning(f"Total number of {len(labels)} found in all the manifest files.")
        return labels

    def __setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config.get('shuffle', False)
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
            dataset = get_tarred_speech_label_dataset(
                featurizer=featurizer,
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = AudioToSpeechLabelDataset(
                manifest_filepath=config['manifest_filepath'],
                labels=config['labels'],
                featurizer=featurizer,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                trim=config.get('trim_silence', False),
                normalize_audio=config.get('normalize_audio', False),
            )

        if hasattr(dataset, 'fixed_seq_collate_fn'):
            collate_fn = dataset.fixed_seq_collate_fn
        else:
            collate_fn = dataset.datasets[0].fixed_seq_collate_fn

        batch_size = config['batch_size']
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self.labels = self.extract_labels(train_data_layer_config)
        train_data_layer_config['labels'] = self.labels
        if 'shuffle' not in train_data_layer_config:
            train_data_layer_config['shuffle'] = True
        self._train_dl = self.__setup_dataloader_from_config(config=train_data_layer_config)
        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_layer_config and train_data_layer_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_layer_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        val_data_layer_config['labels'] = self.labels
        self._validation_dl = self.__setup_dataloader_from_config(config=val_data_layer_config)

    def setup_test_data(self, test_data_layer_params: Optional[Union[DictConfig, Dict]]):
        if hasattr(self, 'dataset'):
            test_data_layer_params['labels'] = self.labels

        self.embedding_dir = test_data_layer_params.get('embedding_dir', './')
        self._test_dl = self.__setup_dataloader_from_config(config=test_data_layer_params)
        self.test_manifest = test_data_layer_params.get('manifest_filepath', None)

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), audio_eltype),
            "input_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'D'), LogitsType()),
            "embs": NeuralType(('B', 'D'), AcousticEncodedRepresentation()),
        }

    def forward_for_export(self, processed_signal, processed_signal_len):
        encoded, length = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits, embs = self.decoder(encoder_output=encoded, length=length)
        return logits, embs

    @typecheck()
    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)

        encoded, length = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits, embs = self.decoder(encoder_output=encoded, length=length)
        return logits, embs

    # PTL-specific methods
    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss = self.loss(logits=logits, labels=labels)

        self.log('loss', loss)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        self._accuracy(logits=logits, labels=labels)
        top_k = self._accuracy.compute()
        self._accuracy.reset()
        for i, top_i in enumerate(top_k):
            self.log(f'training_batch_accuracy_top@{i}', top_i)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc_top_k = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        return {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc_top_k': acc_top_k,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['val_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        logging.info("val_loss: {:.3f}".format(val_loss_mean))
        self.log('val_loss', val_loss_mean)
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('val_epoch_accuracy_top@{}'.format(top_k), score)

        return {
            'val_loss': val_loss_mean,
            'val_acc_top_k': topk_scores,
        }

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc_top_k = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        return {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc_top_k': acc_top_k,
        }

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['test_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['test_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        logging.info("test_loss: {:.3f}".format(test_loss_mean))
        self.log('test_loss', test_loss_mean)
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('test_epoch_accuracy_top@{}'.format(top_k), score)

        return {
            'test_loss': test_loss_mean,
            'test_acc_top_k': topk_scores,
        }

    def setup_finetune_model(self, model_config: DictConfig):
        """
        setup_finetune_model method sets up training data, validation data and test data with new
        provided config, this checks for the previous labels set up during training from scratch, if None,
        it sets up labels for provided finetune data from manifest files

        Args:
            model_config: cfg which has train_ds, optional validation_ds, optional test_ds, 
            mandatory encoder and decoder model params. Make sure you set num_classes correctly for finetune data.

        Returns: 
            None
        """
        logging.info("Setting up data loaders with manifests provided from model_config")

        if 'train_ds' in model_config and model_config.train_ds is not None:
            self.setup_training_data(model_config.train_ds)
        else:
            raise KeyError("train_ds is not found in model_config but you need it for fine tuning")

        if self.labels is None or len(self.labels) == 0:
            raise ValueError(f'New labels must be non-empty list of labels. But I got: {self.labels}')

        if 'validation_ds' in model_config and model_config.validation_ds is not None:
            self.setup_multiple_validation_data(model_config.validation_ds)

        if 'test_ds' in model_config and model_config.test_ds is not None:
            self.setup_multiple_test_data(model_config.test_ds)

        if self.labels is not None:  # checking for new finetune dataset labels
            logging.warning(
                "Trained dataset labels are same as finetune dataset labels -- continuing change of decoder parameters"
            )
        else:
            logging.warning(
                "Either you provided a dummy manifest file during training from scratch or you restored from a pretrained nemo file"
            )

        decoder_config = model_config.decoder
        new_decoder_config = copy.deepcopy(decoder_config)
        if new_decoder_config['num_classes'] != len(self.labels):
            raise ValueError(
                "number of classes provided {} is not same as number of different labels in finetuning data: {}".format(
                    new_decoder_config['num_classes'], len(self.labels)
                )
            )

        del self.decoder
        self.decoder = EncDecSpeakerLabelModel.from_config_dict(new_decoder_config)

        with open_dict(self._cfg.decoder):
            self._cfg.decoder = new_decoder_config

        logging.info(f"Changed decoder output to # {self.decoder._num_classes} classes.")

    @torch.no_grad()
    def get_embedding(self, path2audio_file):
        """
        Returns the speaker embeddings for a provided audio file.

        Args:
            path2audio_file: path to audio wav file

        Returns:
            embs: speaker embeddings 
        """
        audio, sr = librosa.load(path2audio_file, sr=None)
        target_sr = self._cfg.train_ds.get('sample_rate', 16000)
        if sr != target_sr:
            audio = librosa.core.resample(audio, sr, target_sr)
        audio_length = audio.shape[0]
        device = self.device
        audio = np.array(audio)
        audio_signal, audio_signal_len = (
            torch.tensor([audio], device=device),
            torch.tensor([audio_length], device=device),
        )
        mode = self.training
        self.freeze()

        _, embs = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        self.train(mode=mode)
        if mode is True:
            self.unfreeze()
        del audio_signal, audio_signal_len
        return embs

    @torch.no_grad()
    def verify_speakers(self, path2audio_file1, path2audio_file2, threshold=0.7):
        """
        Verify if two audio files are from the same speaker or not.

        Args:
            path2audio_file1: path to audio wav file of speaker 1  
            path2audio_file2: path to audio wav file of speaker 2 
            threshold: cosine similarity score used as a threshold to distinguish two embeddings (default = 0.7)

        Returns:  
            True if both audio files are from same speaker, False otherwise
        """
        embs1 = self.get_embedding(path2audio_file1).squeeze()
        embs2 = self.get_embedding(path2audio_file2).squeeze()
        # Length Normalize
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2
        # Decision
        if similarity_score >= threshold:
            logging.info(" two audio files are from same speaker")
            return True
        else:
            logging.info(" two audio files are from different speakers")
            return False

    @staticmethod
    @torch.no_grad()
    def get_batch_embeddings(speaker_model, manifest_filepath, batch_size=32, sample_rate=16000, device='cuda'):

        speaker_model.eval()
        if device == 'cuda':
            speaker_model.to(device)

        featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        dataset = AudioToSpeechLabelDataset(manifest_filepath=manifest_filepath, labels=None, featurizer=featurizer)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, collate_fn=dataset.fixed_seq_collate_fn,
        )

        all_logits = []
        all_labels = []
        all_embs = []

        for test_batch in tqdm(dataloader):
            if device == 'cuda':
                test_batch = [x.to(device) for x in test_batch]
            audio_signal, audio_signal_len, labels, _ = test_batch
            logits, embs = speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embs.extend(embs.cpu().numpy())

        all_logits, true_labels, all_embs = np.asarray(all_logits), np.asarray(all_labels), np.asarray(all_embs)

        return all_embs, all_logits, true_labels, dataset.id2label
