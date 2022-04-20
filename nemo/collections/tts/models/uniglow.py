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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pystoi import stoi
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.tts.helpers.helpers import OperationMode, split_view, waveglow_log_to_tb_func
from nemo.collections.tts.losses.uniglowloss import UniGlowLoss
from nemo.collections.tts.models.base import GlowVocoder
from nemo.collections.tts.modules.uniglow import UniGlowModule
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    LengthsType,
    LogDeterminantType,
    MelSpectrogramType,
    NormalDistributionSamplesType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@dataclass
class WaveglowConfig:
    waveglow: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    sigma: float = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


@deprecated(version="1.9", explanation="UniGlowModel will be removed. Use WaveGlowModel or HifiGanModel instead.")
class UniGlowModel(GlowVocoder):
    """Waveglow model used to convert betweeen spectrograms and audio"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(WaveglowConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.sigma = self._cfg.sigma
        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.model = UniGlowModule(
            self._cfg.uniglow.n_mel_channels,
            self._cfg.uniglow.n_flows,
            self._cfg.uniglow.n_group,
            self._cfg.uniglow.n_wn_channels,
            self._cfg.uniglow.n_wn_layers,
            self._cfg.uniglow.wn_kernel_size,
            self.get_upsample_factor(),
        )
        self.mode = OperationMode.infer
        self.loss = UniGlowLoss(self._cfg.uniglow.stft_loss_coef)
        self.removed_weightnorm = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        if new_mode == OperationMode.training:
            self.train()
        else:
            self.eval()
        self._mode = new_mode
        self.model.mode = new_mode

    @property
    def input_types(self):
        return {
            "audio": NeuralType(('B', 'T'), AudioSignal()),
            "audio_len": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            output_dict = {
                "pred_normal_dist": NeuralType(('B', 'flowgroup', 'T'), NormalDistributionSamplesType()),
                "logdet": NeuralType(elements_type=LogDeterminantType()),
                "predicted_audio": NeuralType(('B', 'T'), AudioSignal()),
            }
            if self.mode == OperationMode.validation:
                output_dict["spec"] = NeuralType(('B', 'T', 'D'), MelSpectrogramType())
                output_dict["spec_len"] = NeuralType(('B'), LengthsType())
            return output_dict
        return {
            "audio_pred": NeuralType(('B', 'T'), AudioSignal()),
        }

    @typecheck()
    def forward(self, *, audio, audio_len):
        if self.mode != self.model.mode:
            raise ValueError(f"UniGlowModel's mode {self.mode} does not match UniGlowModule's mode {self.model.mode}")
        spec, spec_len = self.audio_to_melspec_precessor(audio, audio_len)
        tensors = self.model(spec=spec, audio=audio, sigma=self.sigma)
        if self.mode == OperationMode.training:
            return tensors  # z, logdet, audio_pred
        elif self.mode == OperationMode.validation:
            z, logdet, audio_pred = tensors
            return z, logdet, audio_pred, spec, spec_len
        return tensors

    @typecheck(
        input_types={
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "sigma": NeuralType(optional=True),
            "denoise": NeuralType(optional=True),
            "denoiser_strength": NeuralType(optional=True),
        },
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def convert_spectrogram_to_audio(
        self, spec: torch.Tensor, sigma: bool = 1.0, denoise: bool = True, denoiser_strength: float = 0.01
    ) -> torch.Tensor:
        if not self.removed_weightnorm:
            self.model.remove_weightnorm()
            self.removed_weightnorm = True
        self.mode = OperationMode.infer

        with torch.no_grad():
            audio = self.model(spec=spec, audio=None, sigma=sigma)
            if denoise:
                audio = self.denoise(audio=audio, strength=denoiser_strength)

        return audio

    def training_step(self, batch, batch_idx):
        self.mode = OperationMode.training
        audio, audio_len = batch
        z, logdet, predicted_audio = self(audio=audio, audio_len=audio_len)
        loss = self.loss(z=z, logdet=logdet, gt_audio=audio, predicted_audio=predicted_audio, sigma=self.sigma)
        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        self.mode = OperationMode.validation
        audio, audio_len = batch
        z, logdet, predicted_audio, spec, spec_len = self(audio=audio, audio_len=audio_len)
        loss = self.loss(z=z, logdet=logdet, gt_audio=audio, predicted_audio=predicted_audio, sigma=self.sigma)

        # compute average stoi score for batch
        stoi_score = 0
        sr = self._cfg.preprocessor.sample_rate
        for audio_i, audio_recon_i in zip(audio.cpu(), predicted_audio.cpu()):
            stoi_score += stoi(audio_i, audio_recon_i, sr)
        stoi_score /= audio.shape[0]

        return {
            "val_loss": loss,
            "predicted_audio": predicted_audio,
            "mel_target": spec,
            "mel_len": spec_len,
            "stoi": stoi_score,
        }

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            waveglow_log_to_tb_func(
                tb_logger,
                tuple(outputs[0].values())[:-1],
                self.global_step,
                tag="eval",
                mel_fb=self.audio_to_melspec_precessor.fb,
            )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_stoi = torch.FloatTensor([x['stoi'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'stoi': avg_stoi}
        logging.info(f"Validation summary | Epoch {self.current_epoch} | NLL {avg_loss:.2f} | STOI: {avg_stoi:.2f}")
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_uniglow",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_uniglow/versions/1.0.0rc1/files/tts_uniglow.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz, and has been tested on generating female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)
        return list_of_models

    def get_upsample_factor(self) -> int:
        """
        As the MelSpectrogram upsampling is done using interpolation, the upsampling factor is determined
        by the ratio of the MelSpectrogram length and the waveform length
        Returns:
            An integer representing the upsampling factor
        """
        audio = torch.ones(1, self._cfg.train_ds.dataset.n_segments)
        spec, spec_len = self.audio_to_melspec_precessor(audio, torch.FloatTensor([len(audio)]))
        spec = spec[:, :, :-1]
        audio = split_view(audio, self._cfg.uniglow.n_group, 1).permute(0, 2, 1)
        upsample_factor = audio.shape[2] // spec.shape[2]
        return upsample_factor

    def load_state_dict(self, state_dict, strict=True):
        """Stopgap measure to keep old checkpoints working before full model deprecation.

        This override fiddles with the state_dict in order to keep functionality from old checkpoints after the
        switch from torch_stft. It will be removed when this model is deprecated.
        """
        if 'audio_to_melspec_precessor.stft.forward_basis' in state_dict:
            logging.warning("Loading old checkpoint, defaulting to hann window.")
            window_dim = state_dict['audio_to_melspec_precessor.stft.forward_basis'].shape[-1]
            state_dict['audio_to_melspec_precessor.window'] = torch.hann_window(window_dim)
            del state_dict['audio_to_melspec_precessor.stft.forward_basis']
            del state_dict['audio_to_melspec_precessor.stft.inverse_basis']
        super().load_state_dict(state_dict, strict=strict)
