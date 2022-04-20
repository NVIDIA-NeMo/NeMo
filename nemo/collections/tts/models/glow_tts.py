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
import torch.utils.data
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.tts.helpers.helpers import log_audio_to_tb, plot_alignment_to_numpy, plot_spectrogram_to_numpy
from nemo.collections.tts.losses.glow_tts_loss import GlowTTSLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.glow_tts import GlowTTSModule
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import LengthsType, MelSpectrogramType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@dataclass
class GlowTTSConfig:
    encoder: Dict[Any, Any] = MISSING
    decoder: Dict[Any, Any] = MISSING
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    test_ds: Optional[Dict[Any, Any]] = None


@deprecated(version="1.9", explanation="GlowTTSModel will be removed. Use FastPitchModel instead.")
class GlowTTSModel(SpectrogramGenerator):
    """
    GlowTTS model used to generate spectrograms from text
    Consists of a text encoder and an invertible spectrogram decoder
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self.parser = instantiate(cfg.parser)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(GlowTTSConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.preprocessor = instantiate(self._cfg.preprocessor)

        encoder = instantiate(self._cfg.encoder)
        decoder = instantiate(self._cfg.decoder)

        self.glow_tts = GlowTTSModule(encoder, decoder, n_speakers=cfg.n_speakers, gin_channels=cfg.gin_channels)
        self.loss = GlowTTSLoss()

    def parse(self, str_input: str) -> torch.tensor:
        if str_input[-1] not in [".", "!", "?"]:
            str_input = str_input + "."

        tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(
        input_types={
            "x": NeuralType(('B', 'T'), TokenIndex()),
            "x_lengths": NeuralType(('B'), LengthsType()),
            "y": NeuralType(('B', 'D', 'T'), MelSpectrogramType(), optional=True),
            "y_lengths": NeuralType(('B'), LengthsType(), optional=True),
            "gen": NeuralType(optional=True),
            "noise_scale": NeuralType(optional=True),
            "length_scale": NeuralType(optional=True),
        }
    )
    def forward(self, *, x, x_lengths, y=None, y_lengths=None, gen=False, noise_scale=0.3, length_scale=1.0):
        if gen:
            return self.glow_tts.generate_spect(
                text=x, text_lengths=x_lengths, noise_scale=noise_scale, length_scale=length_scale
            )
        else:
            return self.glow_tts(text=x, text_lengths=x_lengths, spect=y, spect_lengths=y_lengths)

    def step(self, y, y_lengths, x, x_lengths):
        z, y_m, y_logs, logdet, logw, logw_, y_lengths, attn = self(
            x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, gen=False
        )

        l_mle, l_length, logdet = self.loss(
            z=z,
            y_m=y_m,
            y_logs=y_logs,
            logdet=logdet,
            logw=logw,
            logw_=logw_,
            x_lengths=x_lengths,
            y_lengths=y_lengths,
        )

        loss = sum([l_mle, l_length])

        return l_mle, l_length, logdet, loss, attn

    def training_step(self, batch, batch_idx):
        y, y_lengths, x, x_lengths = batch

        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        l_mle, l_length, logdet, loss, _ = self.step(y, y_lengths, x, x_lengths)

        output = {
            "loss": loss,  # required
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet},
            "log": {"loss": loss, "l_mle": l_mle, "l_length": l_length, "logdet": logdet},
        }

        return output

    def validation_step(self, batch, batch_idx):
        y, y_lengths, x, x_lengths = batch

        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        l_mle, l_length, logdet, loss, attn = self.step(y, y_lengths, x, x_lengths)

        y_gen, attn_gen = self(x=x, x_lengths=x_lengths, gen=True)

        return {
            "loss": loss,
            "l_mle": l_mle,
            "l_length": l_length,
            "logdet": logdet,
            "y": y,
            "y_gen": y_gen,
            "x": x,
            "attn": attn,
            "attn_gen": attn_gen,
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet},
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mle = torch.stack([x['l_mle'] for x in outputs]).mean()
        avg_length_loss = torch.stack([x['l_length'] for x in outputs]).mean()
        avg_logdet = torch.stack([x['logdet'] for x in outputs]).mean()
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_mle': avg_mle,
            'val_length_loss': avg_length_loss,
            'val_logdet': avg_logdet,
        }
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            separated_phonemes = "|".join([self.parser.symbols[c] for c in outputs[0]['x'][0]])
            tb_logger.add_text("separated phonemes", separated_phonemes, self.global_step)
            tb_logger.add_image(
                "real_spectrogram",
                plot_spectrogram_to_numpy(outputs[0]['y'][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            tb_logger.add_image(
                "generated_spectrogram",
                plot_spectrogram_to_numpy(outputs[0]['y_gen'][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            tb_logger.add_image(
                "alignment_for_real_sp",
                plot_alignment_to_numpy(outputs[0]['attn'][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            tb_logger.add_image(
                "alignment_for_generated_sp",
                plot_alignment_to_numpy(outputs[0]['attn_gen'][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            log_audio_to_tb(tb_logger, outputs[0]['y'][0], "true_audio_gf", self.global_step)
            log_audio_to_tb(tb_logger, outputs[0]['y_gen'][0], "generated_audio_gf", self.global_step)
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _setup_dataloader_from_config(self, cfg: DictConfig):

        if 'manifest_filepath' in cfg and cfg['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {cfg}")
            return None

        if 'augmentor' in cfg:
            augmentor = process_augmentations(cfg['augmentor'])
        else:
            augmentor = None

        dataset = _AudioTextDataset(
            manifest_filepath=cfg['manifest_filepath'],
            parser=self.parser,
            sample_rate=cfg['sample_rate'],
            int_values=cfg.get('int_values', False),
            augmentor=augmentor,
            max_duration=cfg.get('max_duration', None),
            min_duration=cfg.get('min_duration', None),
            max_utts=cfg.get('max_utts', 0),
            trim=cfg.get('trim_silence', True),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', False),
            shuffle=cfg['shuffle'],
            num_workers=cfg.get('num_workers', 0),
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def generate_spectrogram(
        self, tokens: 'torch.tensor', noise_scale: float = 0.0, length_scale: float = 1.0
    ) -> torch.tensor:

        self.eval()

        token_len = torch.tensor([tokens.shape[1]]).to(self.device)
        spect, _ = self(x=tokens, x_lengths=token_len, gen=True, noise_scale=noise_scale, length_scale=length_scale)

        return spect

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_glowtts",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_glowtts/versions/1.0.0rc1/files/tts_en_glowtts.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female English voices with an American accent.",
            class_=cls,
            aliases=["GlowTTS-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models

    def load_state_dict(self, state_dict, strict=True):
        """Stopgap measure to keep old checkpoints working before full model deprecation.

        This override fiddles with the state_dict in order to keep functionality from old checkpoints after the
        switch from torch_stft. It will be removed when this model is deprecated.
        """
        if 'preprocessor.featurizer.stft.forward_basis' in state_dict:
            logging.warning("Loading old checkpoint, defaulting to hann window.")
            window_dim = state_dict['preprocessor.featurizer.stft.forward_basis'].shape[-1]
            state_dict['preprocessor.featurizer.window'] = torch.hann_window(window_dim)
            del state_dict['preprocessor.featurizer.stft.forward_basis']
            del state_dict['preprocessor.featurizer.stft.inverse_basis']
        super().load_state_dict(state_dict, strict=strict)
