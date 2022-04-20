# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.collections.tts.losses.fastspeech2loss import DurationLoss, L2MelLoss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@dataclass
class FastSpeech2Config:
    preprocessor: Dict[Any, Any] = MISSING
    encoder: Dict[Any, Any] = MISSING
    decoder: Dict[Any, Any] = MISSING
    variance_adaptor: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


@deprecated(version="1.9", explanation="FastSpeech2Model will be removed. Please, use FastPitchModel instead.")
class FastSpeech2Model(SpectrogramGenerator):
    """FastSpeech 2 model used to convert from text (phonemes) to mel-spectrograms."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(FastSpeech2Config)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.pitch = cfg.add_pitch_predictor
        self.energy = cfg.add_energy_predictor
        self.duration_coeff = cfg.duration_coeff

        self.audio_to_melspec_preprocessor = instantiate(self._cfg.preprocessor)
        self.encoder = instantiate(self._cfg.encoder)
        self.mel_decoder = instantiate(self._cfg.decoder)
        self.variance_adapter = instantiate(self._cfg.variance_adaptor)
        self.loss = L2MelLoss()
        self.mseloss = torch.nn.MSELoss()
        self.durationloss = DurationLoss()

        self.log_train_images = False

        # Parser and mappings are used for inference only.
        self.parser = parsers.make_parser(name='en')
        if 'mappings_filepath' in cfg:
            mappings_filepath = cfg.get('mappings_filepath')
        else:
            logging.error(
                "ERROR: You must specify a mappings.json file in the config file under model.mappings_filepath."
            )
        mappings_filepath = self.register_artifact('mappings_filepath', mappings_filepath)
        with open(mappings_filepath, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
            self.word2phones = mappings['word2phones']
            self.phone2idx = mappings['phone2idx']

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_length": NeuralType(('B'), LengthsType()),
            "spec_len": NeuralType(('B'), LengthsType(), optional=True),
            "durations": NeuralType(('B', 'T'), TokenDurationType(), optional=True),
            "pitch": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
            "energies": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
        },
        output_types={
            "mel_spec": NeuralType(('B', 'T', 'C'), MelSpectrogramType()),
            "log_dur_preds": NeuralType(('B', 'T'), TokenLogDurationType(), optional=True),
            "pitch_preds": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
            "energy_preds": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
            "encoded_text_mask": NeuralType(('B', 'T', 'D'), MaskType()),
        },
    )
    def forward(self, *, text, text_length, spec_len=None, durations=None, pitch=None, energies=None):
        encoded_text, encoded_text_mask = self.encoder(text=text, text_length=text_length)
        aligned_text, log_dur_preds, pitch_preds, energy_preds, spec_len = self.variance_adapter(
            x=encoded_text,
            x_len=text_length,
            dur_target=durations,
            pitch_target=pitch,
            energy_target=energies,
            spec_len=spec_len,
        )
        mel = self.mel_decoder(decoder_input=aligned_text, lengths=spec_len)
        return mel, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask

    def training_step(self, batch, batch_idx):
        f, fl, t, tl, durations, pitch, energies = batch
        spec, spec_len = self.audio_to_melspec_preprocessor(f, fl)
        mel, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask = self(
            text=t, text_length=tl, spec_len=spec_len, durations=durations, pitch=pitch, energies=energies
        )
        total_loss = self.loss(
            spec_pred=mel.transpose(1, 2), spec_target=spec, spec_target_len=spec_len, pad_value=-11.52
        )
        self.log(name="train_mel_loss", value=total_loss.clone().detach())

        # Duration prediction loss
        dur_loss = self.durationloss(
            log_duration_pred=log_dur_preds, duration_target=durations.float(), mask=encoded_text_mask
        )
        dur_loss *= self.duration_coeff
        self.log(name="train_dur_loss", value=dur_loss)
        total_loss += dur_loss

        # Pitch prediction loss
        if self.pitch:
            pitch_loss = self.mseloss(pitch_preds, pitch)
            total_loss += pitch_loss
            self.log(name="train_pitch_loss", value=pitch_loss)

        # Energy prediction loss
        if self.energy:
            energy_loss = self.mseloss(energy_preds, energies)
            total_loss += energy_loss
            self.log(name="train_energy_loss", value=energy_loss)
        self.log(name="train_loss", value=total_loss)
        return {"loss": total_loss, "outputs": [spec, mel]}

    def training_epoch_end(self, outputs):
        if self.log_train_images and self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            spec_target, spec_predict = outputs[0]["outputs"]
            tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
            self.log_train_images = False

            return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        f, fl, t, tl, _, _, _ = batch
        spec, spec_len = self.audio_to_melspec_preprocessor(f, fl)
        mel, _, _, _, _ = self(text=t, text_length=tl, spec_len=spec_len)
        loss = self.loss(spec_pred=mel.transpose(1, 2), spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)
        return {
            "val_loss": loss,
            "mel_target": spec,
            "mel_pred": mel,
        }

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            _, spec_target, spec_predict = outputs[0].values()
            tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss, sync_dist=True)

        self.log_train_images = True

    def parse(self, str_input: str, additional_word2phones=None) -> torch.tensor:
        """
        Parses text input and converts them to phoneme indices.

        str_input (str): The input text to be converted.
        additional_word2phones (dict): Optional dictionary mapping words to phonemes for updating the model's
            word2phones.  This will not overwrite the existing dictionary, just update it with OOV or new mappings.
            Defaults to None, which will keep the existing mapping.
        """
        # Update model's word2phones if applicable
        if additional_word2phones is not None:
            self.word2phones.update(additional_word2phones)

        # Convert text -> normalized text -> list of phones per word -> indices
        if str_input[-1] not in [".", "!", "?"]:
            str_input = str_input + "."
        norm_text = re.findall(r"""[\w']+|[.,!?;"]""", self.parser._normalize(str_input))

        try:
            phones = [self.word2phones[t] for t in norm_text]
        except KeyError as error:
            logging.error(
                f"ERROR: The following word in the input is not in the model's dictionary and could not be converted"
                f" to phonemes: ({error}).\n"
                f"You can pass in an `additional_word2phones` dictionary with a conversion for"
                f" this word, e.g. {{'{error}': \['phone1', 'phone2', ...\]}} to update the model's mapping."
            )
            raise

        tokens = []
        for phone_list in phones:
            inds = [self.phone2idx[p] for p in phone_list]
            tokens += inds

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    @typecheck(output_types={"spect": NeuralType(('B', 'C', 'T'), MelSpectrogramType())})
    def generate_spectrogram(self, tokens: torch.Tensor) -> torch.Tensor:
        self.eval()
        token_len = torch.tensor([tokens.shape[1]]).to(self.device)
        spect, *_ = self(text=tokens, text_length=token_len)

        return spect.transpose(1, 2)

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
                with open_dict(cfg.dataloader_params):
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
        This method returns a list of pre-trained models which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_fastspeech2",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_fastspeech_2/versions/1.0.0/files/tts_en_fastspeech2.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female English voices with an American accent.",
            class_=cls,
            aliases=["FastSpeech2-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models
