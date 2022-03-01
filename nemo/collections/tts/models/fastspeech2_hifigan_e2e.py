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
from itertools import chain

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy
from nemo.collections.tts.losses.fastspeech2loss import DurationLoss, L1MelLoss
from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    LengthsType,
    MaskType,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import NoamAnnealing
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@deprecated(version="1.8", explanation="FastSpeech2HifiGanE2EModel will be removed.")
class FastSpeech2HifiGanE2EModel(TextToWaveform):
    """An end-to-end speech synthesis model based on FastSpeech2 and HiFiGan that converts strings to audio without
    using the intermediate mel spectrogram representation."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(cfg.preprocessor)
        self.encoder = instantiate(cfg.encoder)
        self.variance_adapter = instantiate(cfg.variance_adaptor)

        self.generator = instantiate(cfg.generator)
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiscaledisc = MultiScaleDiscriminator()

        self.melspec_fn = instantiate(cfg.preprocessor, highfreq=None, use_grads=True)
        self.mel_val_loss = L1MelLoss()
        self.durationloss = DurationLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.mseloss = torch.nn.MSELoss()

        self.energy = cfg.add_energy_predictor
        self.pitch = cfg.add_pitch_predictor
        self.mel_loss_coeff = cfg.mel_loss_coeff
        self.pitch_loss_coeff = cfg.pitch_loss_coeff
        self.energy_loss_coeff = cfg.energy_loss_coeff
        self.splice_length = cfg.splice_length

        self.use_energy_pred = False
        self.use_pitch_pred = False
        self.log_train_images = False
        self.logged_real_samples = False
        self._tb_logger = None
        self.sample_rate = cfg.sample_rate
        self.hop_size = cfg.hop_size

        # Parser and mappings are used for inference only.
        self.parser = parsers.make_parser(name='en')
        if 'mappings_filepath' in cfg:
            mappings_filepath = cfg.get('mappings_filepath')
        else:
            logging.error(
                "ERROR: You must specify a mappings.json file in the config file under model.mappings_filepath."
            )
        mappings_filepath = self.register_artifact('mappings_filepath', mappings_filepath)
        with open(mappings_filepath, 'r') as f:
            mappings = json.load(f)
            self.word2phones = mappings['word2phones']
            self.phone2idx = mappings['phone2idx']

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            if self.logger is None and self.logger.experiment is None:
                return None
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            self._tb_logger = tb_logger
        return self._tb_logger

    def configure_optimizers(self):
        gen_params = chain(self.encoder.parameters(), self.generator.parameters(), self.variance_adapter.parameters(),)
        disc_params = chain(self.multiscaledisc.parameters(), self.multiperioddisc.parameters())
        opt1 = torch.optim.AdamW(disc_params, lr=self._cfg.lr)
        opt2 = torch.optim.AdamW(gen_params, lr=self._cfg.lr)
        num_procs = self._trainer.num_gpus * self._trainer.num_nodes
        num_samples = len(self._train_dl.dataset)
        batch_size = self._train_dl.batch_size
        iter_per_epoch = np.ceil(num_samples / (num_procs * batch_size))
        max_steps = iter_per_epoch * self._trainer.max_epochs
        logging.info(f"MAX STEPS: {max_steps}")
        sch1 = NoamAnnealing(opt1, d_model=256, warmup_steps=3000, max_steps=max_steps, min_lr=1e-5)
        sch1_dict = {
            'scheduler': sch1,
            'interval': 'step',
        }
        sch2 = NoamAnnealing(opt2, d_model=256, warmup_steps=3000, max_steps=max_steps, min_lr=1e-5)
        sch2_dict = {
            'scheduler': sch2,
            'interval': 'step',
        }
        return [opt1, opt2], [sch1_dict, sch2_dict]

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_length": NeuralType(('B'), LengthsType()),
            "splice": NeuralType(optional=True),
            "spec_len": NeuralType(('B'), LengthsType(), optional=True),
            "durations": NeuralType(('B', 'T'), TokenDurationType(), optional=True),
            "pitch": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
            "energies": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
        },
        output_types={
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal()),
            "splices": NeuralType(),
            "log_dur_preds": NeuralType(('B', 'T'), TokenLogDurationType()),
            "pitch_preds": NeuralType(('B', 'T'), RegressionValuesType()),
            "energy_preds": NeuralType(('B', 'T'), RegressionValuesType()),
            "encoded_text_mask": NeuralType(('B', 'T', 'D'), MaskType()),
        },
    )
    def forward(self, *, text, text_length, splice=True, durations=None, pitch=None, energies=None, spec_len=None):
        encoded_text, encoded_text_mask = self.encoder(text=text, text_length=text_length)

        context, log_dur_preds, pitch_preds, energy_preds, spec_len = self.variance_adapter(
            x=encoded_text,
            x_len=text_length,
            dur_target=durations,
            pitch_target=pitch,
            energy_target=energies,
            spec_len=spec_len,
        )

        gen_in = context
        splices = []
        if splice:
            # Splice generated spec
            output = []
            for i, sample in enumerate(context):
                start = np.random.randint(low=0, high=min(int(sample.size(0)), int(spec_len[i])) - self.splice_length)
                output.append(sample[start : start + self.splice_length, :])
                splices.append(start)
            gen_in = torch.stack(output)

        output = self.generator(x=gen_in.transpose(1, 2))

        return output, torch.tensor(splices), log_dur_preds, pitch_preds, energy_preds, encoded_text_mask

    def training_step(self, batch, batch_idx, optimizer_idx):
        f, fl, t, tl, durations, pitch, energies = batch
        _, spec_len = self.audio_to_melspec_precessor(f, fl)

        # train discriminator
        if optimizer_idx == 0:
            with torch.no_grad():
                audio_pred, splices, _, _, _, _ = self(
                    spec_len=spec_len,
                    text=t,
                    text_length=tl,
                    durations=durations,
                    pitch=pitch if not self.use_pitch_pred else None,
                    energies=energies if not self.use_energy_pred else None,
                )
                real_audio = []
                for i, splice in enumerate(splices):
                    real_audio.append(f[i, splice * self.hop_size : (splice + self.splice_length) * self.hop_size])
                real_audio = torch.stack(real_audio).unsqueeze(1)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(real_audio, audio_pred)
            real_score_ms, gen_score_ms, _, _ = self.multiscaledisc(y=real_audio, y_hat=audio_pred)

            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_ms, loss_ms_real, _ = self.disc_loss(
                disc_real_outputs=real_score_ms, disc_generated_outputs=gen_score_ms
            )
            loss_mp /= len(loss_mp_real)
            loss_ms /= len(loss_ms_real)
            loss_disc = loss_mp + loss_ms

            self.log("loss_discriminator", loss_disc, prog_bar=True)
            self.log("loss_discriminator_ms", loss_ms)
            self.log("loss_discriminator_mp", loss_mp)
            return loss_disc

        # train generator
        elif optimizer_idx == 1:
            audio_pred, splices, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask = self(
                spec_len=spec_len,
                text=t,
                text_length=tl,
                durations=durations,
                pitch=pitch if not self.use_pitch_pred else None,
                energies=energies if not self.use_energy_pred else None,
            )
            real_audio = []
            for i, splice in enumerate(splices):
                real_audio.append(f[i, splice * self.hop_size : (splice + self.splice_length) * self.hop_size])
            real_audio = torch.stack(real_audio).unsqueeze(1)

            # Do HiFiGAN generator loss
            audio_length = torch.tensor([self.splice_length * self.hop_size for _ in range(real_audio.shape[0])]).to(
                real_audio.device
            )
            real_spliced_spec, _ = self.melspec_fn(real_audio.squeeze(), seq_len=audio_length)
            pred_spliced_spec, _ = self.melspec_fn(audio_pred.squeeze(), seq_len=audio_length)
            loss_mel = torch.nn.functional.l1_loss(real_spliced_spec, pred_spliced_spec)
            loss_mel *= self.mel_loss_coeff
            _, gen_score_mp, real_feat_mp, gen_feat_mp = self.multiperioddisc(real_audio, audio_pred)
            _, gen_score_ms, real_feat_ms, gen_feat_ms = self.multiscaledisc(y=real_audio, y_hat=audio_pred)
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_ms, list_loss_gen_ms = self.gen_loss(disc_outputs=gen_score_ms)
            loss_gen_mp /= len(list_loss_gen_mp)
            loss_gen_ms /= len(list_loss_gen_ms)
            total_loss = loss_gen_mp + loss_gen_ms + loss_mel
            loss_feat_mp = self.feat_matching_loss(fmap_r=real_feat_mp, fmap_g=gen_feat_mp)
            loss_feat_ms = self.feat_matching_loss(fmap_r=real_feat_ms, fmap_g=gen_feat_ms)
            total_loss += loss_feat_mp + loss_feat_ms
            self.log(name="loss_gen_disc_feat", value=loss_feat_mp + loss_feat_ms)
            self.log(name="loss_gen_disc_feat_ms", value=loss_feat_ms)
            self.log(name="loss_gen_disc_feat_mp", value=loss_feat_mp)

            self.log(name="loss_gen_mel", value=loss_mel)
            self.log(name="loss_gen_disc", value=loss_gen_mp + loss_gen_ms)
            self.log(name="loss_gen_disc_mp", value=loss_gen_mp)
            self.log(name="loss_gen_disc_ms", value=loss_gen_ms)

            dur_loss = self.durationloss(
                log_duration_pred=log_dur_preds, duration_target=durations.float(), mask=encoded_text_mask
            )
            self.log(name="loss_gen_duration", value=dur_loss)
            total_loss += dur_loss
            if self.pitch:
                pitch_loss = self.mseloss(pitch_preds, pitch.float()) * self.pitch_loss_coeff
                total_loss += pitch_loss
                self.log(name="loss_gen_pitch", value=pitch_loss)
            if self.energy:
                energy_loss = self.mseloss(energy_preds, energies) * self.energy_loss_coeff
                total_loss += energy_loss
                self.log(name="loss_gen_energy", value=energy_loss)

            # Log images to tensorboard
            if self.log_train_images:
                self.log_train_images = False
                if self.logger is not None and self.logger.experiment is not None:
                    self.tb_logger.add_image(
                        "train_mel_target",
                        plot_spectrogram_to_numpy(real_spliced_spec[0].data.cpu().numpy()),
                        self.global_step,
                        dataformats="HWC",
                    )
                    spec_predict = pred_spliced_spec[0].data.cpu().numpy()
                    self.tb_logger.add_image(
                        "train_mel_predicted",
                        plot_spectrogram_to_numpy(spec_predict),
                        self.global_step,
                        dataformats="HWC",
                    )
            self.log(name="loss_gen", prog_bar=True, value=total_loss)
            return total_loss

    def validation_step(self, batch, batch_idx):
        f, fl, t, tl, _, _, _ = batch
        spec, spec_len = self.audio_to_melspec_precessor(f, fl)
        audio_pred, _, _, _, _, _ = self(spec_len=spec_len, text=t, text_length=tl, splice=False)
        audio_pred.squeeze_()
        pred_spec, _ = self.melspec_fn(audio_pred, seq_len=spec_len)
        loss = self.mel_val_loss(spec_pred=pred_spec, spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)

        return {
            "val_loss": loss,
            "audio_target": f.squeeze() if batch_idx == 0 else None,
            "audio_pred": audio_pred if batch_idx == 0 else None,
        }

    def on_train_epoch_start(self):
        # Switch to using energy predictions after 50% of training
        if not self.use_energy_pred and self.current_epoch >= np.ceil(0.5 * self._trainer.max_epochs):
            logging.info(f"Using energy predictions after epoch: {self.current_epoch}")
            self.use_energy_pred = True

        # Switch to using pitch predictions after 62.5% of training
        if not self.use_pitch_pred and self.current_epoch >= np.ceil(0.625 * self._trainer.max_epochs):
            logging.info(f"Using pitch predictions after epoch: {self.current_epoch}")
            self.use_pitch_pred = True

    def validation_epoch_end(self, outputs):
        if self.tb_logger is not None:
            _, audio_target, audio_predict = outputs[0].values()
            if not self.logged_real_samples:
                self.tb_logger.add_audio("val_target", audio_target[0].data.cpu(), self.global_step, self.sample_rate)
                self.logged_real_samples = True
            audio_predict = audio_predict[0].data.cpu()
            self.tb_logger.add_audio("val_pred", audio_predict, self.global_step, self.sample_rate)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss, sync_dist=True)

        self.log_train_images = True

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

    def convert_text_to_waveform(self, *, tokens):
        """
        Accepts tokens returned from self.parse() and returns a list of tensors. Note: The tensors in the list can have
        different lengths.
        """
        self.eval()
        token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
        audio, _, log_dur_pred, _, _, _ = self(text=tokens, text_length=token_len, splice=False)
        audio = audio.squeeze(1)
        durations = torch.sum(torch.exp(log_dur_pred) - 1, 1).to(torch.int)
        audio_list = []
        for i, sample in enumerate(audio):
            audio_list.append(sample[: durations[i] * self.hop_size])

        return audio_list

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_e2e_fastspeech2hifigan",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_e2e_fastspeech2hifigan/versions/1.0.0/files/tts_en_e2e_fastspeech2hifigan.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models
