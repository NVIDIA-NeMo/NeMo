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

from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from nemo.collections.asr.data.audio_to_text import FastPitchDataset
from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy, regulate_len
from nemo.collections.tts.losses.fastpitchloss import DurationLoss, PitchLoss
from nemo.collections.tts.losses.fastspeech2loss import L1MelLoss
from nemo.collections.tts.losses.hifigan_losses import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss
from nemo.collections.tts.models.base import TextToWaveform
from nemo.collections.tts.modules.hifigan_modules import MultiPeriodDiscriminator, MultiScaleDiscriminator
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    RegressionValuesType,
    TokenDurationType,
    TokenIndex,
    TokenLogDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import NoamAnnealing
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@dataclass
class FastPitchHifiGanE2EConfig:
    parser: Dict[Any, Any] = MISSING
    preprocessor: Dict[Any, Any] = MISSING
    input_fft: Dict[Any, Any] = MISSING
    output_fft: Dict[Any, Any] = MISSING
    duration_predictor: Dict[Any, Any] = MISSING
    pitch_predictor: Dict[Any, Any] = MISSING


@deprecated(version="1.9", explanation="FastPitchHifiGanE2EModel will be removed.")
class FastPitchHifiGanE2EModel(TextToWaveform):
    """An end-to-end speech synthesis model based on FastPitch and HiFiGan that converts strings to audio without using
    the intermediate mel spectrogram representation.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self._parser = parsers.make_parser(
            labels=cfg.labels,
            name='en',
            unk_id=-1,
            blank_id=-1,
            do_normalize=True,
            abbreviation_version="fastpitch",
            make_table=False,
        )

        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(FastPitchHifiGanE2EConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.preprocessor = instantiate(cfg.preprocessor)
        self.melspec_fn = instantiate(cfg.preprocessor, highfreq=None, use_grads=True)

        self.encoder = instantiate(cfg.input_fft)
        self.duration_predictor = instantiate(cfg.duration_predictor)
        self.pitch_predictor = instantiate(cfg.pitch_predictor)

        self.generator = instantiate(cfg.generator)
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiscaledisc = MultiScaleDiscriminator()
        self.mel_val_loss = L1MelLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()

        self.max_token_duration = cfg.max_token_duration

        self.pitch_emb = torch.nn.Conv1d(
            1,
            cfg.symbols_embedding_dim,
            kernel_size=cfg.pitch_embedding_kernel_size,
            padding=int((cfg.pitch_embedding_kernel_size - 1) / 2),
        )

        # Store values precomputed from training data for convenience
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.pitchloss = PitchLoss()
        self.durationloss = DurationLoss()

        self.mel_loss_coeff = cfg.mel_loss_coeff

        self.log_train_images = False
        self.logged_real_samples = False
        self._tb_logger = None
        self.hann_window = None
        self.splice_length = cfg.splice_length
        self.sample_rate = cfg.sample_rate
        self.hop_size = cfg.hop_size

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

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser

        self._parser = parsers.make_parser(
            labels=self._cfg.labels,
            name='en',
            unk_id=-1,
            blank_id=-1,
            do_normalize=True,
            abbreviation_version="fastpitch",
            make_table=False,
        )
        return self._parser

    def parse(self, str_input: str) -> torch.tensor:
        if str_input[-1] not in [".", "!", "?"]:
            str_input = str_input + "."

        tokens = self.parser(str_input)

        x = torch.tensor(tokens).unsqueeze_(0).long().to(self.device)
        return x

    def configure_optimizers(self):
        gen_params = chain(
            self.pitch_emb.parameters(),
            self.encoder.parameters(),
            self.duration_predictor.parameters(),
            self.pitch_predictor.parameters(),
            self.generator.parameters(),
        )
        disc_params = chain(self.multiscaledisc.parameters(), self.multiperioddisc.parameters())
        opt1 = torch.optim.AdamW(disc_params, lr=self._cfg.lr)
        opt2 = torch.optim.AdamW(gen_params, lr=self._cfg.lr)
        num_procs = self._trainer.num_devices * self._trainer.num_nodes
        num_samples = len(self._train_dl.dataset)
        batch_size = self._train_dl.batch_size
        iter_per_epoch = np.ceil(num_samples / (num_procs * batch_size))
        max_steps = iter_per_epoch * self._trainer.max_epochs
        logging.info(f"MAX STEPS: {max_steps}")
        sch1 = NoamAnnealing(opt1, d_model=1, warmup_steps=1000, max_steps=max_steps, last_epoch=-1)
        sch1_dict = {
            'scheduler': sch1,
            'interval': 'step',
        }
        sch2 = NoamAnnealing(opt2, d_model=1, warmup_steps=1000, max_steps=max_steps, last_epoch=-1)
        sch2_dict = {
            'scheduler': sch2,
            'interval': 'step',
        }
        return [opt1, opt2], [sch1_dict, sch2_dict]

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "durs": NeuralType(('B', 'T'), TokenDurationType(), optional=True),
            "pitch": NeuralType(('B', 'T'), RegressionValuesType(), optional=True),
            "pace": NeuralType(optional=True),
            "splice": NeuralType(optional=True),
        },
        output_types={
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal()),
            "splices": NeuralType(),
            "log_dur_preds": NeuralType(('B', 'T'), TokenLogDurationType()),
            "pitch_preds": NeuralType(('B', 'T'), RegressionValuesType()),
        },
    )
    def forward(self, *, text, durs=None, pitch=None, pace=1.0, splice=True):
        if self.training:
            assert durs is not None
            assert pitch is not None

        # Input FFT
        enc_out, enc_mask = self.encoder(input=text, conditioning=0)

        # Embedded for predictors
        pred_enc_out, pred_enc_mask = enc_out, enc_mask

        # Predict durations
        log_durs_predicted = self.duration_predictor(pred_enc_out, pred_enc_mask)
        durs_predicted = torch.clamp(torch.exp(log_durs_predicted) - 1, 0, self.max_token_duration)

        # Predict pitch
        pitch_predicted = self.pitch_predictor(enc_out, enc_mask)
        if pitch is None:
            pitch_emb = self.pitch_emb(pitch_predicted.unsqueeze(1))
        else:
            pitch_emb = self.pitch_emb(pitch.unsqueeze(1))
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        if durs is None:
            len_regulated, dec_lens = regulate_len(durs_predicted, enc_out, pace)
        else:
            len_regulated, dec_lens = regulate_len(durs, enc_out, pace)

        gen_in = len_regulated
        splices = []
        if splice:
            output = []
            for i, sample in enumerate(len_regulated):
                start = np.random.randint(low=0, high=min(int(sample.size(0)), int(dec_lens[i])) - self.splice_length)
                # Splice generated spec
                output.append(sample[start : start + self.splice_length, :])
                splices.append(start)
            gen_in = torch.stack(output)

        output = self.generator(x=gen_in.transpose(1, 2))

        return output, torch.tensor(splices), log_durs_predicted, pitch_predicted

    def training_step(self, batch, batch_idx, optimizer_idx):
        audio, _, text, text_lens, durs, pitch, _ = batch

        # train discriminator
        if optimizer_idx == 0:
            with torch.no_grad():
                audio_pred, splices, _, _ = self(text=text, durs=durs, pitch=pitch)
                real_audio = []
                for i, splice in enumerate(splices):
                    real_audio.append(audio[i, splice * self.hop_size : (splice + self.splice_length) * self.hop_size])
                real_audio = torch.stack(real_audio).unsqueeze(1)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(real_audio, audio_pred)
            real_score_ms, gen_score_ms, _, _ = self.multiscaledisc(real_audio, audio_pred)

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
            audio_pred, splices, log_dur_preds, pitch_preds = self(text=text, durs=durs, pitch=pitch)
            real_audio = []
            for i, splice in enumerate(splices):
                real_audio.append(audio[i, splice * self.hop_size : (splice + self.splice_length) * self.hop_size])
            real_audio = torch.stack(real_audio).unsqueeze(1)

            dur_loss = self.durationloss(log_durs_predicted=log_dur_preds, durs_tgt=durs, len=text_lens)
            pitch_loss = self.pitchloss(pitch_predicted=pitch_preds, pitch_tgt=pitch,)

            # Do HiFiGAN generator loss
            audio_length = torch.tensor([self.splice_length * self.hop_size for _ in range(real_audio.shape[0])]).to(
                real_audio.device
            )
            real_spliced_spec, _ = self.melspec_fn(real_audio.squeeze(), audio_length)
            pred_spliced_spec, _ = self.melspec_fn(audio_pred.squeeze(), audio_length)
            loss_mel = torch.nn.functional.l1_loss(real_spliced_spec, pred_spliced_spec)
            loss_mel *= self.mel_loss_coeff
            _, gen_score_mp, _, _ = self.multiperioddisc(real_audio, audio_pred)
            _, gen_score_ms, _, _ = self.multiscaledisc(y=real_audio, y_hat=audio_pred)
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_ms, list_loss_gen_ms = self.gen_loss(disc_outputs=gen_score_ms)
            loss_gen_mp /= len(list_loss_gen_mp)
            loss_gen_ms /= len(list_loss_gen_ms)
            total_loss = loss_gen_mp + loss_gen_ms + loss_mel
            total_loss += dur_loss
            total_loss += pitch_loss

            self.log(name="loss_gen_mel", value=loss_mel)
            self.log(name="loss_gen_disc", value=loss_gen_mp + loss_gen_ms)
            self.log(name="loss_gen_disc_mp", value=loss_gen_mp)
            self.log(name="loss_gen_disc_ms", value=loss_gen_ms)
            self.log(name="loss_gen_duration", value=dur_loss)
            self.log(name="loss_gen_pitch", value=pitch_loss)

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
        audio, audio_lens, text, _, _, _, _ = batch
        mels, mel_lens = self.preprocessor(audio, audio_lens)

        audio_pred, _, log_durs_predicted, _ = self(text=text, durs=None, pitch=None, splice=False)
        audio_length = torch.sum(torch.clamp(torch.exp(log_durs_predicted - 1), 0), axis=1)
        audio_pred.squeeze_()
        pred_spec, _ = self.melspec_fn(audio_pred, audio_length)
        loss = self.mel_val_loss(
            spec_pred=pred_spec, spec_target=mels, spec_target_len=mel_lens, pad_value=-11.52, transpose=False
        )

        return {
            "val_loss": loss,
            "audio_target": audio if batch_idx == 0 else None,
            "audio_pred": audio_pred.squeeze() if batch_idx == 0 else None,
        }

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

    def _loader(self, cfg):
        dataset = FastPitchDataset(
            manifest_filepath=cfg['manifest_filepath'],
            parser=self.parser,
            sample_rate=cfg['sample_rate'],
            int_values=cfg.get('int_values', False),
            max_duration=cfg.get('max_duration', None),
            min_duration=cfg.get('min_duration', None),
            max_utts=cfg.get('max_utts', 0),
            trim=cfg.get('trim_silence', True),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', True),
            shuffle=cfg['shuffle'],
            num_workers=cfg.get('num_workers', 16),
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_e2e_fastpitchhifigan",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_e2e_fastpitchhifigan/versions/1.0.0/files/tts_en_e2e_fastpitchhifigan.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz with and can be used to generate female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)

        return list_of_models

    def convert_text_to_waveform(self, *, tokens):
        """
        Accepts tokens returned from self.parse() and returns a list of tensors. Note: The tensors in the list can have
        different lengths.
        """
        self.eval()
        audio, _, log_dur_pred, _ = self(text=tokens, splice=False)
        audio = audio.squeeze(1)
        durations = torch.sum(torch.clamp(torch.exp(log_dur_pred) - 1, 0, self.max_token_duration), 1).to(torch.int)
        audio_list = []
        for i, sample in enumerate(audio):
            audio_list.append(sample[: durations[i] * self.hop_size])

        return audio_list
