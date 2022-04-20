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

from typing import List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths, plot_spectrogram_to_numpy
from nemo.collections.tts.models.base import Vocoder
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.core.optim.lr_scheduler import CosineAnnealing
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@deprecated(version="1.9", explanation="MelGanModel will be removed. Use HifiGanModel instead.")
class MelGanModel(Vocoder):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        """NeMo Model that implement Full-band MelGAN as described in https://arxiv.org/abs/2005.05106
        """
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.generator = instantiate(self._cfg.generator)
        if "discriminator" in self._cfg:
            self.discriminator = instantiate(self._cfg.discriminator)

        self.loss = instantiate(self._cfg.loss)
        self.mse_loss = torch.nn.MSELoss()  # Used for LSE GAN loss

        self.start_training_disc = False
        self.logged_real_samples = False
        self.sample_rate = self._cfg.preprocessor.sample_rate

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3, eps=1e-07, amsgrad=True)
        opt2 = torch.optim.Adam(self.generator.parameters(), lr=1e-3, eps=1e-07, amsgrad=True)
        num_procs = self._trainer.num_devices * self._trainer.num_nodes
        num_samples = len(self._train_dl.dataset)
        batch_size = self._train_dl.batch_size
        iter_per_epoch = np.ceil(num_samples / (num_procs * batch_size))
        max_steps = iter_per_epoch * self._trainer.max_epochs
        logging.info(f"MAX STEPS: {max_steps}")
        sch1 = CosineAnnealing(
            opt1, max_steps=max_steps, min_lr=1e-5, warmup_steps=np.ceil(0.2 * max_steps)
        )  # Use warmup to delay start
        sch1_dict = {
            'scheduler': sch1,
            'interval': 'step',
        }
        sch2 = CosineAnnealing(opt2, max_steps=max_steps, min_lr=1e-5)
        sch2_dict = {
            'scheduler': sch2,
            'interval': 'step',
        }
        return [opt1, opt2], [sch1_dict, sch2_dict]

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal(self.sample_rate)),
        }

    @typecheck()
    def forward(self, *, spec):
        """
        Runs the generator, for inputs and outputs see input_types, and output_types
        """
        return self.generator(spec=spec)

    @typecheck(output_types={"audio": NeuralType(('B', 'T'), AudioSignal())})
    def convert_spectrogram_to_audio(self, spec: 'torch.tensor') -> 'torch.tensor':
        return self(spec=spec).squeeze(1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        audio, audio_len = batch
        spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
        audio_pred = self(spec=spec)

        # TODO: Lightning has a bug in 1.1.0, just always log something as a workaround
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5063
        self.log("Dummy", 0.0, logger=False)

        # train discriminator
        if optimizer_idx == 0 and self.start_training_disc:
            fake_score = self.discriminator(x=audio_pred.detach())[0]
            real_score = self.discriminator(x=audio.unsqueeze(1))[0]

            loss_disc_real = [0] * len(fake_score)
            loss_disc_fake = [0] * len(fake_score)
            for i, _ in enumerate(fake_score):
                loss_disc_real[i] += self.mse_loss(real_score[i], real_score[i].new_ones(real_score[i].size()))
                loss_disc_fake[i] += torch.mean(fake_score[i] ** 2)
            sum_loss_dis = sum(loss_disc_real) + sum(loss_disc_fake)
            sum_loss_dis /= len(fake_score)

            self.log("loss_discriminator", sum_loss_dis, prog_bar=True, sync_dist=True)
            for i, _ in enumerate(fake_score):
                self.log(f"loss_discriminator_real_{i}", loss_disc_real[i] / len(fake_score), sync_dist=True)
                self.log(f"loss_discriminator_fake_{i}", loss_disc_fake[i] / len(fake_score), sync_dist=True)
            return sum_loss_dis

        # train generator
        elif optimizer_idx == 1:
            loss = 0

            # full-band loss
            sc_loss, mag_loss = self.loss(x=audio_pred.squeeze(1), y=audio)
            loss_feat = sum(sc_loss) + sum(mag_loss)
            loss_feat /= len(sc_loss)
            loss += loss_feat

            if self.start_training_disc:
                fake_score = self.discriminator(x=audio_pred)[0]

                loss_gan = [0] * len(fake_score)
                for i, scale in enumerate(fake_score):
                    loss_gan[i] += self.mse_loss(scale, scale.new_ones(scale.size()))

                sum_loss_gan = sum(loss_gan) / len(fake_score)

                loss += sum_loss_gan

            self.log("loss_generator", loss, sync_dist=True, prog_bar=True)
            if self.start_training_disc:
                self.log("loss_generator_gan_loss", sum_loss_gan, sync_dist=True)
                for i, _ in enumerate(fake_score):
                    self.log(
                        f"loss_generator_gan_loss_{i}", loss_gan[i] / len(fake_score), sync_dist=True,
                    )
            self.log("loss_generator_feat_loss", loss_feat, sync_dist=True)
            self.log("loss_generator_feat_loss_fb_sc", sum(sc_loss) / len(sc_loss), sync_dist=True)
            self.log("loss_generator_feat_loss_fb_mag", sum(mag_loss) / len(sc_loss), sync_dist=True)
            for i, _ in enumerate(sc_loss):
                self.log(f"loss_generator_feat_loss_fb_sc_{i}", sc_loss[i] / len(sc_loss), sync_dist=True)
                self.log(f"loss_generator_feat_loss_fb_mag_{i}", mag_loss[i] / len(sc_loss), sync_dist=True)

            return loss
        return None

    def validation_step(self, batch, batch_idx):
        audio, audio_len = batch
        with torch.no_grad():
            spec, _ = self.audio_to_melspec_precessor(audio, audio_len)
            audio_pred = self(spec=spec)

            loss = 0
            loss_dict = {}
            spec_pred, _ = self.audio_to_melspec_precessor(audio_pred.squeeze(1), audio_len)

            # Ensure that audio len is consistent between audio_pred and audio
            # For SC Norm loss, we can just zero out
            # For Mag L1 loss, we need to mask
            if audio_pred.shape[-1] < audio.shape[-1]:
                # prediction audio is less than audio, pad predicted audio to real audio
                pad_amount = audio.shape[-1] - audio_pred.shape[-1]
                audio_pred = torch.nn.functional.pad(audio_pred, (0, pad_amount), value=0.0)
            else:
                # prediction audio is larger than audio, slice predicted audio to real audio
                audio_pred = audio_pred[:, :, : audio.shape[1]]

            mask = ~get_mask_from_lengths(audio_len, max_len=torch.max(audio_len))
            mask = mask.unsqueeze(1)
            audio_pred.data.masked_fill_(mask, 0.0)

            # full-band loss
            sc_loss, mag_loss = self.loss(x=audio_pred.squeeze(1), y=audio, input_lengths=audio_len)
            loss_feat = (sum(sc_loss) + sum(mag_loss)) / len(sc_loss)
            loss_dict["sc_loss"] = sc_loss
            loss_dict["mag_loss"] = mag_loss

            loss += loss_feat
            loss_dict["loss_feat"] = loss_feat

            if self.start_training_disc:
                fake_score = self.discriminator(x=audio_pred)[0]

                loss_gen = [0] * len(fake_score)
                for i, scale in enumerate(fake_score):
                    loss_gen[i] += self.mse_loss(scale, scale.new_ones(scale.size()))

                loss_dict["gan_loss"] = loss_gen
                loss += sum(loss_gen) / len(fake_score)

        if not self.logged_real_samples:
            loss_dict["spec"] = spec
            loss_dict["audio"] = audio
        loss_dict["audio_pred"] = audio_pred
        loss_dict["spec_pred"] = spec_pred
        loss_dict["loss"] = loss
        return loss_dict

    def validation_epoch_end(self, outputs):
        # Los images and audio manually
        if self.logger is not None and self.logger.experiment is not None:
            if not self.logged_real_samples:
                self.logger.experiment.add_image(
                    "val_mel_target",
                    plot_spectrogram_to_numpy(outputs[0]["spec"][0].data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_audio(
                    "val_wav_target",
                    outputs[0]["audio"][0].data.cpu().numpy(),
                    self.global_step,
                    sample_rate=self.sample_rate,
                )
                self.logged_real_samples = True
            self.logger.experiment.add_image(
                "val_mel_predicted",
                plot_spectrogram_to_numpy(outputs[0]["spec_pred"][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_audio(
                "val_wav_predicted",
                outputs[0]["audio_pred"][0].data.cpu().numpy(),
                self.global_step,
                sample_rate=self.sample_rate,
            )

        def get_stack(list_of_dict, key):
            """
            Helper function to take a list of losses and reduce across all validation batches
            """
            return_list = [[]] * len(list_of_dict[0][key])
            for dict_ in list_of_dict:
                list_of_losses = dict_[key]
                for i, loss in enumerate(list_of_losses):
                    return_list[i].append(loss)
            for i, loss in enumerate(return_list):
                return_list[i] = torch.mean(torch.stack(loss))
            return return_list

        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("val_loss", loss, sync_dist=True)

        if self.start_training_disc:
            gan_loss = get_stack(outputs, "gan_loss")
            self.log("val_loss_gan_loss", sum(gan_loss) / len(gan_loss), sync_dist=True)
            for i, _ in enumerate(gan_loss):
                self.log(
                    f"val_loss_gan_loss_{i}", gan_loss[i] / len(gan_loss), sync_dist=True,
                )

        sc_loss = get_stack(outputs, "sc_loss")
        mag_loss = get_stack(outputs, "mag_loss")
        self.log("val_loss_feat_loss", torch.stack([x['loss_feat'] for x in outputs]).mean(), sync_dist=True)
        self.log("val_loss_feat_loss_fb_sc", sum(sc_loss) / len(sc_loss), sync_dist=True)
        self.log("val_loss_feat_loss_fb_mag", sum(mag_loss) / len(sc_loss), sync_dist=True)
        for i, _ in enumerate(sc_loss):
            self.log(f"val_loss_feat_loss_fb_sc_{i}", sc_loss[i] / len(sc_loss), sync_dist=True)
            self.log(f"val_loss_feat_loss_fb_mag_{i}", mag_loss[i] / len(sc_loss), sync_dist=True)

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

    def training_epoch_end(self, outputs):
        # Do manual logginging of learning rate and epoch
        if self.current_epoch % 100 == 0:
            lrs = []
            for scheduler in self._trainer.lr_schedulers:
                param_groups = scheduler['scheduler'].optimizer.param_groups
                lrs.append(param_groups[0]['lr'])
            self.logger.experiment.add_scalar("lr-Adam", lrs[0], self.global_step)
            self.logger.experiment.add_scalar("lr-Adam-1", lrs[1], self.global_step)
            self.logger.experiment.add_scalar("epoch", self.current_epoch, self.global_step)

        # Start training discriminator after 20% of training
        if self.current_epoch >= np.ceil(0.2 * self._trainer.max_epochs):
            logging.info(f"MelGAN discriminator was enabled at epoch: {self.current_epoch}")
            self.start_training_disc = True

        return super().training_epoch_end(outputs)

    @classmethod
    def list_available_models(cls) -> 'Optional[Dict[str, str]]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_melgan",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_melgan/versions/1.0.0/files/tts_melgan.nemo",
            description="This model is trained on LJSpeech sampled at 22050Hz, and has been tested on generating female English voices with an American accent.",
            class_=cls,
        )
        list_of_models.append(model)
        return list_of_models
