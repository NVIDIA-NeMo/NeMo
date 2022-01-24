###########################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ##########################################################################
# import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from nemo.collections.tts.losses.radttsloss import RadTTSLoss
from nemo.collections.tts.losses.radttsloss import AttentionBinarizationLoss
from nemo.collections.tts.modules.radtts import RadTTSModule
from nemo.collections.tts.modules.alignment import plot_alignment_to_numpy
from nemo.collections.tts.helpers.helpers import plot_spectrogram_to_numpy

from nemo.collections.tts.helpers.radam import RAdam
from timeit import default_timer as timer
import hashlib
from nemo.collections.tts.torch.tts_tokenizers import BaseTokenizer, EnglishCharsTokenizer, EnglishPhonemesTokenizer

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pytorch_lightning as pl
from nemo.collections.tts.models.base import SpectrogramGenerator
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.core.classes import Exportable
torch.cuda.empty_cache()

class RadTTSModel(SpectrogramGenerator, Exportable):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        
        self._setup_tokenizer(cfg.validation_ds.dataset)
        
        assert self.tokenizer is not None

        num_tokens = len(self.tokenizer.tokens)
        self.tokenizer_pad = self.tokenizer.pad
        self.tokenizer_unk = self.tokenizer.oov
        
        self.text_tokenizer_pad_id = None
        self.tokens = None
        
        super().__init__(cfg=cfg, trainer=trainer)
        self.feat_loss_weight = 1.0
        self.model_config = cfg.modelConfig
        self.train_config = cfg.trainerConfig
        self.optim = cfg.optim
        
        self.criterion = RadTTSLoss(self.train_config.sigma, bool(self.model_config.n_components),
                                      self.train_config.dur_loss_weight, self.feat_loss_weight,
                                      self.model_config.n_group_size,
                                      self.model_config.dur_model_config,
                                      self.model_config.feature_model_config,
                                      self.train_config.mask_unvoiced_f0)
        
        self.attention_kl_loss = AttentionBinarizationLoss()
        self.model = instantiate(cfg.modelConfig)
        self._parser = None
        self._tb_logger = None
        self.cfg = cfg
        self.log_train_images = False
    def batch_dict(self, batch_data):
        batch_data_dict = {
            "audio": batch_data[0],
            "audio_lens": batch_data[1],
            "text": batch_data[2],
            "text_lens": batch_data[3],
            "log_mel": batch_data[4],
            "log_mel_lens": batch_data[5],
            "duration_prior": batch_data[6],
            "pitch": batch_data[7],
            "pitch_lens":batch_data[8],
            "voiced_mask": batch_data[9],
            "p_voiced": batch_data[10],
            "energy": batch_data[11],
            "energy_lens":batch_data[12],
            "speaker_id": batch_data[13],}
        return batch_data_dict
    
    def forward(self, batch):
        batch = self.batch_dict(batch)
        mel = batch['log_mel']
        speaker_ids = ['speaker_id']
        text = batch['text']
        in_lens = batch['text_lens']
        out_lens = batch['log_mel_lens']
        attn_prior = batch['duration_prior']
        aug_idxs = None
        f0 = batch['pitch']
        voiced_mask = batch['voiced_mask']
        p_voiced = batch['p_voiced']
        energy_avg = batch['energy']
        if self.train_config.binarization_start_iter >= 0 and self.global_step >= self.train_config.binarization_start_iter:
            # binarization training phase
            binarize = True
        else:
            # no binarization, soft-only
            binarize = False
        
        return self.model(mel=mel, speaker_ids=speaker_ids , text=text, in_lens =in_lens, out_lens=out_lens,
            binarize_attention=binarize, attn_prior=attn_prior,
            aug_idxs=aug_idxs, f0=f0, energy_avg=energy_avg,
            voiced_mask=voiced_mask, p_voiced=p_voiced)
    
    def training_step(self, batch, batch_idx):
        batch = self.batch_dict(batch)
        mel = batch['log_mel']
        speaker_ids = batch['speaker_id']
        text = batch['text']#batch['lm_token']
        in_lens = batch['text_lens']
        out_lens = batch['log_mel_lens']
        attn_prior = batch['duration_prior']
        aug_idxs = None
        f0 = batch['pitch']
        voiced_mask = batch['voiced_mask']
        p_voiced = batch['p_voiced']
        energy_avg = batch['energy']
                       
        if self.train_config.binarization_start_iter >= 0 and self.global_step >= self.train_config.binarization_start_iter:
            # binarization training phase
            binarize = True
        else:
            # no binarization, soft-only
            binarize = False
    
        outputs = self.model(
            mel, speaker_ids, text, in_lens, out_lens,
            binarize_attention=binarize, attn_prior=attn_prior,
            aug_idxs=aug_idxs, f0=f0, energy_avg=energy_avg,
            voiced_mask=voiced_mask, p_voiced=p_voiced)
        loss_outputs = self.criterion(outputs, in_lens, out_lens)
        
        
        ctc_loss = loss_outputs['loss_ctc']
        dur_loss = loss_outputs['loss_duration']
        fev_loss = loss_outputs['loss_fev']
        mel_loss = loss_outputs['loss_mel']
        
        loss_prior_mel = loss_outputs['loss_prior_mel']
        loss_prior_duration = loss_outputs['loss_prior_duration']
        loss_prior_fev = loss_outputs['loss_prior_fev']
        loss_fev_extra = loss_outputs['loss_fev_extra']
        
        loss = (dur_loss + fev_loss + mel_loss + self.train_config.ctc_loss_weight *
                ctc_loss)
    
        if binarize and self.train_config.kl_loss_start_iter >= 0 and self.global_step >= self.train_config.kl_loss_start_iter:
            binarization_loss = self.attention_kl_loss(
                outputs['attn'], outputs['attn_soft'])
            loss += binarization_loss
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs['binarization_loss'] = binarization_loss
        self.log("total_loss/train_loss", loss)
        self.log("train/binarization_loss", binarization_loss)
        self.log("train/loss_ctc", ctc_loss)
        self.log("train/fev_loss", fev_loss)
        self.log("train/dur_loss", dur_loss)
        self.log("train/mel_loss", mel_loss)
        
        self.log("train/loss_prior_mel", loss_prior_mel)
        self.log("train/loss_prior_duration", loss_prior_duration)
        self.log("train/loss_prior_fev", loss_prior_fev)
        self.log("train/loss_fev_extra", loss_fev_extra)

        
        if self.log_train_images:
            self.log_train_images = False

            self.tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",)
       
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        batch = self.batch_dict(batch)
        speaker_ids = batch['speaker_id']
        text = batch['text']#batch['text']
        in_lens = batch['text_lens']
        out_lens = batch['log_mel_lens']
        attn_prior = batch['duration_prior']
        aug_idxs = None
        f0 = batch['pitch']
        voiced_mask = batch['voiced_mask']
        p_voiced = batch['p_voiced']
        energy_avg = batch['energy']
        mel = batch['log_mel']
        if self.train_config.binarization_start_iter >= 0 and self.global_step >= self.train_config.binarization_start_iter:
            # binarization training phase
            binarize = True
        else:
            # no binarization, soft-only
            binarize = False
        #print("val textttttttttttttttttttttttttttttttttttttt",text)
        outputs = self.model(
            mel, speaker_ids, text, in_lens, out_lens,
            binarize_attention=binarize, attn_prior=attn_prior,
            aug_idxs=aug_idxs, f0=f0, energy_avg=energy_avg,
            voiced_mask=voiced_mask, p_voiced=p_voiced)
        loss_outputs = self.criterion(outputs, in_lens, out_lens)
        
        ctc_loss = loss_outputs['loss_ctc']
        dur_loss = loss_outputs['loss_duration']
        fev_loss = loss_outputs['loss_fev']
        mel_loss = loss_outputs['loss_mel']
        
        loss_prior_mel = loss_outputs['loss_prior_mel']
        loss_prior_duration = loss_outputs['loss_prior_duration']
        loss_prior_fev = loss_outputs['loss_prior_fev']
        loss_fev_extra = loss_outputs['loss_fev_extra']
        
        loss = (dur_loss + fev_loss + mel_loss + self.train_config.ctc_loss_weight *
                ctc_loss)
    
        if binarize and self.train_config.kl_loss_start_iter >= 0 and self.global_step >= self.train_config.kl_loss_start_iter:
            binarization_loss = self.attention_kl_loss(
                outputs['attn'], outputs['attn_soft'])
            loss += binarization_loss
        else:
            binarization_loss = torch.zeros_like(loss)
        loss_outputs['binarization_loss'] = binarization_loss
    
        return {"val_loss": loss,
                "loss_ctc": ctc_loss,
                "fev_loss": fev_loss,
                "dur_loss": dur_loss,
                "mel_loss": mel_loss,
                'loss_prior_mel': loss_prior_mel,
                'loss_prior_duration': loss_prior_duration,
                'loss_prior_fev': loss_prior_fev,
                'loss_fev_extra': loss_fev_extra,
                "mel_target": mel if batch_idx == 0 else None,
                "mel_pred": outputs["z_mel"] if batch_idx == 0 else None,
                "attn": outputs["attn"] if batch_idx == 0 else None,
                "attn_soft": outputs["attn_soft"] if batch_idx == 0 else None,
                "audiopaths": "audio_1" if batch_idx == 0 else None,}
    
    def validation_epoch_end(self, outputs):
        collect = lambda key: torch.stack([x[key] for x in outputs]).mean()
        val_loss = collect("val_loss")
        mel_loss = collect("mel_loss")
        dur_loss = collect("dur_loss")
        fev_loss = collect("fev_loss")
        ctc_loss = collect("loss_ctc")
        loss_prior_mel = collect('loss_prior_mel')
        loss_prior_duration = collect('loss_prior_duration')
        loss_prior_fev = collect('loss_prior_fev')
        loss_fev_extra = collect('loss_fev_extra')
        self.log("total_loss/val_loss", val_loss)
        self.log("val/v_ctc_loss", ctc_loss)
        self.log("val/v_mel_loss", mel_loss)
        self.log("val/v_dur_loss", dur_loss)
        self.log("val/v_fev_loss", fev_loss)
        self.log("val/v_loss_prior_mel", loss_prior_mel)
        self.log("val/v_loss_prior_duration", loss_prior_duration)
        self.log("val/v_loss_prior_fev", loss_prior_fev)
        self.log("val/v_loss_fev_extra", loss_fev_extra)

        val_loss, ctc_loss, fev_loss, dur_loss, mel_loss, loss_prior_mel, loss_prior_duration, loss_prior_fev, loss_fev_extra, mel_target, mel_pred, attn, attn_soft, audiopaths = outputs[0].values()
        self.tb_logger.add_image(
            "val_mel_target", plot_spectrogram_to_numpy(mel_target[0].data.cpu().numpy()), self.global_step, dataformats="HWC", )
        mel_pred = mel_pred[0].data.cpu().numpy()
        self.tb_logger.add_image(
            "val_mel_predicted", plot_spectrogram_to_numpy(mel_pred), self.global_step, dataformats="HWC",
        )
        self.log_train_images = True
        
        #audioname = os.path.basename(audiopaths[0])
        audioname = "audio_2"
        self.tb_logger.add_image(
            'attention_weights_mas',plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy().T, title=audioname), self.global_step, dataformats='HWC')
        
        self.log_train_images = True
        
        self.tb_logger.add_image(
            'attention_weights',plot_alignment_to_numpy(attn_soft[0, 0].data.cpu().numpy().T, title=audioname), self.global_step, dataformats='HWC')
        
        self.log_train_images = True
        
    def configure_optimizers(self):
        print("Initializing %s optimizer" % (self.optim.name))
        if len(self.train_config.finetune_layers):
            for name, param in model.named_parameters():
                if any([l in name for l in self.train_config.finetune_layers]):  # short list hack
                    print("Fine-tuning parameter", name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if self.optim.name == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optim.lr,
                                         weight_decay=self.optim.weight_decay)
        elif self.optim.name == 'RAdam':
            optimizer = RAdam(self.model.parameters(), lr=self.optim.lr,
                              weight_decay=self.optim.weight_decay)
        else:
            print("Unrecognized optimizer %s!" % (self.optim.name))
            exit(1)
    
        if self.optim.sched.name == 'cosine':
            # keeping init restart at epoch = 2
            # operate at iteration level instead of epoch level.
            restart_iteration = 1500
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=restart_iteration, T_mult=2, eta_min=1e-5)
    
        elif self.optim.sched.name == 'exp_decay':
            # Decays at epoch level, 10k steps / epoch
            # gamma = Multiplicative factor of learning rate decay.
            # Set gamma st starting at 0.0035 lr, ending at 1e-5 in 50k steps
            # Setting gamma using equation: 0.0035 * gamma ^ 50000 = 1e-5
            lr_scheduler = ExponentialLR(optimizer, gamma=0.99)
        elif self.optim.sched.name == 'red_on_plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer)
        elif self.optim.sched.name == 'step_decay':
            # have to explicitly set the init lr.
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.optim.sched.name
                param_group['initial_lr'] = self.optim.sched.name
            if self.train_config.scheduler_start_iteration > 1:
                max_decay_steps = self.train_config.scheduler_start_iteration
            else:
                max_decay_steps = 100000
            # decay every 500 steps, calculated gamma using eq: 0.0035 * gamma ^ 50k = 1e-5
            lr_scheduler = StepLR(
                optimizer, step_size=500, gamma=0.99, last_epoch=max_decay_steps)
        else:
            lr_scheduler = None
            # force set the learning rate to what is specified
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.optim.lr
    
        return optimizer

    @staticmethod
    def _loader(cfg):
        try:
            _ = cfg.dataset.manifest_filepath
        except omegaconf.errors.MissingMandatoryValue:
            logging.warning("manifest_filepath was skipped. No dataset for this model.")
            return None

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(  # noqa
            dataset=dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params,
        )

    def setup_training_data(self, cfg):
        self._train_dl = self._loader(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self._loader(cfg)

    def setup_test_data(self, cfg):
        """Omitted."""
        pass
    def generate_spectrogram(self, tokens: 'torch.tensor', speaker: int = 0, pace: float = 1.0) -> torch.tensor:
        self.eval()
        s = [0]
        speaker = torch.tensor(s).long().cuda()#.to(self.device)
        outputs = self.model.infer_complete(speaker, tokens, sigma = 0.8, sigma_txt = 0.666, sigma_feats = 0.666,
                    token_dur_scaling = 1.00, token_duration_max=100, f0_mean=0.00,
                    f0_std=0.00, energy_mean=0.00,
                    energy_std=0.00)

        spect = outputs['mel']
        return spect
    
    @property
    def parser(self):
        if self._parser is not None:
            return self._parser
        return self._parser
    
    def _setup_tokenizer(self, cfg):
        text_tokenizer_kwargs = {}
        if "g2p" in cfg.text_tokenizer:
            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)
        if isinstance(self.tokenizer, BaseTokenizer):
            self.text_tokenizer_pad_id = self.tokenizer.pad
            self.tokens = self.tokenizer.tokens
        else:
            if text_tokenizer_pad_id is None:
                raise ValueError(f"text_tokenizer_pad_id must be specified if text_tokenizer is not BaseTokenizer")

            if tokens is None:
                raise ValueError(f"tokens must be specified if text_tokenizer is not BaseTokenizer")

            self.text_tokenizer_pad_id = text_tokenizer_pad_id
            self.tokens = tokens
        
    
    def parse(self, text: str, normalize=False) -> torch.Tensor:
        if normalize and self.text_normalizer_call is not None:
            text = self.text_normalizer_call(text, **self.text_normalizer_call_kwargs)
        return torch.tensor(self.tokenizer(text)).long().unsqueeze(0).cuda()#.to(self.device)
    
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



