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

from collections import OrderedDict
from typing import List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.collections.tts.modules.talknet import GaussianEmbedding, MaskedInstanceNorm1d, StyleResidual
from nemo.core import Exportable
from nemo.core.classes import ModelPT, PretrainedModelInfo, typecheck
from nemo.core.neural_types import MelSpectrogramType, NeuralType
from nemo.core.neural_types.elements import LengthsType, MelSpectrogramType, RegressionValuesType, TokenIndex
from nemo.utils.decorators import deprecated


@deprecated(version="1.9", explanation="TalkNetDursModel will be removed. Please, use MixerTTSModel instead.")
class TalkNetDursModel(ModelPT):
    """TalkNet's durations prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = nn.Embedding(len(self.vocab.labels), cfg.d_char)
        self.encoder = instantiate(cfg.encoder)
        d_out = cfg.encoder.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, 1, kernel_size=1)

    @typecheck(
        input_types={"text": NeuralType(('B', 'T'), TokenIndex()), "text_len": NeuralType(('B'), LengthsType()),}
    )
    def forward(self, text, text_len):
        x, x_len = self.embed(text).transpose(1, 2), text_len
        y, _ = self.encoder(audio_signal=x, length=x_len)
        durs = self.proj(y).squeeze(1)
        return durs

    @staticmethod
    def _metrics(true_durs, true_text_len, pred_durs):
        loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        mask = get_mask_from_lengths(true_text_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * mask).sum().float() / mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * mask).sum().float() / mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * mask).sum().float() / mask.sum() * 100

        return loss, acc, acc_dist_1, acc_dist_3

    def training_step(self, batch, batch_idx):
        _, _, text, text_len, durs, *_ = batch
        pred_durs = self(text=text, text_len=text_len)
        loss, acc, acc_dist_1, acc_dist_3 = self._metrics(true_durs=durs, true_text_len=text_len, pred_durs=pred_durs,)
        train_log = {
            'train_loss': loss,
            'train_acc': acc,
            'train_acc_dist_1': acc_dist_1,
            'train_acc_dist_3': acc_dist_3,
        }
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs, *_ = batch
        pred_durs = self(text=text, text_len=text_len)
        loss, acc, acc_dist_1, acc_dist_3 = self._metrics(true_durs=durs, true_text_len=text_len, pred_durs=pred_durs,)
        val_log = {'val_loss': loss, 'val_acc': acc, 'val_acc_dist_1': acc_dist_1, 'val_acc_dist_3': acc_dist_3}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    @staticmethod
    def _loader(cfg):
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

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_durs.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate durations "
                "values for English voice with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models


@deprecated(version="1.9", explanation="TalkNetPitchModel will be removed. Please, use MixerTTSModel instead.")
class TalkNetPitchModel(ModelPT):
    """TalkNet's pitch prediction pipeline."""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.encoder = instantiate(cfg.encoder)
        d_out = cfg.encoder.jasper[-1].filters
        self.sil_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.body_proj = nn.Conv1d(d_out, 1, kernel_size=1)
        self.f0_mean, self.f0_std = cfg.f0_mean, cfg.f0_std

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_len": NeuralType(('B'), LengthsType()),
            "durs": NeuralType(('B', 'T'), LengthsType()),
        }
    )
    def forward(self, text, text_len, durs):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        y, _ = self.encoder(audio_signal=x, length=x_len)
        f0_sil = self.sil_proj(y).squeeze(1)
        f0_body = self.body_proj(y).squeeze(1)
        return f0_sil, f0_body

    def _metrics(self, true_f0, true_f0_mask, pred_f0_sil, pred_f0_body):
        sil_mask = true_f0 < 1e-5
        sil_gt = sil_mask.long()
        sil_loss = F.binary_cross_entropy_with_logits(input=pred_f0_sil, target=sil_gt.float(), reduction='none',)
        sil_loss *= true_f0_mask.type_as(sil_loss)
        sil_loss = sil_loss.sum() / true_f0_mask.sum()
        sil_acc = ((torch.sigmoid(pred_f0_sil) > 0.5).long() == sil_gt).float()  # noqa
        sil_acc *= true_f0_mask.type_as(sil_acc)
        sil_acc = sil_acc.sum() / true_f0_mask.sum()

        body_mse = F.mse_loss(pred_f0_body, (true_f0 - self.f0_mean) / self.f0_std, reduction='none')
        body_mask = ~sil_mask
        body_mse *= body_mask.type_as(body_mse)  # noqa
        body_mse = body_mse.sum() / body_mask.sum()  # noqa
        body_mae = ((pred_f0_body * self.f0_std + self.f0_mean) - true_f0).abs()
        body_mae *= body_mask.type_as(body_mae)  # noqa
        body_mae = body_mae.sum() / body_mask.sum()  # noqa

        loss = sil_loss + body_mse

        return loss, sil_acc, body_mae

    def training_step(self, batch, batch_idx):
        _, audio_len, text, text_len, durs, f0, f0_mask = batch
        pred_f0_sil, pred_f0_body = self(text=text, text_len=text_len, durs=durs)
        loss, sil_acc, body_mae = self._metrics(
            true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
        )
        train_log = {'train_loss': loss, 'train_sil_acc': sil_acc, 'train_body_mae': body_mae}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        _, _, text, text_len, durs, f0, f0_mask = batch
        pred_f0_sil, pred_f0_body = self(text=text, text_len=text_len, durs=durs)
        loss, sil_acc, body_mae = self._metrics(
            true_f0=f0, true_f0_mask=f0_mask, pred_f0_sil=pred_f0_sil, pred_f0_body=pred_f0_body,
        )

        val_log = {'val_loss': loss, 'val_sil_acc': sil_acc, 'val_body_mae': body_mae}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    @staticmethod
    def _loader(cfg):
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

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_pitch.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate pitch "
                "values for English voice with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models


@deprecated(version="1.9", explanation="TalkNetSpectModel will be removed. Please, use MixerTTSModel instead.")
class TalkNetSpectModel(SpectrogramGenerator, Exportable):
    """TalkNet's mel spectrogram prediction pipeline."""

    @property
    def output_types(self):
        return OrderedDict({"mel-spectrogram": NeuralType(('B', 'D', 'T'), MelSpectrogramType())})

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg=cfg, trainer=trainer)

        cfg = self._cfg
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**cfg.train_ds.dataset.vocab)
        self.blanking = cfg.train_ds.dataset.blanking
        self.preprocessor = instantiate(cfg.preprocessor)
        self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
        self.norm_f0 = MaskedInstanceNorm1d(1)
        self.res_f0 = StyleResidual(cfg.d_char, 1, kernel_size=3)
        self.encoder = instantiate(cfg.encoder)
        d_out = cfg.encoder.jasper[-1].filters
        self.proj = nn.Conv1d(d_out, cfg.n_mels, kernel_size=1)

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_len": NeuralType(('B'), LengthsType()),
            "durs": NeuralType(('B', 'T'), LengthsType()),
            "f0": NeuralType(('B', 'T'), RegressionValuesType()),
        }
    )
    def forward(self, text, text_len, durs, f0):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        y, _ = self.encoder(audio_signal=x, length=x_len)
        mel = self.proj(y)
        return mel

    @staticmethod
    def _metrics(true_mel, true_mel_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(true_mel_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(input_signal=audio, length=audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        train_log = {'train_loss': loss}
        return {'loss': loss, 'progress_bar': train_log, 'log': train_log}

    def validation_step(self, batch, batch_idx):
        audio, audio_len, text, text_len, durs, f0, f0_mask = batch
        mel, mel_len = self.preprocessor(input_signal=audio, length=audio_len)
        pred_mel = self(text=text, text_len=text_len, durs=durs, f0=f0)
        loss = self._metrics(true_mel=mel, true_mel_len=mel_len, pred_mel=pred_mel)
        val_log = {'val_loss': loss}
        self.log_dict(val_log, prog_bar=False, on_epoch=True, logger=True, sync_dist=True)

    @staticmethod
    def _loader(cfg):
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

    def parse(self, text: str, **kwargs) -> torch.Tensor:
        return torch.tensor(self.vocab.encode(text)).long().unsqueeze(0).to(self.device)

    def generate_spectrogram(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hasattr(self, '_durs_model') and hasattr(self, '_pitch_model')

        if self.blanking:
            tokens = [
                AudioToCharWithDursF0Dataset.interleave(
                    x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device).fill_(self.vocab.blank), y=t,
                )
                for t in tokens
            ]
            tokens = AudioToCharWithDursF0Dataset.merge(tokens, value=self.vocab.pad, dtype=torch.long)

        text_len = torch.tensor(tokens.shape[-1], dtype=torch.long).unsqueeze(0)
        durs = self._durs_model(text=tokens, text_len=text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(text=tokens, text_len=text_len, durs=durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        mel = self(text=tokens, text_len=text_len, durs=durs, f0=f0)

        return mel

    def forward_for_export(self, tokens: torch.Tensor, text_len: torch.Tensor):
        durs = self._durs_model(text=tokens, text_len=text_len)
        durs = durs.exp() - 1
        durs[durs < 0.0] = 0.0
        durs = durs.round().long()

        # Pitch
        f0_sil, f0_body = self._pitch_model(text=tokens, text_len=text_len, durs=durs)
        sil_mask = f0_sil.sigmoid() > 0.5
        f0 = f0_body * self._pitch_model.f0_std + self._pitch_model.f0_mean
        f0 = (~sil_mask * f0).float()

        # Spect
        x, x_len = self.embed(tokens, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        y, _ = self.encoder(audio_signal=x, length=x_len)
        mel = self.proj(y)

        return mel

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        model = PretrainedModelInfo(
            pretrained_model_name="tts_en_talknet",
            location=(
                "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_talknet/versions/1.0.0rc1/files"
                "/talknet_spect.nemo"
            ),
            description=(
                "This model is trained on LJSpeech sampled at 22050Hz, and can be used to generate female "
                "English voices with an American accent."
            ),
            class_=cls,  # noqa
            aliases=["TalkNet-22050Hz"],
        )
        list_of_models.append(model)
        return list_of_models

    @classmethod
    def from_pretrained(cls, model_name: str, *args, **kwargs):
        """Custom TalkNet's three-part load logic."""
        model = super().from_pretrained(model_name, *args, **kwargs)
        model.add_module('_pitch_model', TalkNetPitchModel.from_pretrained(model_name, *args, **kwargs))
        model.add_module('_durs_model', TalkNetDursModel.from_pretrained(model_name, *args, **kwargs))
        return model
