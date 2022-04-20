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

# MIT License

# Copyright (c) 2019 Jeongmin Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from numpy import ndarray
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from torch import Tensor, nn

from nemo.collections.tts.helpers.helpers import eval_tts_scores
from nemo.collections.tts.models.base import LinVocoder
from nemo.collections.tts.modules.degli import OperationMode
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import IntType, LengthsType, SpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import deprecated


@dataclass
class DegliConfig:
    degli: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    train_params: Optional[Dict[Any, Any]] = None
    sched: Optional[Dict[Any, Any]] = None


def reconstruct_wave(*args: ndarray, kwargs_istft, n_sample=-1) -> ndarray:
    """
    construct time-domain wave from complex spectrogram
    Args:
        *args: the complex spectrogram.
        kwargs_istft: arguments of Inverse STFT.
        n_sample: expected audio length.
    Returns:
        audio (numpy)
    """

    if len(args) == 1:
        spec = args[0].squeeze()
        mag = None
        phase = None
        assert np.iscomplexobj(spec)
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)

    wave = librosa.istft(stft_matrix=spec, **kwargs_istft, **kwarg_len)
    return wave


@deprecated(version="1.9", explanation="DegliModel will be removed.")
class DegliModel(LinVocoder):
    """Deep Griffin Lim model used to convert between spectrograms and audio"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(DegliConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.degli = instantiate(self._cfg.degli)
        self.mode = OperationMode.infer
        self.criterion = nn.L1Loss(reduction='none')  # maybe should be loss subclass?
        self.l_hop = self._cfg.degli.hop_length
        self.n_fft = self._cfg.degli.n_fft
        self.kwargs_stft = dict(hop_length=self.l_hop, window='hann', center=True, n_fft=self.n_fft, dtype=np.float32)
        self.kwargs_istft = dict(hop_length=self.l_hop, window='hann', center=True, dtype=np.float32)

        len_weight = self._cfg.train_params.repeat_training
        self.loss_weight = nn.Parameter(torch.tensor([1.0 / i for i in range(len_weight, 0, -1)]), requires_grad=False)
        self.loss_weight /= self.loss_weight.sum()

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
        self.degli.mode = new_mode

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "mag": NeuralType(('B', 'any', 'D', 'T'), SpectrogramType()),
            "max_length": NeuralType(None, LengthsType()),
            "repeats": NeuralType(None, IntType()),
        }

    @property
    def output_types(self):
        if self.mode == OperationMode.training or self.mode == OperationMode.validation:
            return {
                "out_repeats": NeuralType(('B', 'any', 'C', 'D', 'T'), SpectrogramType()),
                "final_out": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
                "residual": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            }
        else:
            return {
                "final_out": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            }

    @typecheck()
    def forward(self, *, x, mag, max_length, repeats):
        if self.mode != self.degli.mode:
            raise ValueError(f"Degli's mode {self.mode} does not match DegliModule's mode {self.degli.mode}")

        tensors = self.degli(x=x, mag=mag, max_length=max_length, repeat=repeats)
        return tensors  # audio_pred

    def convert_linear_spectrogram_to_audio(self, spec: torch.Tensor, Ts=None, repeats: int = 32) -> torch.Tensor:
        self.mode = OperationMode.infer

        batch_size = spec.shape[0]
        if len(spec.shape) == 3:
            spec = spec.unsqueeze(1)

        x = torch.normal(0, 1, [batch_size, 2, spec.shape[2], spec.shape[3]]).to(self.device)
        length = (spec.shape[3] - 1) * self.l_hop
        with torch.no_grad():
            y = self.degli(x=x, mag=spec, max_length=length, repeat=repeats)

        if Ts is None:
            Ts = [y.shape[3]] * batch_size

        max_size = (max(Ts) - 1) * self.l_hop

        audios = torch.zeros(batch_size, max_size)

        for i in range(batch_size):
            y_i = self.postprocess(y, Ts, i)
            my_len = (Ts[i] - 1) * self.l_hop
            audio = reconstruct_wave(y_i, kwargs_istft=self.kwargs_istft, n_sample=my_len)
            audios[i, 0:my_len] = torch.from_numpy(audio)

        return audios

    def calc_loss(self, out_blocks: Tensor, y: Tensor, T_ys: Sequence[int]) -> Tensor:
        """
        calculate L1 loss (criterion) between the real spectrogram and the outputs.

        Args:
            out_blocks: output for the Degli model, may include several concatrated outputs.
            y: desired output.
            T_ys: lengths (time domain) of non-zero parts of the histrograms, for each sample in the batch.
        Returns:
            a loss score.
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = self.criterion(out_blocks, y.unsqueeze(1))
        loss_blocks = torch.zeros(out_blocks.shape[1], device=y.device)
        for T, loss_batch in zip(T_ys, loss_no_red):
            loss_blocks += torch.mean(loss_batch[..., :T], dim=(1, 2, 3))

        if len(loss_blocks) == 1:
            loss = loss_blocks.squeeze()
        else:
            loss = loss_blocks @ self.loss_weight
        return loss

    def training_step(self, batch, batch_idx):
        self.mode = OperationMode.infer

        x, mag, max_length, y, T_ys, _, _ = batch
        output_loss, _, _ = self(x=x, mag=mag, max_length=max_length, repeats=self._cfg.train_params.repeat_training)
        loss = self.calc_loss(output_loss, y, T_ys)

        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    @torch.no_grad()
    def postprocess(self, output: Tensor, Ts: ndarray, idx: int):
        one = output[idx, :, :, : Ts[idx]]
        one = one.permute(1, 2, 0).contiguous()  # F, T, 2
        one = one.cpu().numpy().view(dtype=np.complex64)  # F, T, 1
        return one

    def validation_step(self, batch, batch_idx):
        """
        A validation step that also calculates the STOI/PESQ scores,
        and the scored for high repetition count (repeat_validation argument)
        """
        self.mode = OperationMode.infer
        val_repeats = self._cfg.train_params.repeat_validation

        x, mag, max_length, y, T_ys, length, path_speech = batch
        output_loss, output, _ = self(x=x, mag=mag, max_length=max_length, repeats=1)
        _, output_x, _ = self(x=x, mag=mag, max_length=max_length, repeats=val_repeats)

        loss = self.calc_loss(output_loss, y, T_ys)
        cnt = x.shape[0]

        stoi, pesq, stoi_x, pesq_x = (0.0, 0.0, 0.0, 0.0)
        for p in range(cnt):
            y_wav_path = path_speech[p]
            y_wav = sf.read(y_wav_path)[0].astype(np.float32)

            n_sample = length[p]
            out = self.postprocess(output, T_ys, p)
            out_wav = reconstruct_wave(out, kwargs_istft=self.kwargs_istft, n_sample=n_sample)
            measure = eval_tts_scores(y_wav, out_wav)
            stoi += torch.tensor(measure['STOI'])
            pesq += torch.tensor(measure['PESQ'])

            out = self.postprocess(output_x, T_ys, p)
            out_wav = reconstruct_wave(out, kwargs_istft=self.kwargs_istft, n_sample=n_sample)
            measure = eval_tts_scores(y_wav, out_wav)
            stoi_x += torch.tensor(measure['STOI'])
            pesq_x += torch.tensor(measure['PESQ'])

        return {
            "val_loss": loss,
            "stoi": stoi / cnt,
            "pesq": pesq / cnt,
            "stoi_x%d" % val_repeats: stoi_x / cnt,
            "pesq_x%d" % val_repeats: pesq_x / cnt,
        }

    def validation_epoch_end(self, outputs):
        tensorboard_logs = {}

        for k in outputs[0].keys():
            tensorboard_logs[k] = torch.stack([x[k] for x in outputs]).mean()

        return {'val_loss': tensorboard_logs['val_loss'], 'log': tensorboard_logs}

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")  # TODO
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")  # TODO
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

        return list_of_models
