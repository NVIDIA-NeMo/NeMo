# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import List, Union

import torch

from nemo.core.classes import ModelPT
from nemo.utils import logging, model_utils
from nemo.utils.decorators import experimental

__all__ = ['AudioToAudioModel']


@experimental
class AudioToAudioModel(ModelPT, ABC):
    @abstractmethod
    def process(
        self, paths2audio_files: List[str], output_dir: str, batch_size: int = 4
    ) -> List[Union[str, List[str]]]:
        """
        Takes paths to audio files and returns a list of paths to processed
        audios.

        Args:
            paths2audio_files: paths to audio files to be processed
            output_dir: directory to save processed files
            batch_size: batch size for inference

        Returns:
            Paths to processed audio signals.
        """
        pass

    @abstractmethod
    def evaluation_step(self, batch, batch_idx, dataloader_idx: int = 0, tag: str = 'val'):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluation_step(batch, batch_idx, dataloader_idx, 'test')

    def multi_evaluation_epoch_end(self, outputs, dataloader_idx: int = 0, tag: str = 'val'):
        loss_mean = torch.stack([x[f'{tag}_loss'] for x in outputs]).mean()
        tensorboard_logs = {f'{tag}_loss': loss_mean}
        return {f'{tag}_loss': loss_mean, 'log': tensorboard_logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_evaluation_epoch_end(outputs, dataloader_idx, 'test')

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        # recursively walk the subclasses to generate pretrained model info
        list_of_models = model_utils.resolve_subclass_pretrained_model_info(cls)
        return list_of_models

    def setup_optimization_flags(self):
        """
        Utility method that must be explicitly called by the subclass in order to support optional optimization flags.
        This method is the only valid place to access self.cfg prior to DDP training occurs.

        The subclass may chose not to support this method, therefore all variables here must be checked via hasattr()
        """
        # Skip update if nan/inf grads appear on any rank.
        self._skip_nan_grad = False
        if "skip_nan_grad" in self._cfg and self._cfg["skip_nan_grad"]:
            self._skip_nan_grad = self._cfg["skip_nan_grad"]

    def on_after_backward(self):
        """
        zero-out the gradients which any of them is NAN or INF
        """
        super().on_after_backward()

        if hasattr(self, '_skip_nan_grad') and self._skip_nan_grad:
            device = next(self.parameters()).device
            valid_gradients = torch.tensor([1], device=device, dtype=torch.float32)

            # valid_gradients = True
            for param_name, param in self.named_parameters():
                if param.grad is not None:
                    is_not_nan_or_inf = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    if not is_not_nan_or_inf:
                        valid_gradients = valid_gradients * 0
                        break

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(valid_gradients, op=torch.distributed.ReduceOp.MIN)

            if valid_gradients < 1:
                logging.warning(f'detected inf or nan values in gradients! Setting gradients to zero.')
                self.zero_grad()
