# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import re
from abc import ABC, abstractclassmethod
from typing import Any, Optional

import torch

from nemo.core.classes import ModelPT
from nemo.utils import logging


class DiffusionModel(ModelPT, ABC):
    @abstractclassmethod
    def get_conditioning(self, c: Any) -> Any:
        """
        Encode conditioning c.
        For txt2img use-case, the input conditioning would be the plain text,
        and output would be the encoded embedding for the corresponding text;
        For img2img use-case, the input conditioning would be the raw image,
        and output would be the corresponding image embedding

        Args:
            c: conditioning
        
        Returns:
            encoded conditioning
        """
        pass

    @abstractclassmethod
    def apply_model(self, x_t: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply Diffusion model.
        If c is not given, the model acts as an unconditional diffusion model.
        For diffusion model that applies on the pixel space, x_t should be in the pixel space;
        for diffusion model that applies on the latent space, x_t is in latent space.

        Args:
            x_t: noisy input x at timestamp t
            t: timestamp
            c: conditioning
        
        Returns:
            Predicted result that has the same shape as x_t
        """

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        try:
            init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
        except (ValueError, TypeError, IndexError):
            logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
            init_consumed_samples = 0

        return init_consumed_samples

    def compute_consumed_samples(self, steps_since_resume=0):
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume
            * self.trainer.world_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)
