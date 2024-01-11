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

import os

import torch
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


class MegatronStableDiffusionTrainerBuilder(MegatronTrainerBuilder):
    """Builder for SD model Trainer with overrides."""

    def _training_strategy(self) -> NLPDDPStrategy:
        """
        Returns a ddp strategy passed to Trainer.strategy.
        """
        ddp_overlap = self.cfg.model.get('ddp_overlap', True)
        if ddp_overlap:
            return NLPDDPStrategy(
                no_ddp_communication_hook=False,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=True,
                bucket_cap_mb=256,
            )
        else:
            return NLPDDPStrategy(
                no_ddp_communication_hook=True,
                gradient_as_bucket_view=self.cfg.model.gradient_as_bucket_view,
                find_unused_parameters=False,
            )


@hydra_runner(config_path='conf', config_name='sd_train')
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.model.capture_cudagraph_iters >= 0:
        # Required by CUDA graph with DDP
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # Hack to avoid CUDA graph issue with AMP, PyTorch Lightning doesn't support
        # changing autocast arguments for now.
        # https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/graphs.py#L234
        def amp_autocast_init(self, *args, **kwargs):
            if "cache_enabled" not in kwargs:
                kwargs["cache_enabled"] = False
            return self.__orig_init__(*args, **kwargs)

        torch.cuda.amp.autocast.__orig_init__ = torch.cuda.amp.autocast.__init__
        torch.cuda.amp.autocast.__init__ = amp_autocast_init
        torch.autocast.__orig_init__ = torch.autocast.__init__
        torch.autocast.__init__ = amp_autocast_init

    trainer = MegatronStableDiffusionTrainerBuilder(cfg).create_trainer()

    exp_manager(trainer, cfg.exp_manager)

    model = MegatronLatentDiffusion(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
