# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.models.flux.model import FluxModelParams, MegatronFluxModel
from nemo.collections.llm.recipes.log.default import default_resume, tensorboard_logger

NAME = "flux-535m"


@dataclass
class DummyModelParams(FluxModelParams):
    """
    Initialize a toy model that only has one layer of each type.
    """

    def __post_init__(self):
        self.t5_params = None
        self.clip_params = None
        self.vae_config = None
        self.flux_config.num_single_layers = 1
        self.flux_config.num_joint_layers = 1


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Flux sample model configuration with only 1 transformer layers.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Flux sample (535 million) model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=bert_110m ...

        Python API usage:
            >>> model_config = model(flux_params)
            >>> print(model_config)
    """

    return run.Config(
        MegatronFluxModel,
        flux_params=run.Config(
            DummyModelParams,
        ),
    )


@run.cli.factory(target=llm.train, name=NAME)
def unit_test_recipe(
    name: str = "default",
    dir: Optional[str] = None,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
):
    """
    Flux ci test recipe with default trainer, no parallelism.
    """
    return run.Partial(
        llm.train,
        model=model(),
        trainer=run.Config(
            nl.Trainer,
            devices=num_gpus_per_node,
            num_nodes=num_nodes,
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                ),
            ),
            plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
            num_sanity_val_steps=0,
            max_steps=10,
            log_every_n_steps=1,
        ),
        log=run.Config(
            nl.NeMoLogger,
            ckpt=None,
            name=name,
            tensorboard=tensorboard_logger(name=name),
            log_dir=dir,
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
                use_distributed_optimizer=True,
                weight_decay=0,
            ),
        ),
        resume=default_resume(),
    )
