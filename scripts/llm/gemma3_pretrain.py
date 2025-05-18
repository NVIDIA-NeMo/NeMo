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

"""Gemma3 language model pretrain"""

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.gemma3 import Gemma3Config1B, Gemma3Model
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main():
    """Entrypoint"""
    data = llm.MockDataModule(
        num_train_samples=1_000_000,
        seq_length=1024,
        global_batch_size=32,
        micro_batch_size=1,
    )

    model_config = Gemma3Config1B()
    model = Gemma3Model(model_config)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_save=True,
        ckpt_parallel_load=True,
        ckpt_parallel_save_optim=True,
        ckpt_load_strictness="log_all",
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        max_steps=10,
        limit_val_batches=0,
        val_check_interval=5,
        log_every_n_steps=1,
        strategy=strategy,
        accumulate_grad_batches=1,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        enable_checkpointing=False,
        callbacks=[
            TimingCallback(),
        ],
    )

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=3e-4,
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )
    lr_scheduler = CosineAnnealingScheduler(
        warmup_steps=2000,
        constant_steps=0,
        min_lr=3e-5,
    )
    opt = MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

    ckpt = nl.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        save_optim_on_train_end=False,
        filename="{val_loss:.2f}-{step}-{consumed_samples}",
    )
    tb = TensorBoardLogger(
        save_dir="tensorboard",
        name="",
    )
    logger = nl.NeMoLogger(
        explicit_log_dir="/tmp/gemma3",
        log_global_rank_0_only=True,
        update_logger_directory=True,
        ckpt=ckpt,
        tensorboard=tb,
    )

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        resume=resume,
        optim=opt,
    )


if __name__ == "__main__":
    main()
