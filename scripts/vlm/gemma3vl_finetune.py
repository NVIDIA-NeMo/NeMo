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


import torch
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm, vlm

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.vlm.gemma3vl.data.mock import Gemma3VLMockDataModule
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


NAME = "gemma3vl_4b"

HF_MODEL_NAME = "google/gemma-3-4b-it"


def finetune_recipe(
    log_dir: str,
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
):
    """Gemma3 VL finetune"""

    max_sequence_length = 512

    tokenizer = AutoTokenizer(HF_MODEL_NAME)
    language_transformer_config = llm.Gemma3Config4B(seq_length=max_sequence_length)
    vision_transformer_config = vlm.Gemma3VLVisionConfig()
    vision_projection_config = vlm.Gemma3VLMultimodalProjectorConfig(
        input_size=vision_transformer_config.hidden_size,
        hidden_size=language_transformer_config.hidden_size,
    )
    gemma3vl_config = vlm.Gemma3VLConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        freeze_language_model=False,
        freeze_vision_model=True,
        freeze_vision_projection=True,
    )
    model = vlm.Gemma3VLModel(gemma3vl_config, tokenizer=tokenizer)
    llm.import_ckpt(model=model, source=f"hf://{HF_MODEL_NAME}")

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        encoder_tensor_model_parallel_size=0,
        encoder_pipeline_model_parallel_size=0,
        context_parallel_size=1,
        sequence_parallel=True,
        ckpt_async_save=True,
        ckpt_parallel_save=True,
        ckpt_parallel_load=True,
        ckpt_parallel_save_optim=True,
        ckpt_load_strictness="log_all",
        gradient_as_bucket_view=True,
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
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        max_steps=5,
        limit_val_batches=0,
        val_check_interval=10,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        strategy=strategy,
        accumulate_grad_batches=1,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        enable_checkpointing=False,
        callbacks=[TimingCallback()],
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

    data = Gemma3VLMockDataModule(
        seq_length=max_sequence_length,
        global_batch_size=4,
        micro_batch_size=1,
        tokenizer=tokenizer,
        num_workers=4,
    )

    ckpt = nl.ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        save_optim_on_train_end=False,
        filename="{val_loss:.2f}-{step}-{consumed_samples}",
    )
    tb = TensorBoardLogger(
        save_dir="tensorboard",  # The name of tfevents folder
        name="",  # No need further subfolder
    )
    logger = nl.NeMoLogger(
        explicit_log_dir=log_dir,
        log_global_rank_0_only=True,
        update_logger_directory=True,
        ckpt=ckpt,
        tensorboard=tb,
    )

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    finetune_recipe(f"/tmp/{NAME}", num_gpus_per_node=2)
