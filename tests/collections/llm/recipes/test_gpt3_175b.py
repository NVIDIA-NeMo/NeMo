# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import nemo_run as run
import torch

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model import GPTConfig175B, GPTModel
from nemo.collections.llm.recipes import gpt3_175b
from nemo.lightning import Trainer
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback


class TestGPT3_175B:
    def test_model(self):
        model_config = gpt3_175b.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == GPTModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == GPTConfig175B

    def test_trainer_default_settings(self):
        trainer = gpt3_175b.trainer()
        assert isinstance(trainer, run.Config)
        assert trainer.__fn_or_cls__ == Trainer

        # Check default parallelism settings
        assert trainer.strategy.tensor_model_parallel_size == 4
        assert trainer.strategy.pipeline_model_parallel_size == 8
        assert trainer.strategy.pipeline_dtype == torch.bfloat16
        assert trainer.strategy.virtual_pipeline_model_parallel_size == 6
        assert trainer.strategy.context_parallel_size == 1
        assert trainer.strategy.sequence_parallel is True

        # Check default training settings
        assert trainer.max_steps == 1168251
        assert trainer.accumulate_grad_batches == 1
        assert trainer.limit_test_batches == 50
        assert trainer.limit_val_batches == 32
        assert trainer.log_every_n_steps == 10
        assert trainer.val_check_interval == 2000
        assert trainer.num_nodes == 64
        assert trainer.devices == 8

        # Check DDP settings
        assert trainer.strategy.ddp.check_for_nan_in_grad is True
        assert trainer.strategy.ddp.grad_reduce_in_fp32 is True
        assert trainer.strategy.ddp.overlap_grad_reduce is True
        assert trainer.strategy.ddp.overlap_param_gather is True
        assert trainer.strategy.ddp.average_in_collective is True

    def test_trainer_custom_settings(self):
        trainer = gpt3_175b.trainer(
            tensor_parallelism=8,
            pipeline_parallelism=4,
            virtual_pipeline_parallelism=4,
            context_parallelism=2,
            sequence_parallelism=False,
            num_nodes=32,
            num_gpus_per_node=4,
            max_steps=500000,
        )

        # Check custom parallelism settings
        assert trainer.strategy.tensor_model_parallel_size == 8
        assert trainer.strategy.pipeline_model_parallel_size == 4
        assert trainer.strategy.virtual_pipeline_model_parallel_size == 4
        assert trainer.strategy.context_parallel_size == 2
        assert trainer.strategy.sequence_parallel is False

        # Check custom training settings
        assert trainer.max_steps == 500000
        assert trainer.num_nodes == 32
        assert trainer.devices == 4

    def test_trainer_with_callbacks(self):
        callbacks = [run.Config(TimingCallback)]
        trainer = gpt3_175b.trainer(callbacks=callbacks)
        assert trainer.callbacks == callbacks

    def test_pretrain_recipe_default_settings(self):
        recipe = gpt3_175b.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain

        # Check model configuration
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == GPTModel
        assert isinstance(recipe.model.config, run.Config)
        assert recipe.model.config.__fn_or_cls__ == GPTConfig175B

        # Check data configuration
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 2048
        assert recipe.data.micro_batch_size == 2

    def test_pretrain_recipe_custom_settings(self):
        recipe = gpt3_175b.pretrain_recipe(
            name="custom_run",
            num_nodes=32,
            num_gpus_per_node=4,
        )
        assert recipe.trainer.num_nodes == 32
        assert recipe.trainer.devices == 4

    def test_pretrain_performance_optimizations(self):
        base_recipe = gpt3_175b.pretrain_recipe()
        recipe = gpt3_175b.pretrain_performance_optimizations(base_recipe)

        # Check that callbacks were added
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == GarbageCollectionCallback
            for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == MegatronCommOverlapCallback
            for cb in recipe.trainer.callbacks
        )

        # Check specific MegatronCommOverlapCallback settings
        comm_overlap_cb = next(
            cb
            for cb in recipe.trainer.callbacks
            if isinstance(cb, run.Config) and cb.__fn_or_cls__ == MegatronCommOverlapCallback
        )
        assert comm_overlap_cb.tp_comm_overlap is True
        assert comm_overlap_cb.defer_embedding_wgrad_compute is True
        assert comm_overlap_cb.wgrad_deferral_limit == 50
        assert comm_overlap_cb.overlap_param_gather_with_optimizer_step is False

        # Check that grad_reduce_in_fp32 was disabled
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False

    def test_pretrain_recipe_with_performance_mode(self):
        recipe = gpt3_175b.pretrain_recipe(performance_mode=True)

        # Verify performance optimizations were applied
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == GarbageCollectionCallback
            for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == MegatronCommOverlapCallback
            for cb in recipe.trainer.callbacks
        )
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False
