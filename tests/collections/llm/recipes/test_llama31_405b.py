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
import pytest
import torch

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.llama import Llama31Config405B, LlamaModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import llama31_405b
from nemo.lightning import Trainer
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback


class TestLlama31_405B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama31_405b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama31Config405B
        assert model_config.config.seq_length == 8192

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 64  # Much larger than 70B
        assert trainer_config.max_steps == 1168251

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 8
        assert trainer_config.strategy.pipeline_model_parallel_size == 8
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 2
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True
        assert trainer_config.strategy.account_for_embedding_in_pipeline_split is True
        assert trainer_config.strategy.account_for_loss_in_pipeline_split is True

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 8192
        assert recipe.data.global_batch_size == 512
        assert recipe.data.micro_batch_size == 1

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 6  # Different from other models
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.dim == 16
        assert recipe.peft.alpha == 32
        assert recipe.optim.config.lr == 1e-4

        # Check pipeline split settings
        assert recipe.trainer.strategy.account_for_embedding_in_pipeline_split is True
        assert recipe.trainer.strategy.account_for_loss_in_pipeline_split is True

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 8
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 14
        assert recipe.data.global_batch_size == 6
        assert recipe.optim.config.lr == 5e-6
        assert not hasattr(recipe, 'peft') or recipe.peft is None

    def test_pretrain_performance_optimizations(self, recipe_module):
        recipe = recipe_module.pretrain_performance_optimizations(recipe_module.pretrain_recipe())
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False

        # Check callbacks
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == GarbageCollectionCallback
            for cb in recipe.trainer.callbacks
        )
        comm_overlap_cb = next(
            cb
            for cb in recipe.trainer.callbacks
            if isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback"
        )
        assert comm_overlap_cb.tp_comm_overlap is True
        assert comm_overlap_cb.defer_embedding_wgrad_compute is True
        assert comm_overlap_cb.wgrad_deferral_limit == 50
        assert comm_overlap_cb.overlap_param_gather_with_optimizer_step is False

    def test_finetune_performance_optimizations(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True, peft_scheme=None)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 8
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 14
        assert recipe.trainer.strategy.sequence_parallel is True
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False

        # Check DDP settings
        assert recipe.trainer.strategy.ddp.grad_reduce_in_fp32 is False
        assert recipe.trainer.strategy.ddp.overlap_grad_reduce is True
        assert recipe.trainer.strategy.ddp.overlap_param_gather is True
        assert recipe.trainer.strategy.ddp.average_in_collective is True

    def test_finetune_performance_optimizations_with_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True, peft_scheme='lora')
        assert recipe.peft.target_modules == ['linear_qkv']
        assert recipe.trainer.strategy.tensor_model_parallel_size == 4
        assert recipe.trainer.strategy.pipeline_model_parallel_size == 4
        assert recipe.trainer.strategy.virtual_pipeline_model_parallel_size == 4
        assert recipe.trainer.strategy.sequence_parallel is True
