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
from nemo.collections.llm.gpt.model.baichuan import Baichuan2Config7B, Baichuan2Model
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import baichuan2_7b
from nemo.lightning import Trainer
from nemo.utils.exp_manager import TimingCallback


class TestBaichuan2_7B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return baichuan2_7b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == Baichuan2Model
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Baichuan2Config7B

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 1
        assert trainer_config.max_steps == 1168251

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_dtype is None
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 2
        assert trainer_config.strategy.sequence_parallel is False
        assert trainer_config.strategy.gradient_as_bucket_view is True
        assert trainer_config.strategy.ckpt_async_save is True
        assert trainer_config.strategy.ckpt_parallel_load is True

        # Check DDP configuration
        assert trainer_config.strategy.ddp.check_for_nan_in_grad is True
        assert trainer_config.strategy.ddp.grad_reduce_in_fp32 is True
        assert trainer_config.strategy.ddp.overlap_grad_reduce is True
        assert trainer_config.strategy.ddp.overlap_param_gather is True

        # Check other trainer configurations
        assert trainer_config.accumulate_grad_batches == 1
        assert trainer_config.limit_test_batches == 50
        assert trainer_config.limit_val_batches == 32
        assert trainer_config.log_every_n_steps == 10
        assert trainer_config.use_distributed_sampler is False
        assert trainer_config.val_check_interval == 2000

        # Check plugins
        assert isinstance(trainer_config.plugins, run.Config)
        assert trainer_config.plugins.__fn_or_cls__.__name__ == "MegatronMixedPrecision"

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == Baichuan2Model
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
        assert recipe.model.__fn_or_cls__ == Baichuan2Model
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SquadDataModule
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 128
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.optim.config.lr == 1e-4

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_pretrain_performance_optimizations(self, recipe_module):
        recipe = recipe_module.pretrain_recipe_performance()
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback"
            for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == TimingCallback for cb in recipe.trainer.callbacks
        )

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.trainer(
            tensor_parallelism=2,
            pipeline_parallelism=2,
            pipeline_parallelism_type=torch.bfloat16,
            virtual_pipeline_parallelism=4,
            context_parallelism=4,
            sequence_parallelism=True,
        )
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 4
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 2
        assert recipe.optim.config.lr == 5e-6
        assert not hasattr(recipe, 'peft') or recipe.peft is None

    def test_finetune_recipe_with_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid_scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid_scheme")
