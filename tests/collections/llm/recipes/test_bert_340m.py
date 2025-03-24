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

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.bert.data.mock import BERTMockDataModule
from nemo.collections.llm.recipes import bert_340m
from nemo.lightning import Trainer
from nemo.utils.exp_manager import TimingCallback


class TestBERT_340M:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return bert_340m

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        # Note: Actual model class assertions would depend on bert_model implementation
        # which isn't shown in the provided code

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.pretrain_recipe().trainer
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
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is False

        # Check other trainer configurations
        assert trainer_config.accumulate_grad_batches == 1
        assert trainer_config.limit_test_batches == 32
        assert trainer_config.limit_val_batches == 32
        assert trainer_config.log_every_n_steps == 10
        assert trainer_config.val_check_interval == 2000

        # Check callbacks
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == TimingCallback for cb in trainer_config.callbacks
        )

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == BERTMockDataModule
        assert recipe.data.seq_length == 512
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 2

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_trainer_parallelism_options(self, recipe_module):
        trainer_config = recipe_module.pretrain_recipe(
            tensor_parallelism=2,
            pipeline_parallelism=2,
            pipeline_parallelism_type=torch.bfloat16,
            virtual_pipeline_parallelism=4,
            context_parallelism=4,
            sequence_parallelism=True,
        ).trainer

        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_dtype == torch.bfloat16
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size == 4
        assert trainer_config.strategy.context_parallel_size == 4
        assert trainer_config.strategy.sequence_parallel is True
