# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes import hyena_base
from nemo.lightning import Trainer


class TestHyenaBase:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return hyena_base

    def test_model(self, recipe_module):
        model_config = recipe_module.model(tp_comm_overlap=False, seq_length=2048)
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__.__name__ == "HyenaModel"
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__.__name__ == "HyenaTestConfig"

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer_recipe(
            tensor_parallelism=4,
            pipeline_parallelism=2,
            num_nodes=1,
            num_gpus_per_node=4,
            max_steps=200,
        )
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.devices == 4
        assert trainer_config.num_nodes == 1
        assert trainer_config.max_steps == 200

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 4
        assert trainer_config.strategy.pipeline_model_parallel_size == 2

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(
            dataset_config=None,
            global_batch_size=16,
            micro_batch_size=2,
            num_nodes=1,
            num_gpus_per_node=4,
            seq_length=2048,
            model_size="test",
        )
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__.__name__ == "HyenaModel"
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 16

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe(
            resume_path="dummy_path",
            dataset_config=None,
            global_batch_size=16,
            micro_batch_size=2,
            num_nodes=1,
            num_gpus_per_node=4,
            seq_length=2048,
            model_size="test",
        )
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__.__name__ == "HyenaModel"
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__.__name__ == "MockDataModule"
        assert recipe.data.seq_length == 2048
        assert recipe.data.global_batch_size == 16

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 4), (2, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            global_batch_size=16,
            micro_batch_size=2,
            seq_length=2048,
            model_size="test",
        )
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_tokenizer_recipe(self, recipe_module):
        tokenizer_config = recipe_module.tokenizer()
        assert isinstance(tokenizer_config, run.Config)
        assert tokenizer_config.__fn_or_cls__.__name__ == "get_nmt_tokenizer"
        assert tokenizer_config.library == "byte-level"

    def test_invalid_model_size(self, recipe_module):
        with pytest.raises(NotImplementedError, match="Unsupported model size: invalid_size"):
            recipe_module.model_recipe(model_size="invalid_size", tp_comm_overlap=False, seq_length=2048)

    def test_wandb_logger(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(
            dataset_config=None,
            global_batch_size=16,
            micro_batch_size=2,
            num_nodes=1,
            num_gpus_per_node=4,
            seq_length=2048,
            model_size="test",
            wandb_project="test_project",
        )

        # Ensure the log configuration exists
        assert recipe.log is not None

        # Validate the wandb_logger configuration
        assert recipe.log.wandb.project == "test_project"

    def test_callbacks(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(
            dataset_config=None,
            global_batch_size=16,
            micro_batch_size=2,
            num_nodes=1,
            num_gpus_per_node=4,
            seq_length=2048,
            model_size="test",
            tflops_callback=True,
        )
        assert any(
            callback.__fn_or_cls__.__name__ == "FLOPsMeasurementCallback" for callback in recipe.trainer.callbacks
        )

    def test_nsys_profiling(self, recipe_module):
        recipe = recipe_module.pretrain_recipe(
            dataset_config=None,
            global_batch_size=16,
            micro_batch_size=2,
            num_nodes=1,
            num_gpus_per_node=4,
            seq_length=2048,
            model_size="test",
            nsys_profiling=True,
            nsys_start_step=10,
            nsys_end_step=20,
        )
        assert any(callback.__fn_or_cls__.__name__ == "NsysCallback" for callback in recipe.trainer.callbacks)
