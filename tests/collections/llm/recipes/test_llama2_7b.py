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

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.llama import Llama2Config7B, LlamaModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import llama2_7b
from nemo.lightning import Trainer
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.utils.exp_manager import TimingCallback


class TestLlama2_7B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama2_7b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama2Config7B

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
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_dtype is None
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is False

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
        assert recipe.data.seq_length == 4096
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
        assert recipe.data.seq_length == 2048  # Default for unpacked sequence
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.dim == 8
        assert recipe.peft.alpha == 16
        assert recipe.optim.config.lr == 1e-4

    def test_finetune_recipe_with_packed_sequence(self, recipe_module):
        recipe = recipe_module.finetune_recipe(packed_sequence=True)
        assert recipe.data.seq_length == 4096
        assert recipe.data.dataset_kwargs == {'pad_to_max_length': True}
        assert hasattr(recipe.data, 'packed_sequence_specs')
        assert recipe.data.packed_sequence_specs.packed_sequence_size == 4096

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_pretrain_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.pretrain_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 2
        assert recipe.optim.config.lr == 5e-6
        assert not hasattr(recipe, 'peft') or recipe.peft is None

    def test_finetune_recipe_with_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid_scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid_scheme")

    def test_finetune_performance_optimizations(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == TimingCallback for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == GarbageCollectionCallback
            for cb in recipe.trainer.callbacks
        )

    def test_finetune_performance_optimizations_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True, peft_scheme=None)
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False
        assert recipe.trainer.strategy.ddp.grad_reduce_in_fp32 is False
        assert recipe.trainer.strategy.ddp.overlap_grad_reduce is True
        assert recipe.trainer.strategy.ddp.overlap_param_gather is True
        assert recipe.trainer.strategy.ddp.average_in_collective is True
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback"
            for cb in recipe.trainer.callbacks
        )

    def test_finetune_performance_optimizations_with_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(performance_mode=True, peft_scheme='lora')
        assert recipe.peft.target_modules == ['linear_qkv']
