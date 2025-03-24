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

from nemo.collections.llm import SpecterDataModule
from nemo.collections.llm.api import finetune
from nemo.collections.llm.recipes import e5_340m
from nemo.lightning import Trainer


class TestE5_340M:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return e5_340m

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        # Note: Actual model class assertions would depend on bert_embedding_model implementation
        # which isn't shown in the provided code

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == SpecterDataModule
        assert recipe.data.seq_length == 512
        assert recipe.data.global_batch_size == 32
        assert recipe.data.micro_batch_size == 4

    def test_finetune_recipe_with_custom_values(self, recipe_module):
        recipe = recipe_module.finetune_recipe(
            seq_length=1024,
            micro_batch_size=8,
            global_batch_size=64,
        )
        assert recipe.data.seq_length == 1024
        assert recipe.data.micro_batch_size == 8
        assert recipe.data.global_batch_size == 64

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_finetune_recipe_with_different_configurations(self, recipe_module, num_nodes, num_gpus_per_node):
        recipe = recipe_module.finetune_recipe(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert recipe.trainer.num_nodes == num_nodes
        assert recipe.trainer.devices == num_gpus_per_node

    def test_finetune_recipe_with_peft(self, recipe_module):
        with pytest.raises(AssertionError, match="E5 only supports SFT."):
            recipe_module.finetune_recipe(peft_scheme='lora')

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert not hasattr(recipe, 'peft') or recipe.peft is None
