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

from nemo.collections import llm
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes import mamba2_780m
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import Trainer


class TestMamba2_780M:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return mamba2_780m

    def test_tokenizer(self, recipe_module):
        tokenizer_config = recipe_module.tokenizer()
        assert isinstance(tokenizer_config, run.Config)
        assert tokenizer_config.__fn_or_cls__ == get_nmt_tokenizer
        assert tokenizer_config.library == 'huggingface'
        assert tokenizer_config.model_name == "EleutherAI/gpt-neox-20b"
        assert tokenizer_config.use_fast is True

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == llm.GPTModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == llm.BaseMambaConfig780M

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 1
        assert trainer_config.max_steps == 100
        assert trainer_config.val_check_interval == 100
        assert trainer_config.limit_test_batches == 50
        assert trainer_config.limit_val_batches == 32
        assert trainer_config.log_every_n_steps == 10
        assert trainer_config.use_distributed_sampler is False

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_dtype is None
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is False
        assert trainer_config.strategy.gradient_as_bucket_view is True
        assert trainer_config.strategy.ckpt_async_save is False
        assert trainer_config.strategy.ckpt_parallel_load is True

    def test_pretrain_recipe(self, recipe_module):
        recipe = recipe_module.pretrain_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == pretrain
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == llm.GPTModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == MockDataModule
        assert recipe.data.seq_length == 4096
        assert recipe.data.global_batch_size == 8
        assert recipe.data.micro_batch_size == 1

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(
            resume_path="test_path", tokenizer_model="test_tokenizer", peft_scheme=None
        )
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert recipe.optim.config.lr == 5e-6

    def test_finetune_recipe_with_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid_scheme"):
            recipe_module.finetune_recipe(
                resume_path="test_path", tokenizer_model="test_tokenizer", peft_scheme="invalid_scheme"
            )
