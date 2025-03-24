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
from nemo.collections.llm import Llama32EmbeddingConfig1B, LlamaEmbeddingModel
from nemo.collections.llm.api import finetune
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes import llama_embedding_1b
from nemo.lightning import Trainer
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.utils.exp_manager import TimingCallback


class TestLlamaEmbedding_1B:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return llama_embedding_1b

    def test_model(self, recipe_module):
        model_config = recipe_module.model()
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == LlamaEmbeddingModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == Llama32EmbeddingConfig1B

    def test_trainer(self, recipe_module):
        trainer_config = recipe_module.trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 1
        assert trainer_config.max_steps == 1168251
        assert trainer_config.limit_val_batches == 32
        assert trainer_config.limit_test_batches == 50
        assert trainer_config.log_every_n_steps == 10
        assert trainer_config.val_check_interval == 2000
        assert trainer_config.use_distributed_sampler is False

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_dtype is None
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 2
        assert trainer_config.strategy.sequence_parallel is False

        # Check DDP configuration
        assert trainer_config.strategy.ddp.check_for_nan_in_grad is True
        assert trainer_config.strategy.ddp.grad_reduce_in_fp32 is True
        assert trainer_config.strategy.ddp.overlap_grad_reduce is True
        assert trainer_config.strategy.ddp.overlap_param_gather is True
        assert trainer_config.strategy.ddp.average_in_collective is True

    def test_finetune_recipe(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        assert isinstance(recipe, run.Partial)
        assert recipe.__fn_or_cls__ == finetune
        assert isinstance(recipe.model, run.Config)
        assert recipe.model.__fn_or_cls__ == LlamaEmbeddingModel
        assert isinstance(recipe.trainer, run.Config)
        assert recipe.trainer.__fn_or_cls__ == Trainer
        assert isinstance(recipe.data, run.Config)
        assert recipe.data.__fn_or_cls__ == llm.SpecterDataModule  # Different from other Llama models
        assert recipe.data.seq_length == 512  # Default for embedding model
        assert recipe.data.micro_batch_size == 4  # Default for embedding model
        assert recipe.data.global_batch_size == 64  # Default for embedding model
        assert isinstance(recipe.peft, run.Config)
        assert recipe.peft.__fn_or_cls__ == LoRA
        assert recipe.peft.dim == 8
        assert recipe.peft.alpha == 16
        assert recipe.optim.config.lr == 1e-4

    def test_finetune_recipe_with_packed_sequence(self, recipe_module):
        with pytest.raises(AssertionError, match='pack_sequence is not supported for Embedding model finetuning.'):
            recipe_module.finetune_recipe(packed_sequence=True)

    def test_finetune_recipe_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert recipe.optim.config.lr == 5e-6
        assert not hasattr(recipe, 'peft') or recipe.peft is None

    def test_finetune_recipe_with_invalid_peft(self, recipe_module):
        with pytest.raises(ValueError, match="Unrecognized peft scheme: invalid_scheme"):
            recipe_module.finetune_recipe(peft_scheme="invalid_scheme")

    def test_finetune_performance_optimizations(self, recipe_module):
        recipe = recipe_module.finetune_recipe()
        recipe = recipe_module.finetune_performance_optimizations(recipe, peft_scheme='lora')
        assert recipe.trainer.strategy.tensor_model_parallel_size == 1
        assert recipe.peft.target_modules == ['linear_qkv']
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == TimingCallback for cb in recipe.trainer.callbacks
        )
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__ == GarbageCollectionCallback
            for cb in recipe.trainer.callbacks
        )

    def test_finetune_performance_optimizations_without_peft(self, recipe_module):
        recipe = recipe_module.finetune_recipe(peft_scheme=None)
        recipe = recipe_module.finetune_performance_optimizations(recipe, peft_scheme=None)
        assert recipe.trainer.plugins.grad_reduce_in_fp32 is False
        assert recipe.trainer.strategy.ddp.grad_reduce_in_fp32 is False
        assert recipe.trainer.strategy.ddp.overlap_grad_reduce is True
        assert recipe.trainer.strategy.ddp.overlap_param_gather is True
        assert recipe.trainer.strategy.ddp.average_in_collective is True
        assert any(
            isinstance(cb, run.Config) and cb.__fn_or_cls__.__name__ == "MegatronCommOverlapCallback"
            for cb in recipe.trainer.callbacks
        )
