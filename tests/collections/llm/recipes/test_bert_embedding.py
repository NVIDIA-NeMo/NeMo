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

from nemo.collections.llm import BertEmbeddingLargeConfig, BertEmbeddingMiniConfig, BertEmbeddingModel
from nemo.collections.llm.recipes import bert_embedding
from nemo.lightning import Trainer
from nemo.utils.exp_manager import TimingCallback


class TestBertEmbedding:

    def test_bert_embedding_model_110m(self):
        model_config = bert_embedding.bert_embedding_model("bert_110m")
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == BertEmbeddingModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == BertEmbeddingMiniConfig

    def test_bert_embedding_model_340m(self):
        model_config = bert_embedding.bert_embedding_model("bert_340m")
        assert isinstance(model_config, run.Config)
        assert model_config.__fn_or_cls__ == BertEmbeddingModel
        assert isinstance(model_config.config, run.Config)
        assert model_config.config.__fn_or_cls__ == BertEmbeddingLargeConfig

    def test_bert_embedding_model_invalid_version(self):
        with pytest.raises(AssertionError, match="Invalid BERT version: invalid_version"):
            bert_embedding.bert_embedding_model("invalid_version")

    def test_bert_trainer_default_settings(self):
        trainer = bert_embedding.bert_trainer()
        assert isinstance(trainer, run.Config)
        assert trainer.__fn_or_cls__ == Trainer

        # Check default parallelism settings
        assert trainer.strategy.tensor_model_parallel_size == 2
        assert trainer.strategy.pipeline_model_parallel_size == 1
        assert trainer.strategy.pipeline_dtype is None
        assert trainer.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer.strategy.context_parallel_size == 1
        assert trainer.strategy.sequence_parallel is False

        # Check default training settings
        assert trainer.max_steps == 1168251
        assert trainer.accumulate_grad_batches == 1
        assert trainer.limit_test_batches == 32
        assert trainer.limit_val_batches == 32
        assert trainer.log_every_n_steps == 10
        assert trainer.val_check_interval == 2000

    def test_bert_trainer_custom_settings(self):
        trainer = bert_embedding.bert_trainer(
            tensor_parallelism=4,
            pipeline_parallelism=2,
            pipeline_parallelism_type=torch.float16,
            virtual_pipeline_parallelism=4,
            context_parallelism=2,
            sequence_parallelism=True,
            num_nodes=2,
            num_gpus_per_node=4,
            max_steps=500000,
            precision="16-mixed",
            accumulate_grad_batches=2,
            limit_test_batches=64,
            limit_val_batches=64,
            log_every_n_steps=20,
            val_check_interval=1000,
        )

        # Check custom parallelism settings
        assert trainer.strategy.tensor_model_parallel_size == 4
        assert trainer.strategy.pipeline_model_parallel_size == 2
        assert trainer.strategy.pipeline_dtype == torch.float16
        assert trainer.strategy.virtual_pipeline_model_parallel_size == 4
        assert trainer.strategy.context_parallel_size == 2
        assert trainer.strategy.sequence_parallel is True

        # Check custom training settings
        assert trainer.max_steps == 500000
        assert trainer.accumulate_grad_batches == 2
        assert trainer.limit_test_batches == 64
        assert trainer.limit_val_batches == 64
        assert trainer.log_every_n_steps == 20
        assert trainer.val_check_interval == 1000
        assert trainer.num_nodes == 2
        assert trainer.devices == 4

    def test_bert_trainer_with_callbacks(self):
        callbacks = [run.Config(TimingCallback)]
        trainer = bert_embedding.bert_trainer(callbacks=callbacks)
        assert trainer.callbacks == callbacks

    def test_bert_trainer_ddp_settings(self):
        trainer = bert_embedding.bert_trainer()
        assert trainer.strategy.ddp.check_for_nan_in_grad is True
        assert trainer.strategy.ddp.grad_reduce_in_fp32 is True
        assert trainer.strategy.ddp.overlap_grad_reduce is False
        assert trainer.strategy.ddp.overlap_param_gather is True
        assert trainer.strategy.ddp.average_in_collective is True
