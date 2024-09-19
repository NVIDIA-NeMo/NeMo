import nemo_run as run
import pytest

from nemo.collections.llm.gpt.model.nemotron import Nemotron3Config4B, NemotronModel
from nemo.collections.llm.recipes import nemotron
from nemo.lightning import Trainer


class TestNemotron:
    @pytest.fixture(scope="class")
    def recipe_module(self):
        return nemotron

    def test_nemotron_model(self, recipe_module):
        model = recipe_module.nemotron_model(version="nemotron3_4b")
        assert isinstance(model, run.Config)
        assert model.__fn_or_cls__ == NemotronModel
        assert isinstance(model.config, run.Config)
        assert model.config.__fn_or_cls__ == Nemotron3Config4B

    def test_model_config_parameters(self, recipe_module):
        model = recipe_module.nemotron_model(version="nemotron3_4b")
        nemotron_config = model.config
        assert nemotron_config.num_layers == 32
        assert nemotron_config.hidden_size == 3072
        assert nemotron_config.seq_length == 4096
        assert nemotron_config.num_attention_heads == 24

    def test_nemotron_trainer(self, recipe_module):
        trainer_config = recipe_module.nemotron_trainer()
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == 8
        assert trainer_config.num_nodes == 1

        # Check strategy configuration
        assert isinstance(trainer_config.strategy, run.Config)
        assert trainer_config.strategy.__fn_or_cls__.__name__ == "MegatronStrategy"
        assert trainer_config.strategy.tensor_model_parallel_size == 2
        assert trainer_config.strategy.pipeline_model_parallel_size == 1
        assert trainer_config.strategy.pipeline_dtype is None
        assert trainer_config.strategy.virtual_pipeline_model_parallel_size is None
        assert trainer_config.strategy.context_parallel_size == 1
        assert trainer_config.strategy.sequence_parallel is False

    @pytest.mark.parametrize("num_nodes,num_gpus_per_node", [(1, 8), (2, 4), (4, 2)])
    def test_trainer_with_different_gpu_configs(self, recipe_module, num_nodes, num_gpus_per_node):
        trainer_config = recipe_module.nemotron_trainer(num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node)
        assert isinstance(trainer_config, run.Config)
        assert trainer_config.__fn_or_cls__ == Trainer
        assert trainer_config.accelerator == "gpu"
        assert trainer_config.devices == num_gpus_per_node
        assert trainer_config.num_nodes == num_nodes

    @pytest.mark.parametrize(
        "tensor_model_parallel_size,pipeline_model_parallel_size,context_parallel_size,sequence_parallel",
        [(2, 2, 4, True), (4, 1, 2, False)],
    )
    def test_trainer_with_different_parallelism_options(
        self,
        recipe_module,
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        sequence_parallel,
    ):
        trainer_config = recipe_module.nemotron_trainer(
            tensor_parallelism=tensor_model_parallel_size,
            pipeline_parallelism=pipeline_model_parallel_size,
            context_parallelism=context_parallel_size,
            sequence_parallelism=sequence_parallel,
        )
        assert trainer_config.strategy.tensor_model_parallel_size == tensor_model_parallel_size
        assert trainer_config.strategy.pipeline_model_parallel_size == pipeline_model_parallel_size
        assert trainer_config.strategy.context_parallel_size == context_parallel_size
        assert trainer_config.strategy.sequence_parallel == sequence_parallel
