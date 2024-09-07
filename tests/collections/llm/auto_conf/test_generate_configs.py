import nemo_run as run

from nemo.collections.llm import (
    GemmaConfig7B,
    GPTConfig5B,
    Llama3Config70B,
    MistralConfig7B,
    MixtralConfig8x22B,
    Nemotron3Config8B,
)
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator, generate_configs


def get_auto_configs(configs):
    auto_configs = []
    for run_name, config in configs.items():
        auto_configs.append(
            [
                config.trainer.strategy.tensor_model_parallel_size,
                config.trainer.strategy.pipeline_model_parallel_size,
                config.trainer.strategy.context_parallel_size,
                config.trainer.strategy.expert_model_parallel_size,
                config.data.micro_batch_size,
            ]
        )

    return auto_configs


class TestGenerateConfgis:
    def test_gpt_model(self):
        # GPT3 126M
        runner = AutoConfigurator(
            model=run.Config(GPTConfig5B),
            num_nodes=16,
            seq_length=2048,
            global_batch_size=2048,
            tensor_parallel_sizes=[4],
            pipeline_parallel_sizes=[2],
            micro_batch_sizes=[1, 2],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=8,
            max_model_parallel_size=8,
            data_paths="/",
            path_to_logs="/",
        )

        _, configs = generate_configs(runner)

        mbs = [1, 2]
        for run_name, config, mb in zip(configs.keys(), configs.values(), mbs):
            assert config.data.micro_batch_size == mb
            assert config.data.seq_length == 2048
            assert config.data.global_batch_size == 2048

        assert len(configs) == 2, f"{len(configs)} configurations were generated but 2 were expected."

        auto_configs = get_auto_configs(configs)
        assert auto_configs[0] == [
            4,
            2,
            1,
            1,
            1,
        ], f"[4, 2, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            4,
            2,
            1,
            1,
            2,
        ], f"[4, 2, 1, 1, 2] is expected configuration output but got {auto_configs[1]}."

    def test_llama_model(self):
        # Llama3 70B
        runner = AutoConfigurator(
            model=run.Config(Llama3Config70B),
            num_nodes=128,
            seq_length=8192,
            global_batch_size=2048,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes=[1],
            context_parallel_sizes=[1, 2, 4],
            expert_parallel_sizes=[1],
            min_model_parallel_size=16,
            max_model_parallel_size=64,
            data_paths="/",
            path_to_logs="/",
        )

        _, configs = generate_configs(runner)

        mbs = [1, 1, 1]
        for run_name, config, mb in zip(configs.keys(), configs.values(), mbs):
            assert config.data.micro_batch_size == mb
            assert config.data.seq_length == 8192
            assert config.data.global_batch_size == 2048

        assert len(configs) == 3, f"{len(configs)} configurations were generated but 3 were expected."

        auto_configs = get_auto_configs(configs)
        assert auto_configs[0] == [
            4,
            1,
            4,
            1,
            1,
        ], f"[4, 1, 4, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            8,
            1,
            2,
            1,
            1,
        ], f"[8, 1, 2, 1, 1] is expected configuration output but got {auto_configs[1]}."

        assert auto_configs[2] == [
            8,
            1,
            4,
            1,
            1,
        ], f"[8, 1, 4, 1, 1] is expected configuration output but got {auto_configs[2]}."

    def test_mistral_model(self):
        # Mistral 7B
        runner = AutoConfigurator(
            model=run.Config(MistralConfig7B),
            num_nodes=16,
            seq_length=4096,
            global_batch_size=2048,
            tensor_parallel_sizes=[4],
            pipeline_parallel_sizes=[1, 2],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
            data_paths="/",
            path_to_logs="/",
        )

        _, configs = generate_configs(runner)

        mbs = [1, 1]
        for run_name, config, mb in zip(configs.keys(), configs.values(), mbs):
            assert config.data.micro_batch_size == mb
            assert config.data.seq_length == 4096
            assert config.data.global_batch_size == 2048

        assert len(configs) == 2, f"{len(configs)} configurations were generated but 3 were expected."

        auto_configs = get_auto_configs(configs)
        assert auto_configs[0] == [
            4,
            1,
            1,
            1,
            1,
        ], f"[4, 1, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            4,
            2,
            1,
            1,
            1,
        ], f"[4, 2, 1, 1, 1] is expected configuration output but got {auto_configs[1]}."

    def test_mixtral_model(self):
        # Mixtral 8x22B
        runner = AutoConfigurator(
            model=run.Config(MixtralConfig8x22B),
            num_nodes=16,
            seq_length=4096,
            global_batch_size=2048,
            tensor_parallel_sizes=[4],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1, 2],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
            data_paths="/",
            path_to_logs="/",
        )

        _, configs = generate_configs(runner)

        mbs = [1, 1]
        for run_name, config, mb in zip(configs.keys(), configs.values(), mbs):
            assert config.data.micro_batch_size == mb
            assert config.data.seq_length == 4096
            assert config.data.global_batch_size == 2048

        assert len(configs) == 2, f"{len(configs)} configurations were generated but 3 were expected."

        auto_configs = get_auto_configs(configs)
        assert auto_configs[0] == [
            4,
            1,
            1,
            1,
            1,
        ], f"[4, 1, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            4,
            1,
            1,
            2,
            1,
        ], f"[4, 1, 1, 2, 1] is expected configuration output but got {auto_configs[1]}."

    def test_gemma_model(self):
        # Gemma 7B
        runner = AutoConfigurator(
            model=run.Config(GemmaConfig7B),
            num_nodes=16,
            seq_length=8192,
            global_batch_size=2048,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[2],
            micro_batch_sizes=[1, 2],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
            data_paths="/",
            path_to_logs="/",
        )

        _, configs = generate_configs(runner)

        mbs = [1, 2]
        for run_name, config, mb in zip(configs.keys(), configs.values(), mbs):
            assert config.data.micro_batch_size == mb
            assert config.data.seq_length == 8192
            assert config.data.global_batch_size == 2048

        assert len(configs) == 2, f"{len(configs)} configurations were generated but 3 were expected."

        auto_configs = get_auto_configs(configs)
        assert auto_configs[0] == [
            2,
            2,
            1,
            1,
            1,
        ], f"[2, 2, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            2,
            2,
            1,
            1,
            2,
        ], f"[2, 2, 1, 1, 2] is expected configuration output but got {auto_configs[1]}."

    def test_nemotron_model(self):
        # Nemotron3 8B
        runner = AutoConfigurator(
            model=run.Config(Nemotron3Config8B),
            num_nodes=16,
            seq_length=4096,
            global_batch_size=2048,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[4],
            micro_batch_sizes=[1, 2],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
            data_paths="/",
            path_to_logs="/",
        )

        _, configs = generate_configs(runner)

        mbs = [1, 2]
        for run_name, config, mb in zip(configs.keys(), configs.values(), mbs):
            assert config.data.micro_batch_size == mb
            assert config.data.seq_length == 4096
            assert config.data.global_batch_size == 2048

        assert len(configs) == 2, f"{len(configs)} configurations were generated but 3 were expected."

        auto_configs = get_auto_configs(configs)
        assert auto_configs[0] == [
            1,
            4,
            1,
            1,
            1,
        ], f"[2, 2, 1, 1, 1] is expected configuration output but got {auto_configs[0]}."

        assert auto_configs[1] == [
            1,
            4,
            1,
            1,
            2,
        ], f"[2, 2, 1, 1, 2] is expected configuration output but got {auto_configs[1]}."
