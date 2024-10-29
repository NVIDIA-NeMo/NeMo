# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from functools import partial

from nemo.collections import llm
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
    def test_llama_model(self):
        # Llama3 70B
        recipe = partial(llm.llama3_70b.pretrain_recipe, num_nodes=128, num_gpus_per_node=8)()
        recipe.data.global_batch_size = 2048
        runner = AutoConfigurator(
            recipe=recipe,
            tensor_parallel_sizes="auto",
            pipeline_parallel_sizes="auto",
            micro_batch_sizes=[1],
            context_parallel_sizes=[1, 2, 4],
            expert_parallel_sizes=[1],
            min_model_parallel_size=16,
            max_model_parallel_size=64,
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
        recipe = partial(llm.mistral_7b.pretrain_recipe, num_nodes=16, num_gpus_per_node=8)()
        recipe.data.seq_length = 4096
        recipe.data.global_batch_size = 2048
        recipe.model.config.seq_length = recipe.data.seq_length

        runner = AutoConfigurator(
            recipe=recipe,
            tensor_parallel_sizes=[4],
            pipeline_parallel_sizes=[1, 2],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
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
        recipe = partial(llm.mixtral_8x22b.pretrain_recipe, num_nodes=16, num_gpus_per_node=8)()
        recipe.data.seq_length = 4096
        recipe.data.global_batch_size = 2048
        recipe.model.config.seq_length = recipe.data.seq_length

        runner = AutoConfigurator(
            recipe=recipe,
            tensor_parallel_sizes=[4],
            pipeline_parallel_sizes=[1],
            micro_batch_sizes=[1],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1, 2],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
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
        recipe = partial(llm.gemma_7b.pretrain_recipe, num_nodes=16, num_gpus_per_node=8)()
        recipe.data.seq_length = 8192
        recipe.data.global_batch_size = 2048
        recipe.model.config.seq_length = recipe.data.seq_length

        runner = AutoConfigurator(
            recipe=recipe,
            tensor_parallel_sizes=[2],
            pipeline_parallel_sizes=[2],
            micro_batch_sizes=[1, 2],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
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
        recipe = partial(llm.nemotron3_8b.pretrain_recipe, num_nodes=16, num_gpus_per_node=8)()
        recipe.data.seq_length = 4096
        recipe.data.global_batch_size = 2048
        recipe.model.config.seq_length = recipe.data.seq_length

        runner = AutoConfigurator(
            recipe=recipe,
            tensor_parallel_sizes=[1],
            pipeline_parallel_sizes=[4],
            micro_batch_sizes=[1, 2],
            context_parallel_sizes=[1],
            expert_parallel_sizes=[1],
            min_model_parallel_size=4,
            max_model_parallel_size=8,
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
