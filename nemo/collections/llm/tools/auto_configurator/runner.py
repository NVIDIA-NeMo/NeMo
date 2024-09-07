# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import re

from typing import List, Optional

from nemo.collections.llm import GPTModel
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.tools.auto_configurator.core.training_config import generate_grid_search_configs
from nemo.collections.llm.tools.auto_configurator.core.utils import generic_base_config
from nemo.collections.llm.utils import Config, Partial
from nemo.utils import logging

SUPPORTED_MODELS = [
    "gpt3",
    "llama",
    "mixtral",
    "mistral",
    "gemma",
    "nemotron",
]

SUPPORTED_TOKENIZERS = [
    "autotokenizer",
    "sentencepiece",
    "huggingface",
]


class AutoConfigurator:
    """Auto Configurator runner config class."""

    def __init__(
        self,
        model: Config = None,
        num_nodes: int = None,
        data_paths: List = None,
        path_to_logs: str = None,
        tokenizer_type: Optional[str] = "autotokenizer",
        tokenizer_path: Optional[str] = "GPT2BPETokenizer",
        gpus_per_node: Optional[int] = 8,
        gpu_memory_gb: Optional[int] = 80,
        seq_length: Optional[int] = 2048,
        global_batch_size: Optional[int] = "auto",
        tensor_parallel_sizes: Optional[List[int]] = "auto",
        pipeline_parallel_sizes: Optional[List[int]] = "auto",
        micro_batch_sizes: Optional[List[int]] = "auto",
        context_parallel_sizes: Optional[List[int]] = [1],
        expert_parallel_sizes: Optional[List[int]] = [1],
        min_model_parallel_size: Optional[int] = "auto",
        max_model_parallel_size: Optional[int] = "auto",
        num_tokens_in_b: Optional[int] = 300,
        tflops_per_gpu: Optional[int] = 140,
        max_minutes_per_run: Optional[int] = 30,
        max_training_days: Optional[int] = 2,
        max_steps_per_run: Optional[int] = 50,
        vocab_size: Optional[int] = 51200,
    ):
        """
        Args:
            model_type (Config): model type to be used for training.
            num_nodes (int): number of nodes to be used for training.
            data_paths (List): list of datafiles to be used for training.
            path_to_logs (str): path to the directory where the logs will be stored.
            tokenizer_type (Optional[str]): tokenizer type.
            tokenizer_path (Optional[str]): path to the tokenizer model.
            model_size (Optional[int]): size of model to be trained.
            gpus_per_node (Optional[int]): number of GPUs per node to be used.
            gpu_memory_gb (Optional[int]): memory per GPU, in GB. Currently 40GB and 80GB A100s/H100s supported.
            seq_length (Optional[int]): model sequence length. Available seq_length list for GPT-based models: [2048, 4096, 8192, 16384, 32768].
            global_batch_size (Optional[int]): model global batch size. Set to "auto" if you want auto configurator to find optimal gbs.
            tensor_parallel_sizes (Optional[List[int]]): set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
            pipeline_parallel_sizes (Optional[List[int]]): set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
            micro_batch_sizes (Optional[List[int]]): set to "auto" to use our recommendation, or a list, such as [1, 2, 4, 8].
            context_parallel_sizes (Optional[List[int]]): model context parallel size. A list, such as [1, 2, 4, 8].
            expert_parallel_sizes (Optional[List[int]]): model expert parallel size. A list, such as [1, 2, 4, 8].
            min_model_parallel_size (Optional[int]): set to "auto" to use our recommendation, or a value for the minimum desired parallelism.
            max_model_parallel_size (Optional[int]): set to "auto" to use our recommendation, or a value for the maximum desired parallelism.
            num_tokens_in_b (Optional[int]): number of tokens in billions in train dataset.
            tflops_per_gpu (Optional[int]): estimated tflops per GPU.
            max_minutes_per_run (Optional[int]): maximum number of minutes per run for the grid search.
            max_training_days (Optional[int]): number of days expected model to be trained.
            max_steps_per_run (Optional[int]): maximum number of steps per run for the grid search.
            vocab_size (Optional[int]): size of tokenizer vocabulary.
        """

        # Print out the config
        config = locals()
        config.pop('self')
        for key, value in config.items():
            setattr(self, key, value)
        logging.info(self._get_message(config))

        model_type = self._get_model_type(model)
        assert model_type in SUPPORTED_MODELS, f"model_type must be set to one of {SUPPORTED_MODELS}."
        assert tokenizer_type in SUPPORTED_TOKENIZERS, f"tokenizer_type must be set to one of {SUPPORTED_TOKENIZERS}."
        assert num_nodes, "num_nodes value must be specified."
        assert data_paths, "training data must be specified."
        assert path_to_logs, f"path_to_logs parameter must be specified."
        gpu_count = num_nodes * gpus_per_node
        assert gpu_count > 0, "num_nodes * gpus_per_node must be an int larger than zero."
        assert gpu_memory_gb in (
            40,
            80,
        ), "gpu_memory_gb can only be 40 or 80."
        assert max_minutes_per_run >= 10, "max_minutes_per_run must be an int and be at least 10 minutes."

        self.model_type = model_type
        self.model_size_in_b = self._get_model_size(model)
        self.gpu_count = gpu_count
        self.num_gpus = gpus_per_node

    def _get_message(self, config: dict) -> str:
        """
        Function that returns runner config line by line.

        Args:
            config (dict): runner config.

        Returns:
            str: runner config params.
        """

        message = "AutoConfigurator runner config:\n"
        for key, value in config.items():
            message += f"{key}: {value}\n"

        return message

    def _get_model_type(self, model: Config) -> str:
        """
        Function that returns model type from model class name.

        Args:
            models (Config): model object.

        Returns:
            str: model type.
        """

        match = re.search(r"\w+\d+[MB]", str(model))
        if match:
            model = match.group(0)

        if "GPT" in model:
            return "gpt3"
        elif "Llama" in model:
            return "llama"
        elif "Mixtral" in model:
            return "mixtral"
        elif "Mistral" in model:
            return "mistral"
        elif "Gemma" in model:
            return "gemma"
        elif "Nemotron" in model:
            return "nemotron"
        else:
            return None

    def _get_model_size(self, model: Config) -> int:
        """
        Function that returns model size from model class name.

        Args:
            model (Config): model class name.

        Returns:
            int: model size.
        """
        match = re.search(r'(\d+)([BM])', str(model))
        if match:
            size = int(match.group(1))
            measure = match.group(2)
            if measure == 'B':
                return size
            elif measure == 'M':
                return size / 1000  # Convert millions to billions
        return None


def generate_configs(runner_config: AutoConfigurator = None) -> dict:
    """
    Function that returns a dictionary of Partial configs.

    Args:
        config (AutoConfigurator): Auto Configurator object.

    Returns:
        dict: dictionary of Partial configs.
    """

    # Generate base config for the given model size
    base_cfg, train_cfg = generic_base_config(runner_config)

    # Launch grid search for training constraints
    base_config, train_configs = generate_grid_search_configs(base_cfg, train_cfg)

    tokenizer = base_config.tokenizer
    model = Config(GPTModel, config=base_config.model, tokenizer=tokenizer)

    configs = {}
    for name, config in train_configs.items():
        trainer = copy.deepcopy(base_config.trainer)
        data = copy.deepcopy(base_config.data)
        log = copy.deepcopy(base_config.log)

        # Set data params
        data.micro_batch_size = config.get("micro_batch_size")
        data.global_batch_size = config.get("global_batch_size")

        # Set strategy params
        trainer.strategy.tensor_model_parallel_size = config.get("tensor_model_parallel_size")
        trainer.strategy.pipeline_model_parallel_size = config.get("pipeline_model_parallel_size")
        trainer.strategy.context_parallel_size = config.get("context_parallel_size")
        trainer.strategy.expert_model_parallel_size = config.get("expert_model_parallel_size")
        trainer.strategy.virtual_pipeline_model_parallel_size = config.get(
            "virtual_pipeline_model_parallel_size", None
        )
        if config.get("tensor_model_parallel_size") > 1:
            trainer.strategy.sequence_parallel = True

        # Set the directory where to save the logs
        configs[name] = Partial(
            pretrain,
            model=model,
            trainer=trainer,
            data=data,
            optim=base_config.optim,
            log=log,
            resume=None,
        )

    return base_cfg, configs
