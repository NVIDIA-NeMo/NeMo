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

from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, model_validator


def infer_global_batch_size(
    micro_batch_size: int,
    num_nodes: int,
    devices: int,
    accumulate_grad_batches: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_model_parallel_size: int = 1,
) -> int:
    """Infers the global batch size based on the micro batch size, number of nodes, devices, accumulation of gradient
        batches, and model parallel sizes.

    Args:
        micro_batch_size (int): The micro batch size.
        num_nodes (int): The number of nodes.
        devices (int): The number of devices.
        accumulate_grad_batches (int): The accumulation of gradient batches. Defaults to 1.
        tensor_model_parallel_size (int): The tensor model parallel size. Defaults to 1.
        pipeline_model_parallel_size (int): The pipeline model parallel size. Defaults to 1.
        context_model_parallel_size (int): The context model parallel size. Defaults to 1.

    Returns:
        int: The global batch size.
    """
    if not all(
        isinstance(arg, int)
        for arg in [
            micro_batch_size,
            num_nodes,
            devices,
            accumulate_grad_batches,
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_model_parallel_size,
        ]
    ):
        raise ValueError(
            f"All arguments must be of type int, got {type(micro_batch_size)}, {type(num_nodes)}, {type(devices)}, "
            f"{type(accumulate_grad_batches)}, {type(tensor_model_parallel_size)}, "
            f"{type(pipeline_model_parallel_size)}, and {type(context_model_parallel_size)}"
        )
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be greater than 0, got {micro_batch_size}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be greater than 0, got {num_nodes}")
    if devices <= 0:
        raise ValueError(f"devices must be greater than 0, got {devices}")
    if accumulate_grad_batches <= 0:
        raise ValueError(f"accumulate_grad_batches must be greater than 0, got {accumulate_grad_batches}")
    if tensor_model_parallel_size <= 0:
        raise ValueError(f"tensor_model_parallel_size must be greater than 0, got {tensor_model_parallel_size}")
    if pipeline_model_parallel_size <= 0:
        raise ValueError(f"pipeline_model_parallel_size must be greater than 0, got {pipeline_model_parallel_size}")
    if context_model_parallel_size <= 0:
        raise ValueError(f"context_model_parallel_size must be greater than 0, got {context_model_parallel_size}")

    world_size = num_nodes * devices
    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size) != 0:
        raise ValueError(
            f"world_size must be divisible by tensor_model_parallel_size * pipeline_model_parallel_size *"
            f" context_model_parallel_size, got {world_size} and TP{tensor_model_parallel_size} * "
            f"PP{pipeline_model_parallel_size} * CP{context_model_parallel_size}"
        )

    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_model_parallel_size
    data_parallel_size = world_size // model_parallel_size
    global_batch_size = micro_batch_size * data_parallel_size * accumulate_grad_batches
    return global_batch_size


class Evo2BlendedDatasetConfig(BaseModel):
    """Configuration for blended dataset specifications.

    Validates and constructs dataset paths, weights and splits configuration.
    Ensures dataset paths exist and are properly resolved relative to base data path.

    Attributes:
        dataset_path: Base directory path for datasets. Used to resolve relative dataset prefixes.
        dataset_prefix: Path prefix for dataset files. Can be absolute or relative to dataset_path.
        dataset_weight: Weight factor for this dataset during blending (0-1).
        dataset_split: Dataset partition - 'train', 'validation' or 'test'.

    Raises:
        ValueError: If dataset path doesn't exist or prefix can't be resolved.
    """

    dataset_path: str | None = None
    dataset_prefix: str
    dataset_weight: float
    dataset_split: Literal["train", "validation", "test"]

    @model_validator(mode="before")
    @classmethod
    def validate_dataset_prefix(cls, values: dict) -> dict:
        """Ensure dataset_prefix paths exist and are properly resolved or are relative to base dataset_path if
            provided.

        Args:
            values (dict): Dictionary containing dataset_path and dataset_prefix.

        Returns:
            dict: Dictionary containing validated dataset_path and dataset_prefix.
        """
        dataset_path = Path(values.get("dataset_path")) if values.get("dataset_path") else None
        prefix = Path(values.get("dataset_prefix"))

        if not prefix.is_absolute():
            if dataset_path:
                prefix = dataset_path / prefix
            else:
                prefix = Path(prefix).resolve()
        parent = prefix.parent
        stem = prefix.stem
        if not parent.exists():
            raise ValueError(f"dataset_prefix parent path does not exist: {parent}")
        matching_files = list(parent.glob(f"{stem}.*"))
        if not matching_files:
            raise ValueError(f"dataset_prefix file does not exist: {prefix}")
        values["dataset_prefix"] = str(prefix)
        return values


def parse_dataset_config(dataset_config_path: str, dataset_path: Optional[str] = None):
    """Parse the blended training datasplit configuration and renormalize data split weights for training Hyena.

    Args:
        dataset_config_path (str): Path to the dataset configuration YAML file.
        dataset_path (str): Path to the dataset directory. Defaults to None.

    Returns:
        defaultdict: A dictionary where keys are dataset splits and values are lists containing the normalized weight
                     and dataset prefix for each split.
    """
    blended_dataset_config = defaultdict(list)
    weight_sums = defaultdict(float)
    with open(dataset_config_path, "r") as config_file:
        dataset_config_batch = yaml.safe_load(config_file)
        for dataset_config in dataset_config_batch:
            # Validate.
            config_model = Evo2BlendedDatasetConfig(dataset_path=dataset_path, **dataset_config)
            # Integrate the weights for renormalization.
            weight_sums[config_model.dataset_split] += abs(config_model.dataset_weight)
        for dataset_config in dataset_config_batch:
            # Validate.
            config_model = Evo2BlendedDatasetConfig(dataset_path=dataset_path, **dataset_config)
            # Add indexed dataset to split and associate with blended training weight.
            blended_dataset_config[config_model.dataset_split].extend(
                [config_model.dataset_weight / weight_sums[config_model.dataset_split], config_model.dataset_prefix]
            )
    return blended_dataset_config
