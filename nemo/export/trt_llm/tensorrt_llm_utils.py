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

import logging

import tensorrt as trt
from tensorrt_llm.layers import Embedding, LayerNorm, PromptTuningEmbedding, RmsNorm
from tensorrt_llm.module import Module

from nemo.export.trt_llm.model_config import LAYERNORM_DEFAULT, LAYERNORM_RMS, EmbeddingConfig, LayernormConfig
from nemo.export.trt_llm.tensor_utils import get_tensor_parallel_group

LOGGER = logging.getLogger("NeMo")


def build_embedding_from_config(
    config: EmbeddingConfig,
    dtype: trt.DataType,
    tensor_parallel: int = 1,
    tensor_parallel_rank: int = 0,
    use_prompt_tuning: bool = False,
):
    """Returns the tensorrt_llm embedding layer from the embedding config."""
    # If the config is empty, return an empty impl.
    if config is None:
        return None
    EmbeddingCls = PromptTuningEmbedding if use_prompt_tuning else Embedding

    trt_embedding = EmbeddingCls(
        config.weight.shape[0] * tensor_parallel,
        config.weight.shape[1],
        dtype=dtype,
        tp_size=tensor_parallel,
        tp_rank=tensor_parallel_rank,
        tp_group=get_tensor_parallel_group(tensor_parallel),
    )
    trt_embedding.weight.value = config.weight
    return trt_embedding


def build_layernorm_from_config(config: LayernormConfig, dtype: trt.DataType):
    """Returns the tensorrt_llm layernorm layer from the torch layernorm."""
    # If the config is empty, return an empty impl.
    if config is None:
        return None

    if config.layernorm_type == LAYERNORM_DEFAULT:
        trt_layernorm = LayerNorm(normalized_shape=config.weight.shape[0], dtype=dtype)
        trt_layernorm.weight.value = config.weight
        trt_layernorm.bias.value = config.bias
    elif config.layernorm_type == LAYERNORM_RMS:
        trt_layernorm = RmsNorm(normalized_shape=config.weight.shape[0], dtype=dtype)
        trt_layernorm.weight.value = config.weight
    else:
        raise NotImplementedError(f"{config.layernorm_type} not supported")
    return trt_layernorm


def print_tensorrt_llm(name: str, tensorrt_llm_module: Module):
    """Prints the tensorrt llm structure including weights and related data for debugging purpose."""
    for tensor_name in [
        "weight",
        "bias",
        "activation_scaling_factor",
        "weights_scaling_factor",
        "prequant_scaling_factor",
    ]:
        if hasattr(tensorrt_llm_module, tensor_name):
            tensor = getattr(tensorrt_llm_module, tensor_name)
            if tensor is not None:
                LOGGER.info(f"{name}.{tensor_name}:{tensor._value.dtype}:{tensor._value.shape}:\n{tensor._value}")

    for k, v in tensorrt_llm_module.named_children():
        print_tensorrt_llm(f"{name}.{k}({v._get_name()})", v)
