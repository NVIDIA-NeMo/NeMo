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


import numpy as np
from tensorrt_llm.layers import Linear, RowLinear
from tensorrt_llm.quantization.layers import FP8Linear, FP8RowLinear, Int8SmoothQuantLinear, Int8SmoothQuantRowLinear

from nemo.export.trt_llm.model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_INT8_SQ,
    QUANTIZATION_NONE,
    LinearConfig,
    ModelConfig,
)


def quantize_linear(tensorrt_llm_layer, quantization: str, layer_config: LinearConfig):
    """Returns the quantized tensorrt_llm linear layer."""
    if quantization == QUANTIZATION_NONE:
        return tensorrt_llm_layer

    if quantization == QUANTIZATION_FP8:
        # FP8 is not sensitive to scaling factors. So we just quantize all layers possible.
        default_scaling_factor = np.array([1], dtype=np.float32)
        if layer_config.activation_scaling_factor is None:
            layer_config.activation_scaling_factor = default_scaling_factor
        if layer_config.weights_scaling_factor is None:
            layer_config.weights_scaling_factor = default_scaling_factor

    if layer_config.activation_scaling_factor is None or layer_config.weights_scaling_factor is None:
        print(f"No valid scaling factors in {tensorrt_llm_layer._get_name()}, skipping quantization" " on this layer")
        return tensorrt_llm_layer
    else:
        assert np.all(layer_config.activation_scaling_factor > 0)
        assert np.all(layer_config.weights_scaling_factor > 0)

    bias = tensorrt_llm_layer.bias is not None

    linear_layer_type = type(tensorrt_llm_layer)
    if linear_layer_type == Linear:
        if quantization == QUANTIZATION_FP8:
            linear = FP8Linear
        elif quantization == QUANTIZATION_INT8_SQ:
            linear = Int8SmoothQuantLinear
        else:
            assert False, f"{quantization} is not supported."
        quantized_linear_layer = linear(
            in_features=tensorrt_llm_layer.in_features,
            out_features=tensorrt_llm_layer.out_features * tensorrt_llm_layer.tp_size,
            bias=bias,
            dtype=tensorrt_llm_layer.dtype,
            tp_group=tensorrt_llm_layer.tp_group,
            tp_size=tensorrt_llm_layer.tp_size,
            gather_output=tensorrt_llm_layer.gather_output,
        )
    elif linear_layer_type == RowLinear:
        if quantization == QUANTIZATION_FP8:
            row_linear = FP8RowLinear
        elif quantization == QUANTIZATION_INT8_SQ:
            row_linear = Int8SmoothQuantRowLinear
        else:
            assert False, f"{quantization} is not supported."
        quantized_linear_layer = row_linear(
            in_features=tensorrt_llm_layer.in_features * tensorrt_llm_layer.tp_size,
            out_features=tensorrt_llm_layer.out_features,
            bias=bias,
            dtype=tensorrt_llm_layer.dtype,
            tp_group=tensorrt_llm_layer.tp_group,
            tp_size=tensorrt_llm_layer.tp_size,
        )
    else:
        assert False, f"{linear_layer_type} is not supported."

    quantized_linear_layer.weight = tensorrt_llm_layer.weight
    quantized_linear_layer.bias = tensorrt_llm_layer.bias

    quantized_linear_layer.activation_scaling_factor.value = layer_config.activation_scaling_factor
    quantized_linear_layer.weights_scaling_factor.value = layer_config.weights_scaling_factor

    if hasattr(quantized_linear_layer, "prequant_scaling_factor"):
        quantized_linear_layer.prequant_scaling_factor.value = layer_config.prequant_scaling_factor

    return quantized_linear_layer


def naive_quantization(config: ModelConfig, quantization: str):
    """Generates a constant scaling factor (1) with target quantization.

    This is for debugging and performance measurement only.
    """
    config.quantization = quantization
    # Here the scaling factor is not inversed.
    # In nvidia systems:
    # pytorch_quantization uses inv scale
    # onnx & trt uses non-inv scale
    # cask uses inv scale
    default_scaling_factor = np.array([1], dtype=np.float32)

    if quantization == QUANTIZATION_FP8:
        for layer in config.layers:
            linear_layers = [
                layer.attention.qkv,
                layer.attention.dense,
                layer.mlp.fc,
                layer.mlp.proj,
                layer.mlp.gate,
            ]
            for linear_layer in linear_layers:
                if linear_layer:
                    linear_layer.activation_scaling_factor = default_scaling_factor
                    linear_layer.weights_scaling_factor = default_scaling_factor
        config.lm_head.activation_scaling_factor = default_scaling_factor
        config.lm_head.weights_scaling_factor = default_scaling_factor

    else:
        assert False, f"{quantization} not supported"
