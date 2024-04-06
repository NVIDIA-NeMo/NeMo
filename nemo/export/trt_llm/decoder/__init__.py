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

from typing import Dict, Type

import tensorrt as trt

from nemo.export.trt_llm.decoder.decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder
from nemo.export.trt_llm.decoder.falcon import FALCONDecoderLayerBuilder, FALCONDecoderLayerConfigBuilder
from nemo.export.trt_llm.decoder.gemma import GemmaDecoderLayerBuilder, GemmaDecoderLayerConfigBuilder
from nemo.export.trt_llm.decoder.gpt import GPTDecoderLayerBuilder, GPTDecoderLayerConfigBuilder
from nemo.export.trt_llm.decoder.gptj import GPTJDecoderLayerBuilder, GPTJDecoderLayerConfigBuilder
from nemo.export.trt_llm.decoder.llama import LLAMADecoderLayerBuilder, LLAMADecoderLayerConfigBuilder
from nemo.export.trt_llm.model_config import (
    DECODER_FALCON,
    DECODER_GEMMA,
    DECODER_GPT2,
    DECODER_GPTJ,
    DECODER_GPTNEXT,
    DECODER_LLAMA,
    QUANTIZATION_NONE,
)

DECODER_CONFIG_REGISTRY: Dict[str, Type[DecoderLayerConfigBuilder]] = {
    DECODER_GPT2: GPTDecoderLayerConfigBuilder,
    DECODER_GPTJ: GPTJDecoderLayerConfigBuilder,
    DECODER_LLAMA: LLAMADecoderLayerConfigBuilder,
    DECODER_FALCON: FALCONDecoderLayerConfigBuilder,
    DECODER_GEMMA: GemmaDecoderLayerConfigBuilder,
}


def build_decoder_layer_config(layer, decoder: str, dtype=trt.float16, rank=0, tensor_parallel=1):
    """Builds the decoder layer config with the input torch module."""
    assert decoder in DECODER_CONFIG_REGISTRY, f"{decoder} not supported"
    return DECODER_CONFIG_REGISTRY[decoder](decoder, dtype, rank, tensor_parallel).build_layer(layer)


DECODER_REGISTRY: Dict[str, Type[DecoderLayerBuilder]] = {
    DECODER_GPT2: GPTDecoderLayerBuilder,
    DECODER_GPTJ: GPTJDecoderLayerBuilder,
    DECODER_LLAMA: LLAMADecoderLayerBuilder,
    DECODER_GPTNEXT: GPTDecoderLayerBuilder,
    DECODER_FALCON: FALCONDecoderLayerBuilder,
    DECODER_GEMMA: GemmaDecoderLayerBuilder,
}


def build_decoder_layer(
    layer,
    layer_id: int,
    num_layers: int,
    dtype=trt.float16,
    quantization=QUANTIZATION_NONE,
    rank=0,
    tensor_parallel=1,
    tp_group=None,
):
    """Builds the tensorrt llm decoder layer module with the layer config as the input."""
    assert layer.decoder_type in DECODER_REGISTRY, f"{layer.decoder_type} not supported"
    builder = DECODER_REGISTRY[layer.decoder_type]
    decoder_builder = builder(layer, layer_id, num_layers, dtype, quantization, rank, tensor_parallel, tp_group)
    return decoder_builder.decoder
