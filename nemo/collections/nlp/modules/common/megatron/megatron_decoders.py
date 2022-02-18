# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from nemo.collections.nlp.modules.common.megatron.megatron_transformer_decoder import MegatronTransformerDecoderModule
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()


__all__ = []

AVAILABLE_DECODERS = ["transformer"]


def get_decoder_model(
    arch,
    hidden_size,
    ffn_hidden_size,
    num_layers,
    num_attention_heads,
    apply_query_key_layer_scaling=True,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
    init_method_std=0.02,
    use_cpu_initialization=False,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    layernorm_epsilon=1e-5,
    bias_gelu_fusion=True,
    masked_softmax_fusion=True,
    persist_layer_norm=False,
    openai_gelu=False,
    activation="gelu",
    onnx_safe=False,
    hidden_steps=-1,
    hidden_blocks=1,
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads

    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    if arch == "transformer":
        # Language model.
        decoder = MegatronTransformerDecoderModule(
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            decoder_attn_mask_type=decoder_attn_mask_type,
            pre_process=pre_process,
            post_process=post_process,
            use_cpu_initialization=use_cpu_initialization,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            bias_gelu_fusion=bias_gelu_fusion,
            masked_softmax_fusion=masked_softmax_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown decoder arch = {arch}. Available decoder arch = {AVAILABLE_DECODERS}")

    return decoder
