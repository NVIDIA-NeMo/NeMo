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

from typing import Optional

from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import Attention, AttentionMaskType, GatedMLP, PositionEmbeddingType, RmsNorm
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from tensorrt_llm.module import Module
from tensorrt_llm.quantization import QuantMode
from typing_extensions import override

from nemo.export.trt_llm.decoder.decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder
from nemo.export.trt_llm.model_config import (
    LINEAR_COLUMN,
    LINEAR_ROW,
    AttentionConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
)


class GemmaDecoderLayer(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )

        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attention_head_size=config.head_size,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )

        mlp_hidden_size = config.hidden_size * 4 if config.intermediate_size is None else config.intermediate_size

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=config.hidden_act,
            dtype=config.dtype,
            bias=config.mlp_bias,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        medusa_packed_mask=None,  # For Medusa support
        medusa_position_offsets=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            medusa_packed_mask=medusa_packed_mask,  # For Medusa support
            medusa_position_offsets=medusa_position_offsets,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
        )

        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class GemmaDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The LLAMA implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        return layer.mlp.act_fn

    @override
    def infer_num_attention_heads(self, layer):
        return layer.self_attn.num_heads

    @override
    def infer_num_kv_heads(self, layer):
        return layer.self_attn.num_key_value_heads

    @override
    def infer_max_position_embeddings(self, layer):
        return layer.self_attn.max_position_embeddings

    @override
    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.input_layernorm, dtype=self.dtype)

    @override
    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.self_attn.o_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.gate_proj, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.down_proj, LINEAR_ROW, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )
        config.gate = LinearConfig.from_nn_module(
            layer.mlp.up_proj, LINEAR_COLUMN, rank=self.rank, tensor_parallel=self.tensor_parallel, dtype=self.dtype,
        )

        return config

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.post_attention_layernorm, dtype=self.dtype)


class GemmaDecoderLayerBuilder(DecoderLayerBuilder):
    """The Gemma implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_scaling = None
        if layer.rotary_scaling is not None:
            rotary_scaling = {"type": "linear", "factor": float(layer.rotary_scaling)}

        config = PretrainedConfig(
            architecture=None,
            dtype=self.dtype,
            logits_dtype=self.dtype,
            vocab_size=layer.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_kv_heads,
            head_size=layer.kv_channels,
            hidden_act=self.hidden_act.split("-")[-1] if layer.moe_num_experts else non_gated_version(self.hidden_act),
            intermediate_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            norm_epsilon=layer.norm_epsilon,
            position_embedding_type="rope_gpt_neox",
            world_size=self.tensor_parallel,
            tp_size=self.tensor_parallel,
            pp_size=1,
            quant_mode=QuantMode(0),
            quant_kwargs=None,
            max_lora_rank=layer.max_lora_rank,
        )

        config.set_if_not_exist('mlp_bias', False)
        config.set_if_not_exist('attn_bias', False)
        config.set_if_not_exist('rotary_base', layer.rotary_base)
        config.set_if_not_exist('rotary_scaling', rotary_scaling)
        config.set_if_not_exist('enable_pos_shift', False)
        config.set_if_not_exist('dense_context_fmha', False)
        config.set_if_not_exist('moe_num_experts', 0)

        return GemmaDecoderLayer(config=config, layer_idx=self.layer_id,)
