# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from typing import Literal, Optional

from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from nemo.collections.diffusion.models.dit.dit_layer_spec import AdaLN


class MoviegGenLayer(TransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    DiT with Adapative Layer Normalization.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
    ):
        def _replace_no_cp_submodules(submodules):
            modified_submods = copy.deepcopy(submodules)
            modified_submods.cross_attention = IdentityOp
            # modified_submods.temporal_self_attention = IdentityOp
            return modified_submods

        # Replace any submodules that will have CP disabled and build them manually later after TransformerLayer init.
        modified_submods = _replace_no_cp_submodules(submodules)
        super().__init__(
            config=config, submodules=modified_submods, layer_number=layer_number, hidden_dropout=hidden_dropout
        )

        # Override Cross Attention to disable CP.
        # Disable TP Comm overlap as well. Not disabling will attempt re-use of buffer size same as Q and lead to incorrect tensor shapes.
        cp_override_config = copy.deepcopy(config)
        cp_override_config.context_parallel_size = 1
        cp_override_config.tp_comm_overlap = False
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=cp_override_config,
            layer_number=layer_number,
        )

        self.adaLN = AdaLN(config=self.config, n_adaln_chunks=6, norm=TENorm)

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # timestep embedding
        timestep_emb = attention_mask
        factorized_pos_emb = rotary_pos_emb
        hidden_states = hidden_states + factorized_pos_emb

        # ******************************************** full self attention ******************************************************
        shift_full, scale_full, gate_full, shift_mlp, scale_mlp, gate_mlp = self.adaLN(timestep_emb)

        # adaLN with scale + shift
        pre_full_attn_layernorm_output_ada = self.adaLN.modulated_layernorm(
            hidden_states, shift=shift_full, scale=scale_full
        )

        attention_output, _ = self.self_attention(
            pre_full_attn_layernorm_output_ada,
            attention_mask=None,
            packed_seq_params=None if packed_seq_params is None else packed_seq_params['self_attention'],
        )

        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=attention_output, gate=gate_full)

        # ******************************************** cross attention ******************************************************
        attention_output, _ = self.cross_attention(
            hidden_states,
            attention_mask=context_mask,
            key_value_states=context,
            packed_seq_params=None if packed_seq_params is None else packed_seq_params['cross_attention'],
        )

        # ******************************************** mlp ******************************************************
        pre_mlp_layernorm_output_ada = self.adaLN.modulated_layernorm(
            attention_output, shift=shift_mlp, scale=scale_mlp
        )

        mlp_output, _ = self.mlp(pre_mlp_layernorm_output_ada)
        hidden_states = self.adaLN.scale_add(residual=hidden_states, x=mlp_output, gate=gate_mlp)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        elif use_te and fp8:
            linear_fc1 = TEColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            raise ValueError("Invalid combination of use_te and moe_grouped_gemm")

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=MoELayer,
            submodules=MoESubmodules(
                experts=(
                    MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                    if not moe_grouped_gemm or use_te_grouped_gemm
                    else None
                ),
                shared_experts=ModuleSpec(
                    module=SharedExpertMLP,
                    params={"gate": False},
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    ),
                ),
            ),
        )


def get_dit_llama_spec(num_experts=None, attn_mask_type=AttnMaskType.padding) -> ModuleSpec:
    params = {"attn_mask_type": attn_mask_type}
    return ModuleSpec(
        module=MoviegGenLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params=params,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params=params,
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=_get_mlp_module_spec(use_te=True, num_experts=num_experts, moe_grouped_gemm=True, fp8=None),
        ),
    )
