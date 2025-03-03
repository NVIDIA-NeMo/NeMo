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

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from nemo.collections.llm.gpt.model.megatron.hyena.hyena_block import HyenaStack, HyenaStackSubmodules
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_layer import HyenaLayer, HyenaLayerSubmodules
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_mixer import HyenaMixer, HyenaMixerSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

    def _raise_te_import_error(*args, **kwargs):
        raise ImportError("Transformer Engine is not installed")

    # NeMo has a number of tests that make sure that you can initialize some modules without TE installed.
    TENorm = _raise_te_import_error
    TELayerNormColumnParallelLinear = _raise_te_import_error
    TERowParallelLinear = _raise_te_import_error
    TEDotProductAttention = _raise_te_import_error

# Layer spec with TE modules
if HAVE_TE:
    hyena_stack_spec = ModuleSpec(
        module=HyenaStack,
        submodules=HyenaStackSubmodules(
            hyena_layer=ModuleSpec(
                module=HyenaLayer,
                submodules=HyenaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=HyenaMixer,
                        submodules=HyenaMixerSubmodules(
                            dense_projection=TELayerNormColumnParallelLinear, dense=TERowParallelLinear
                        ),
                    ),
                    hyena_bda=get_bias_dropout_add,
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    mlp=ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )
else:
    hyena_stack_spec = ModuleSpec(module=None)

# Layer spec without TE modules, for debugging

hyena_stack_spec_no_te = ModuleSpec(
    module=HyenaStack,
    submodules=HyenaStackSubmodules(
        hyena_layer=ModuleSpec(
            module=HyenaLayer,
            submodules=HyenaLayerSubmodules(
                norm=TENorm,
                mixer=ModuleSpec(
                    module=HyenaMixer,
                    submodules=HyenaMixerSubmodules(dense_projection=ColumnParallelLinear, dense=RowParallelLinear),
                ),
                hyena_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=DotProductAttention,
                        linear_proj=RowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=TENorm,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
    ),
)
