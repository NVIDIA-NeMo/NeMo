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

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import lightning.pytorch as L
import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from nemo.collections.vlm.vision.base import CLIPViTConfig
from nemo.lightning import io, teardown


class InternViTRMSNorm(torch.nn.Module):

    def __init__(
            self,
            config,
            hidden_size: int,
            eps: float = 1e-6,
            sequence_parallel: bool = False,
            compute_var: bool = False,
    ):
        """Custom RMSNorm for InternViT.

        Args:
            config (TransformerConfig): Config.
            hidden_size (int): Input hidden size.
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
            compute_var (bool): Indicator to compute statistic manually.
        """
        super().__init__()
        self.config = config
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self._compute_var = compute_var

        assert not sequence_parallel, "Sequence parallelism is not supported with InternViT."

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x, var):
        if var is None:
            var = x.pow(2).mean(-1, keepdim=True)

        return x * torch.rsqrt(var + self.eps)

    def forward(self, x):
        """Run RMSNorm with an option to compute custom statistic."""
        var = None
        if self._compute_var:
            unpadded_hidden_size = self.config.hidden_size  # 3200
            max_dim = x.shape[-1]  # 128

            x = x.reshape(x.size(0), x.size(1), -1)
            var = self._gather_var(x.float().pow(2), max_dim) / unpadded_hidden_size

        output = self._norm(x.float(), var).type_as(x)
        output = output * self.weight

        if self._compute_var:
            output = output.reshape(output.size(0), output.size(1), -1, max_dim)

        return output

    def _gather_var(self, input_, max_dim, valid_ranks=6):
        """Compute statistic across the non-dummy heads."""
        world_size = get_tensor_model_parallel_world_size()
        # assert world_size == 8, "tested only with TP=8"

        # Size and dimension.
        last_dim = input_.dim() - 1
        rank = get_tensor_model_parallel_rank()

        if rank < valid_ranks:  # Ranks 0-5 have 24 non-dummy attention heads.
            var = input_.sum(-1, keepdim=True)
        elif rank == valid_ranks:  # Rank 6 has 1 non-dummy attention head.
            var = input_[..., :max_dim].sum(-1, keepdim=True)
        else:
            var = input_.sum(-1, keepdim=True) * 0.0  # Zero-out the dummy heads.

        tensor_list = [torch.empty_like(var) for _ in range(world_size)]
        tensor_list[rank] = var
        torch.distributed.all_gather(tensor_list, var, group=get_tensor_model_parallel_group())

        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output.sum(-1, keepdim=True)


def get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )


# Handle InternViT's layer scaling.
def _bias_dropout_add_func_internvit(ls, x_with_bias, residual, prob, training):
    x, bias = x_with_bias  # unpack
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
    if bias is not None:
        x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out * ls
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out * ls
        return out


def bias_dropout_add_unfused_internvit(ls, training):
    """Bias-dropout-add as in Megatron but with added LayerScaling handling."""

    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func_internvit(ls, x_with_bias, residual, prob, training)

    return _bias_dropout_add


def get_bias_dropout_add_internvit(ls, training, fused):
    """Bias-dropout-add as in Megatron but with added LayerScaling handling."""
    assert not fused, "Fused bias-dropout-add not implemented for InternViT."
    return bias_dropout_add_unfused_internvit(ls, training)


# Add InternViT specialties to our default TransformerLayer.
class InternViTTransformerLayer(TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ls1 = torch.nn.Parameter(torch.ones(self.config.hidden_size))
        self.ls2 = torch.nn.Parameter(torch.ones(self.config.hidden_size))

        self.self_attn_bda = partial(self.self_attn_bda, self.ls1)
        self.mlp_bda = partial(self.mlp_bda, self.ls2)


# Override a few things that are special in InternViT and not supported by the SelfAttention class.
class InternViTSelfAttention(SelfAttention):
    def __init__(
            self, config: TransformerConfig, submodules: SelfAttentionSubmodules, *args, **kwargs
    ):
        super().__init__(config=config, submodules=submodules, *args, **kwargs)

        # Need to override linear_qkv, q_layernorm and k_layernorm.
        qkv_bias = False

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        qk_layernorm_hidden_size = (
                self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        )  # 512 for internvit
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=qk_layernorm_hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
            compute_var=True,
        )

        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=qk_layernorm_hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
            compute_var=True,
        )


class InternViTTEDotProductAttention(TEDotProductAttention):
    """Adjusted Attention for InternViT"""

    def forward(self, *args, **kwargs):
        """Regular TEDotProductAttention + zero-out dummy attention heads."""
        out = super().forward(*args, **kwargs)

        # This makes sure the dummy attention heads are zeroed out.
        mask = torch.ones_like(out, dtype=out.dtype, device=out.device)
        rank = get_tensor_model_parallel_rank()
        max_dim = out.shape[-1]  # 128
        valid_ranks = 6

        if rank == valid_ranks:
            mask[..., max_dim:] *= 0.0
        elif rank > valid_ranks:
            mask *= 0.0
        out *= mask

        return out


def get_internvit_layer_spec(use_te) -> ModuleSpec:
    mlp = get_mlp_module_spec(use_te)  # no norm

    return ModuleSpec(
        module=InternViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=InternViTRMSNorm,
            self_attention=ModuleSpec(
                module=InternViTSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                    core_attention=TEDotProductAttention if use_te else DotProductAttention,
                    linear_proj=TERowParallelLinear if use_te else RowParallelLinear,
                    q_layernorm=InternViTRMSNorm,
                    k_layernorm=InternViTRMSNorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add_internvit,
            pre_mlp_layernorm=InternViTRMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add_internvit,
        ),
    )


@dataclass
class InternViTConfig(CLIPViTConfig):
    vision_model_type: str = "internvit"
    patch_dim: int = 14
    img_h: int = 448
    img_w: int = 448
    num_layers: int = 45
    num_attention_heads: int = 25
    num_query_groups: int = 25
    kv_channels: int = 128
    add_bias_linear: bool = True
    add_qkv_bias: bool = False
    hidden_size: int = 3200
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 12800
    gated_linear_unit: bool = False
    activation_func: Callable = torch.nn.functional.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'RMSNorm'
    layernorm_epsilon: float = 1e-6
    apply_rope_fusion: bool = False
    transformer_layer_spec: ModuleSpec = get_internvit_layer_spec(use_te=True)

@dataclass
class InternViT_6B_448px_V1_5_Config(InternViTConfig):
    vision_model_type: str = "internvit"


class InternVitModel(L.LightningModule, io.IOMixin, io.ConnectorMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model()


@io.model_importer(InternVitModel, "hf")
class HFInternVitImporter(io.ModelConnector["InternVisionModel", InternVitModel]):
    def init(self) -> InternVitModel:
        return InternVitModel(self.config)

    def apply(self, output_path: Path) -> Path:
        from transformers import AutoModel
        source = AutoModel.from_pretrained(str(self), trust_remote_code=True)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        print(f"Converted Llava model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Llava model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        mapping = {
            # Embeddings
            "embeddings.class_embedding": "class_token",
            "embeddings.patch_embedding.weight": "conv1.weight",
            "embeddings.patch_embedding.bias": "conv1.bias",

            # Transformer Layers
            "encoder.layers.*.ls1": "decoder.layers.*.ls1",
            "encoder.layers.*.ls2": "decoder.layers.*.ls2",

            # Attention QKV
            "encoder.layers.*.attn.q_norm.weight": "decoder.layers.*.self_attention.q_layernorm.weight",
            "encoder.layers.*.attn.k_norm.weight": "decoder.layers.*.self_attention.k_layernorm.weight",
            "encoder.layers.*.attn.proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "encoder.layers.*.attn.proj.bias": "decoder.layers.*.self_attention.linear_proj.bias",

            # MLP
            "encoder.layers.*.mlp.fc1.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "encoder.layers.*.mlp.fc1.bias": "decoder.layers.*.mlp.linear_fc1.bias",
            "encoder.layers.*.mlp.fc2.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "encoder.layers.*.mlp.fc2.bias": "decoder.layers.*.mlp.linear_fc2.bias",

            # Layer Norm
            "encoder.layers.*.norm1.weight": "decoder.layers.*.input_layernorm.weight",
            "encoder.layers.*.norm2.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
        }

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_import_position_embedding, _import_qkv],
        )

    @property
    def config(self) -> CLIPViTConfig:
        from transformers import AutoConfig

        source = AutoConfig.from_pretrained(str(self), trust_remote_code=True)
        output = InternViTConfig(
            patch_dim=source.patch_size,
            attention_dropout=source.attention_dropout,
            hidden_size=source.hidden_size,
            img_h=source.image_size,
            img_w=source.image_size,
            ffn_hidden_size=source.intermediate_size,
            layernorm_epsilon=source.layer_norm_eps,
            num_attention_heads=source.num_attention_heads,
            num_layers=source.num_hidden_layers,
        )
        return output


@io.state_transform(
    source_key="embeddings.position_embedding",
    target_key="position_embeddings.weight",
)
def _import_position_embedding(ctx: io.TransformCTX, pos_emb):
    return pos_emb.squeeze(0)


@io.state_transform(
    source_key="encoder.layers.*.attn.qkv.weight",
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, qkv):

    def import_qkv(q, k, v, head_num, num_query_groups, heads_per_group, hidden_size, head_size):
        old_tensor_shape = q.size()
        new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
        new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

        q = q.view(*new_q_tensor_shape)
        k = k.view(*new_kv_tensor_shape)
        v = v.view(*new_kv_tensor_shape)

        qkv_weights_l = []
        for i in range(num_query_groups):
            qkv_weights_l.append(q[i * heads_per_group: (i + 1) * heads_per_group, :, :])
            qkv_weights_l.append(k[i: i + 1, :, :])
            qkv_weights_l.append(v[i: i + 1, :, :])
        qkv_weights = torch.cat(qkv_weights_l)
        assert qkv_weights.ndim == 3, qkv_weights.shape
        assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
        assert qkv_weights.shape[1] == head_size, qkv_weights.shape
        assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

        qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

        return qkv_weights

    megatron_config = ctx.target.config
    q, k, v = qkv.chunk(3)
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )
