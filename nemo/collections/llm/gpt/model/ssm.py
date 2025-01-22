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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional
import torch
import torch.nn.functional as F
from nemo.utils import logging
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_data_step, torch_dtype_from_mcore_config
from nemo.lightning import get_vocab_size, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from transformers import Nemotron5Config, Nemotron5ForCausalLM, AutoTokenizer

try:
    from megatron.core import parallel_state
    from megatron.core.models.mamba import MambaModel as MCoreMambaModel
    from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
    from megatron.core.dist_checkpointing.serialization import load_plain_tensors
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    HAVE_MEGATRON_CORE_OR_TE = True

except (ImportError, ModuleNotFoundError):
    logging.warning("The package `megatron.core` was not imported in this environment which is needed for SSMs.")
    HAVE_MEGATRON_CORE_OR_TE = False

def ssm_forward_step(model, batch) -> torch.Tensor:

    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }
    forward_args["attention_mask"] = None
    return model(**forward_args)

def dist_ckpt_handler(checkpoint_dir):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Ensure this port is available
    world_size = 1
    rank = 0
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)  # ckpt conversion done on CPU
    
    state_dict = load_plain_tensors(checkpoint_dir)

    key_list = list(state_dict.keys())
    for k in key_list:
        if "optimizer" in k:
            state_dict.pop(k)
    dist_ckpt_args = state_dict['args']
    state_dict.pop('args')
    state_dict.pop('checkpoint_version')
    state_dict.pop('iteration')
    state_dict.pop('opt_param_scheduler')
    state_dict.pop('num_floating_point_operations_so_far')

    for i, symbol in enumerate(dist_ckpt_args.hybrid_override_pattern):
        if symbol == 'M':
            state_dict[f'decoder.layers.{i}.mixer.in_proj.weight'] = torch.cat(
                [
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.z'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.x'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.B'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.C'],
                    state_dict[f'decoder.layers.{i}.mixer.in_proj.weight.dt'],
                ],
                dim=0,
            )

            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.z')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.x')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.B')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.C')
            state_dict.pop(f'decoder.layers.{i}.mixer.in_proj.weight.dt')

            state_dict[f'decoder.layers.{i}.mixer.conv1d.weight'] = torch.cat(
                [
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.weight.x'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.weight.B'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.weight.C'],
                ],
                dim=0,
            )
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.weight.x')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.weight.B')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.weight.C')

            state_dict[f'decoder.layers.{i}.mixer.conv1d.bias'] = torch.cat(
                [
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.bias.x'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.bias.B'],
                    state_dict[f'decoder.layers.{i}.mixer.conv1d.bias.C'],
                ],
                dim=0,
            )
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.bias.x')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.bias.B')
            state_dict.pop(f'decoder.layers.{i}.mixer.conv1d.bias.C')

    return state_dict, dist_ckpt_args

@dataclass
class SSMConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.mamba.mamba_model.MambaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    fp16: bool = False
    bf16: bool = True
    num_layers: int = 2
    mamba_ssm_ngroups: int = 8
    num_attention_heads: int = 1
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 8192
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none'
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    normalization: str = 'RMSNorm'
    add_bias_linear: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    attention_backend: AttnBackend = AttnBackend.flash
    forward_step_fn: Callable = ssm_forward_step
    data_step_fn: Callable = gpt_data_step
    vocab_file: str = None
    tokenizer_model_path: str = None
    deallocate_pipeline_outputs: bool = True
    bias_dropout_fusion: bool = True

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreMambaModel":

        return MCoreMambaModel(
            self,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            mamba_ssm_ngroups=self.mamba_ssm_ngroups,
            hybrid_attention_ratio=self.hybrid_attention_ratio,
            hybrid_mlp_ratio=self.hybrid_mlp_ratio,
            hybrid_override_pattern=self.hybrid_override_pattern,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
        )


@io.model_importer(GPTModel, "pytorch")
class PyTorchSSMImporter(io.ModelConnector["GPTModel", GPTModel]):

    def __new__(cls, path: str, model_config=None):
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance

    def init(self) -> GPTModel:

        return GPTModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, source_dist_ckpt: bool = False) -> Path:

        if source_dist_ckpt:
            source, dist_ckpt_args = dist_ckpt_handler(
                str(self)
            )
        else:
            source = torch.load(str(self), map_location='cpu')
        if 'model' in source:
            source = source['model']

        class ModelState:
            def __init__(self, state_dict):
                self._state_dict = state_dict

            def state_dict(self):
                return self._state_dict

            def to(self, dtype):
                for k, v in self._state_dict.items():
                    if v.dtype != dtype:
                        logging.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
                    self._state_dict[k] = v.to(dtype)

        source = ModelState(source)
        target = self.init()
        trainer = self.nemo_setup(target)
        source.to(self.config.params_dtype)
        target.to(self.config.params_dtype)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        logging.info(f"Converted SSM model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):

        if self.model_config.mapping_type == "base":
            mapping = {
                'backbone.embedding.weight': 'embedding.word_embeddings.weight',
                'backbone.layers.*.mixer.A_log': 'decoder.layers.*.mixer.A_log',
                'backbone.layers.*.mixer.D': 'decoder.layers.*.mixer.D',
                'backbone.layers.*.mixer.conv1d.weight': 'decoder.layers.*.mixer.conv1d.weight',
                'backbone.layers.*.mixer.conv1d.bias': 'decoder.layers.*.mixer.conv1d.bias',
                'backbone.layers.*.mixer.in_proj.weight': 'decoder.layers.*.mixer.in_proj.weight',
                'backbone.layers.*.mixer.dt_bias': 'decoder.layers.*.mixer.dt_bias',
                'backbone.layers.*.mixer.out_proj.weight': 'decoder.layers.*.mixer.out_proj.weight',
                'backbone.layers.*.mixer.norm.weight': 'decoder.layers.*.mixer.norm.weight',
                'backbone.layers.*.norm.weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',
                'backbone.norm_f.weight': 'decoder.final_norm.weight',
                'lm_head.weight': 'output_layer.weight',
            }
        elif "nvidia" in self.model_config.mapping_type:
            mapping = {
                'embedding.word_embeddings.weight': 'embedding.word_embeddings.weight',
                'decoder.layers.*.mixer.A_log': 'decoder.layers.*.mixer.A_log',
                'decoder.layers.*.mixer.D': 'decoder.layers.*.mixer.D',
                'decoder.layers.*.mixer.conv1d.weight': 'decoder.layers.*.mixer.conv1d.weight',
                'decoder.layers.*.mixer.conv1d.bias': 'decoder.layers.*.mixer.conv1d.bias',
                'decoder.layers.*.mixer.in_proj.weight': 'decoder.layers.*.mixer.in_proj.weight',
                'decoder.layers.*.mixer.dt_bias': 'decoder.layers.*.mixer.dt_bias',
                'decoder.layers.*.mixer.out_proj.weight': 'decoder.layers.*.mixer.out_proj.weight',
                'decoder.layers.*.mixer.norm.weight': 'decoder.layers.*.mixer.norm.weight',
                'decoder.final_norm.weight': 'decoder.final_norm.weight',
                'output_layer.weight': 'output_layer.weight',
            }
            if "nemotron5" in self.model_config.mapping_type:
                mapping.update({
                    'decoder.layers.*.mixer.in_proj.layer_norm_weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',
                })
            else:
                mapping.update({
                    'decoder.layers.*.norm.weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',
                })
            if "hybrid" in self.model_config.mapping_type:
                mapping.update(
                    {
                        'decoder.layers.*.mlp.linear_fc1.layer_norm_weight': 'decoder.layers.*.mlp.linear_fc1.layer_norm_weight',
                        'decoder.layers.*.mlp.linear_fc1.weight': 'decoder.layers.*.mlp.linear_fc1.weight',
                        'decoder.layers.*.mlp.linear_fc2.weight': 'decoder.layers.*.mlp.linear_fc2.weight',
                        'decoder.layers.*.self_attention.linear_proj.weight': 'decoder.layers.*.self_attention.linear_proj.weight',
                        'decoder.layers.*.self_attention.linear_qkv.layer_norm_weight': 'decoder.layers.*.self_attention.linear_qkv.layer_norm_weight',
                        'decoder.layers.*.self_attention.linear_qkv.weight': 'decoder.layers.*.self_attention.linear_qkv.weight',
                    }
                )
        else:
            raise AttributeError(f"mapping type [{self.mapping_type}] not found.")
        return io.apply_transforms(source, target, mapping=mapping)

    @property
    def tokenizer(self):
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=self.model_config.tokenizer_library,
            model_name=self.model_config.tokenizer_name,
            vocab_file=self.model_config.vocab_file,
            tokenizer_model=self.model_config.tokenizer_model_path,
            use_fast=True,
        )

        return tokenizer

    @property
    def config(self) -> SSMConfig:
        return self.model_config


@io.model_importer(GPTModel, "hf")
class HFNemotron5Importer(io.ModelConnector["Nemotron5ForCausalLM", GPTModel]):
    def init(self) -> GPTModel:
        return GPTModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:

        source = Nemotron5ForCausalLM.from_pretrained("/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/nm5_exp/nm5_from_nemo_to_hf")
        target = self.init()
        trainer = self.nemo_setup(target)
        source = source.to(self.config.params_dtype)
        target = target.to(self.config.params_dtype)
        self.convert_state(source, target)
        
        self.nemo_save(output_path, trainer)

        print(f"Converted Nemotron5 Hybrid model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):

        mapping = {
            'backbone.embeddings.weight': 'embedding.word_embeddings.weight',
            'backbone.layers.*.mixer.A_log': 'decoder.layers.*.mixer.A_log',
            'backbone.layers.*.mixer.D': 'decoder.layers.*.mixer.D',
            'backbone.layers.*.mixer.conv1d.weight': 'decoder.layers.*.mixer.conv1d.weight',
            'backbone.layers.*.mixer.conv1d.bias': 'decoder.layers.*.mixer.conv1d.bias',
            'backbone.layers.*.mixer.in_proj.weight': 'decoder.layers.*.mixer.in_proj.weight',
            'backbone.layers.*.mixer.dt_bias': 'decoder.layers.*.mixer.dt_bias',
            'backbone.layers.*.mixer.out_proj.weight': 'decoder.layers.*.mixer.out_proj.weight',
            'backbone.layers.*.mixer.norm.weight': 'decoder.layers.*.mixer.norm.weight',
            'backbone.layers.*.mixer.up_proj.weight': 'decoder.layers.*.mlp.linear_fc1.weight',
            'backbone.layers.*.mixer.down_proj.weight': 'decoder.layers.*.mlp.linear_fc2.weight',
            'backbone.layers.*.mixer.o_proj.weight': 'decoder.layers.*.self_attention.linear_proj.weight',
            'backbone.norm_f.weight': 'decoder.final_norm.weight',
            'lm_head.weight': 'output_layer.weight',
        }
        for i, layer_type in enumerate(source.config.hybrid_override_pattern):
            if layer_type == "M":
                mapping[f'backbone.layers.{i}.norm.weight'] = f'decoder.layers.{i}.mixer.in_proj.layer_norm_weight'
            elif layer_type == "-":
                mapping[f'backbone.layers.{i}.norm.weight'] = f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'
            elif layer_type == "*":
                mapping[f'backbone.layers.{i}.norm.weight'] = f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'
            else:
                raise AttributeError(f"layer type {layer_type} not found.")


        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv])

    @property
    def tokenizer(self) -> "AutoTokenizer":
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets("/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/nm5_exp/nm5_from_nemo_to_hf"))

    @property
    def config(self) -> SSMConfig:
        from transformers import Nemotron5Config as HFNemotron5Config

        source = HFNemotron5Config.from_pretrained("/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/checkpoints/nm5_exp/nm5_from_nemo_to_hf")
        source.torch_dtype = torch.bfloat16
        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = SSMConfig(
            num_layers=source.num_hidden_layers,
            hybrid_override_pattern=source.hybrid_override_pattern,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            layernorm_epsilon=source.layer_norm_epsilon,
            num_query_groups=source.num_key_value_heads,
            mamba_ssm_ngroups=source.n_groups,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(GPTModel, "hf")
class HFNemotron5Exporter(io.ModelConnector[GPTModel, "Nemotron5ForCausalLM"]):
    def init(self, dtype=torch.bfloat16) -> "Nemotron5ForCausalLM":
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return Nemotron5ForCausalLM(self.config)

    def apply(self, output_path: Path) -> Path:
        
        source, _ = self.nemo_load(str(self))
        source = source.to(torch_dtype_from_mcore_config(source.config))
        target = self.init().to(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            self.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target):

        mapping = {
            'decoder.layers.*.mixer.A_log': 'backbone.layers.*.mixer.A_log',
            'decoder.layers.*.mixer.D': 'backbone.layers.*.mixer.D',
            'decoder.layers.*.mixer.conv1d.weight': 'backbone.layers.*.mixer.conv1d.weight',
            'decoder.layers.*.mixer.conv1d.bias': 'backbone.layers.*.mixer.conv1d.bias',
            'decoder.layers.*.mixer.in_proj.weight': 'backbone.layers.*.mixer.in_proj.weight',
            'decoder.layers.*.mixer.dt_bias': 'backbone.layers.*.mixer.dt_bias',
            'decoder.layers.*.mixer.out_proj.weight': 'backbone.layers.*.mixer.out_proj.weight',
            'decoder.layers.*.mixer.norm.weight': 'backbone.layers.*.mixer.norm.weight',
            'decoder.layers.*.mlp.linear_fc1.weight': 'backbone.layers.*.mixer.up_proj.weight',
            'decoder.layers.*.mlp.linear_fc2.weight': 'backbone.layers.*.mixer.down_proj.weight',
            'decoder.layers.*.self_attention.linear_proj.weight': 'backbone.layers.*.mixer.o_proj.weight',
            'decoder.final_norm.weight': 'backbone.norm_f.weight',
        }

        for i, layer_type in enumerate(source.config.hybrid_override_pattern):
            if layer_type == "M":
                mapping[f'decoder.layers.{i}.mixer.in_proj.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
            elif layer_type == "-":
                mapping[f'decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
            elif layer_type == "*":
                mapping[f'decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight'] = f'backbone.layers.{i}.norm.weight'
            else:
                raise AttributeError(f"layer type {layer_type} not found.")

        transforms = [_export_qkv, _export_embedding, _export_head]

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("nvidia/Mistral-NeMo-Minitron-8B-Instruct")

    
    @property
    def config(self) -> "Nemotron5Config":
        source: SSMConfig = io.load_context(str(self), subpath="model.config")

        from transformers import Nemotron5Config as HFNemotron5Config

        return HFNemotron5Config(
            hybrid_override_pattern=source.hybrid_override_pattern,
            n_groups=source.mamba_ssm_ngroups,
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            vocab_size=source.vocab_size,
        )

@io.state_transform(
    source_key=(
        "backbone.layers.*.mixer.q_proj.weight",
        "backbone.layers.*.mixer.k_proj.weight",
        "backbone.layers.*.mixer.v_proj.weight",
    ),
    target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, q, k, v):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
    new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights

@io.state_transform(
    source_key="decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "backbone.layers.*.mixer.q_proj.weight",
        "backbone.layers.*.mixer.k_proj.weight",
        "backbone.layers.*.mixer.v_proj.weight",
    ),
)
def _export_qkv(ctx: io.TransformCTX, linear_qkv):
    megatron_config = ctx.source.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj

@io.state_transform(
    source_key="embedding.word_embeddings.weight",
    target_key="backbone.embeddings.weight",
)
def _export_embedding(ctx: io.TransformCTX, embedding):
    megatron_config = ctx.target.config
    # prune padding.
    return embedding[: megatron_config.vocab_size, :]


@io.state_transform(
    source_key="output_layer.weight",
    target_key="lm_head.weight",
)
def _export_head(ctx: io.TransformCTX, embedding):
    megatron_config = ctx.target.config
    # prune padding.
    return embedding[: megatron_config.vocab_size, :]

@dataclass
class BaseMambaConfig130M(SSMConfig):
    hybrid_override_pattern: str = "M" * 24
    num_layers: int = 24
    seq_length: int = 2048
    hidden_size: int = 768
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 768
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig370M(SSMConfig):
    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1024
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 1024
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig780M(SSMConfig):
    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1536
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 1536
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig1_3B(SSMConfig):
    hybrid_override_pattern: str = "M" * 48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 2048
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 2048
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class BaseMambaConfig2_7B(SSMConfig):
    hybrid_override_pattern: str = "M" * 64
    num_layers: int = 64
    seq_length: int = 2048
    hidden_size: int = 2560
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 2560
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"


@dataclass
class NVIDIAMambaConfig8B(SSMConfig):
    hybrid_override_pattern: str = "M" * 56
    num_attention_heads: int = 32
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 4096
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = 'megatron'
    tokenizer_name: str = "GPTSentencePieceTokenizer"
    mapping_type: str = "nvidia-pure"


@dataclass
class NVIDIAMambaHybridConfig8B(SSMConfig):
    hybrid_override_pattern: str = "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = 'megatron'
    tokenizer_name: str = "GPTSentencePieceTokenizer"
    mapping_type: str = "nvidia-hybrid"

@dataclass
class Nemotron5HybridConfig8B(SSMConfig):
    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    seq_length: int = 8192
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 32
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    activation_func: callable = lambda x: torch.pow(F.relu(x), 2)
    tokenizer_library: str = 'tiktoken'
    tokenizer_name: str = "TiktokenTokenizer"
    mapping_type: str = "nvidia-hybrid-nemotron5"
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    attention_softmax_in_fp32: bool = False
    vocab_size: int = 131072

__all__ = [
    "SSMConfig",
    "BaseMambaConfig130M",
    "BaseMambaConfig370M",
    "BaseMambaConfig780M",
    "BaseMambaConfig1_3B",
    "BaseMambaConfig2_7B",
    "NVIDIAMambaConfig8B",
    "NVIDIAMambaHybridConfig8B",
    "Nemotron5HybridConfig8B",
]
