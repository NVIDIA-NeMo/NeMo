import torch
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Literal
from pathlib import Path
from nemo.collections.llm.gpt.model.base import GPTModel, gpt_forward_step, gpt_data_step
from megatron.core.transformer.transformer_config import TransformerConfig
from nemo.lightning import get_vocab_size, io, teardown
from megatron.core import parallel_state
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba import MambaModel as MCoreMambaModel

if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

@dataclass
class SSMConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.mamba.mamba_model.MambaModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    num_layers: int = 2
    mamba_ssm_ngroups: int = 8
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    hybrid_override_pattern: str = None
    post_process: bool = True
    pre_process: bool = True
    seq_length: int = 2048
    params_dtype: torch.dtype = torch.bfloat16
    # Mamba with no attention has no need for position embeddings, so none is default
    position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'none'
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    seq_len_interpolation_factor: Optional[float] = None
    apply_rope_fusion: bool = True
    make_vocab_size_divisible_by: int = 128
    gated_linear_unit: bool = False
    fp32_residual_connections: bool = False
    normalization: str = 'RMSNorm'
    add_bias_linear: bool = False
    # TODO: Move this to better places?
    get_attention_mask_from_fusion: bool = False
    
    forward_step_fn: Callable = gpt_forward_step
    data_step_fn: Callable = gpt_data_step

    def configure_model(self, tokenizer) -> "MCoreMambaModel":

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
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )

class SSMModel(GPTModel):

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        attention_mask = None
        output_tensor = self.module(
            input_ids=input_ids, 
            position_ids=position_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        return output_tensor

@io.model_importer(SSMModel, "pytorch")
class PyTorchSSMImporter(io.ModelConnector["SSMModel", SSMModel]):

    def __new__(cls, path: str, model_config=None):
        instance = super().__new__(cls, path)
        instance.model_config = model_config
        return instance
    def init(self) -> SSMModel:

        return SSMModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        
        source = torch.load(str(self), map_location='cpu')
        if 'model' in source:
            source = source['model']

        class ModelState:
            def __init__(self, state_dict, mapping_type="base"):
                self._state_dict = state_dict
                self.mapping_type = mapping_type
                self.update_dict()

            def update_dict(self):
                if self.mapping_type == "base":
                    pattern = re.compile(r'backbone\.layers\.\d+\.norm\.weight')
                elif self.mapping_type == "nvidia":
                    pattern = re.compile(r'decoder\.layers\.\d+\.norm\.weight')
                elif self.mapping_type == "mistral":
                    pattern = re.compile(r'model.backbone\.layers\.\d+\.norm\.weight')
                else:
                    raise AttributeError(f"mapping type [{self.mapping_type}] not found.")
                # Create a new dictionary with the updated keys
                self._state_dict = {
                    (re.sub(r'norm\.weight', 'in_proj_layer_norm_weight', k) if pattern.match(k) else k): v
                    for k, v in self._state_dict.items()
                    }
            def state_dict(self):
                return self._state_dict

        source = ModelState(source, mapping_type="base")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted SSM model to Nemo, model saved to {output_path}")

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
                'backbone.layers.*.in_proj_layer_norm_weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',  
                'backbone.norm_f.weight': 'decoder.final_norm.weight',
                'lm_head.weight': 'output_layer.weight',
            }
        elif self.model_config.mapping_type == "mistral":
            mapping = {
                'model.backbone.embedding.weight': 'embedding.word_embeddings.weight',
                'model.backbone.layers.*.mixer.A_log': 'decoder.layers.*.mixer.A_log', 
                'model.backbone.layers.*.mixer.D': 'decoder.layers.*.mixer.D', 
                'model.backbone.layers.*.mixer.conv1d.weight': 'decoder.layers.*.mixer.conv1d.weight',
                'model.backbone.layers.*.mixer.conv1d.bias': 'decoder.layers.*.mixer.conv1d.bias',
                'model.backbone.layers.*.mixer.in_proj.weight': 'decoder.layers.*.mixer.in_proj.weight',
                'model.backbone.layers.*.mixer.dt_bias': 'decoder.layers.*.mixer.dt_bias',  
                'model.backbone.layers.*.mixer.out_proj.weight': 'decoder.layers.*.mixer.out_proj.weight',
                'model.backbone.layers.*.mixer.norm.weight': 'decoder.layers.*.mixer.norm.weight',
                'model.backbone.layers.*.in_proj_layer_norm_weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',  
                'model.backbone.norm_f.weight': 'decoder.final_norm.weight',
                'model.lm_head.weight': 'output_layer.weight',
            }       
        elif self.model_config.mapping_type == "nvidia":
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
                'decoder.layers.*.in_proj_layer_norm_weight': 'decoder.layers.*.mixer.in_proj.layer_norm_weight',
                'decoder.layers.*.mlp.linear_fc1.layer_norm_weight': 'decoder.layers.*.mlp.linear_fc1.layer_norm_weight', 
                'decoder.layers.*.mlp.linear_fc1.weight': 'decoder.layers.*.mlp.linear_fc1.weight', 
                'decoder.layers.*.mlp.linear_fc2.weight': 'decoder.layers.*.mlp.linear_fc2.weight',
                'decoder.layers.*.self_attention.linear_proj.weight': 'decoder.layers.*.self_attention.linear_proj.weight', 
                'decoder.layers.*.self_attention.linear_qkv.layer_norm_weight': 'decoder.layers.*.self_attention.linear_qkv.layer_norm_weight', 
                'decoder.layers.*.self_attention.linear_qkv.weight': 'decoder.layers.*.self_attention.linear_qkv.weight',
                'decoder.final_norm.weight': 'decoder.final_norm.weight',
                'output_layer.weight': 'output_layer.weight',
            }
  
        return io.apply_transforms(source, target, mapping=mapping)

    @property
    def tokenizer(self):
        from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

        tokenizer = get_nmt_tokenizer(
            library=self.model_config.tokenizer_library,
            model_name=self.model_config.tokenizer_name,
            tokenizer_model=self.model_config.tokenizer_model_path,
            use_fast=True,
        )

        return tokenizer

    @property
    def config(self) -> SSMConfig:
        return self.model_config

@dataclass
class Mamba2Config370m(SSMConfig):
    hybrid_override_pattern: str = "M"*48
    num_layers: int = 48
    seq_length: int = 2048
    hidden_size: int = 1024
    mamba_ssm_ngroups: int = 1
    ffn_hidden_size: int = 1024
    num_attention_heads: int = 1
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    make_vocab_size_divisible_by: int = 16
    tokenizer_library: str = 'huggingface'
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    mapping_type: str = "base"

@dataclass
class HybridConfig8b(SSMConfig):
    hybrid_override_pattern: str = "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"
    num_layers: int = 56
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    num_query_groups: int = 8
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = 'megatron'
    tokenizer_name: str = "GPTSentencePieceTokenizer"
    mapping_type: str = "nvidia"

@dataclass
class CodestralMamba(SSMConfig):
    hybrid_override_pattern: str = "M"*64
    num_layers: int = 64
    seq_length: int = 4096
    hidden_size: int = 4096
    mamba_ssm_ngroups: int = 8
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    make_vocab_size_divisible_by: int = 128
    tokenizer_library: str = 'megatron'
    tokenizer_name: str = "GPTSentencePieceTokenizer"
    mapping_type: str = "mistral"


__all__ = [
    "SSMModel",
    "SSMConfig",
    "Mamba2Config370m",
    "HybridConfig8b",
    "CodestralMamba"
]
