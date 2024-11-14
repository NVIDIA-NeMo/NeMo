from dataclasses import dataclass
from typing import Optional

import torch
from megatron.core.models.vision.clip_vit_model import CLIPViTModel as MCoreCLIPViTModel
from nemo.collections.llm.fn.activation import openai_gelu, quick_gelu

from nemo.collections.vlm import CLIPViTConfig


@dataclass
class CLIPViTL_14_336_Config(CLIPViTConfig):
    vision_model_type = "clip"
    patch_dim = 14
    img_h = 336
    img_w = 336
    num_layers = 24
    num_attention_heads = 16
    add_bias_linear = True
    add_qkv_bias = True
    hidden_size = 1024
    hidden_dropout = 0.0
    attention_dropout = 0.0
    ffn_hidden_size = 4096
    gated_linear_unit = False
    activation_func = quick_gelu
    kv_channels = 64
    num_query_groups = 16
    layernorm_zero_centered_gamma = False
    apply_query_key_layer_scaling = False
    bias_activation_fusion = False
    bias_dropout_fusion = False
    attention_softmax_in_fp32 = True
    normalization = 'LayerNorm'
    apply_rope_fusion = False


@dataclass
class SigLIPViT400M_14_384_Config(CLIPViTConfig):
    vision_model_type = "siglip"
    patch_dim = 14
    img_h = 384
    img_w = 384
    num_layers = 27
    num_attention_heads = 16
    add_bias_linear = True
    add_qkv_bias = True
    hidden_size = 1152
    hidden_dropout = 0.0
    attention_dropout = 0.0
    ffn_hidden_size = 4304
    gated_linear_unit = False
    activation_func = openai_gelu
    kv_channels = 72
    num_query_groups = 16
    layernorm_zero_centered_gamma = False
    apply_query_key_layer_scaling = False
    bias_activation_fusion = False
    bias_dropout_fusion = False
    attention_softmax_in_fp32 = True
    normalization = 'LayerNorm'
    apply_rope_fusion = False
    qk_layernorm = False
    layernorm_epsilon = 1e-6


class CLIPViTModel(MCoreCLIPViTModel):
    """CLIP ViT vision model."""

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_unused_layers: int = 0
    ) -> torch.Tensor:
        if num_unused_layers > 0:
            unused_layers = self.decoder.layers[-num_unused_layers:]
            self.decoder.layers = self.decoder.layers[:-num_unused_layers]
            x = super().forward(x, attention_mask)
            self.decoder.layers.append(unused_layers)
            return x

        return super().forward(x, attention_mask)
