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

from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from torch import nn

from nemo.collections.vlm.vision.siglip_vit import SigLIPViT400M_14_384_Config
from nemo.lightning import io
from nemo.utils.import_utils import safe_import_from

TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")


@dataclass
class Gemma3VLVisionConfig(SigLIPViT400M_14_384_Config):
    """Gemma3 VL vision model base config"""

    img_h: int = 896
    img_w: int = 896
    image_token_id: int = 262144


@dataclass
class Gemma3VLMultimodalProjectorConfig(TransformerConfig, io.IOMixin):
    """Gemma3 VL multimodal projector config"""

    input_size: int = 1152
    hidden_size: int = 2560

    image_size: int = 896
    patch_dim: int = 14
    tokens_per_image: int = 256

    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    layernorm_epsilon: float = 1e-6

    # Do not change
    num_layers: int = 1
    num_attention_heads: int = 8

    def configure_model(self) -> "Gemma3VLMultimodalProjector":
        """Get module"""
        return Gemma3VLMultimodalProjector(self)


class Gemma3VLMultimodalProjector(MegatronModule):
    """Gemma3 VL multimodal projector"""

    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__(config=config)

        self.patches_per_side = config.image_size // config.patch_dim
        tokens_per_side = int(config.tokens_per_image**0.5)
        kernel_size = self.patches_per_side // tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)

        # TODO: fuse layer norm with proj
        self.mm_soft_embed_norm = TENorm(config, config.input_size, eps=config.layernorm_epsilon)
        self.proj = ColumnParallelLinear(
            input_size=config.input_size,
            output_size=config.hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=True,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name=None,
        )

    def forward(self, x):
        """Downsample, norm and projection"""
        # (B, 64*64, M)
        batch_size, _, hidden_size = x.shape
        # (B, M, S)
        x = x.transpose(1, 2)
        # (B, M, 64, 64)
        x = x.reshape(batch_size, hidden_size, self.patches_per_side, self.patches_per_side).contiguous()
        # (B, M, 16, 16)
        x = self.avg_pool(x)
        # (B, M, 256)
        x = x.flatten(2)
        # (B, 256, M)
        x = x.transpose(1, 2)
        # (B, 256, M)
        x = self.mm_soft_embed_norm(x)
        # (B, 256, D)
        x, _ = self.proj(x)
        return x
