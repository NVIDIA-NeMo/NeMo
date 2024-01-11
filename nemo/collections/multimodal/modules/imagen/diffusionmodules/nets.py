# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.multimodal.modules.imagen.diffusionmodules.attention import SelfAttentionPooling
from nemo.collections.multimodal.modules.imagen.diffusionmodules.blocks import (
    ConditionalSequential,
    DBlock,
    FusedCrossAttentionBlock,
    ResBlock,
    StackedCrossAttentionBlock,
    UBlock,
)
from nemo.collections.multimodal.modules.imagen.diffusionmodules.embs import (
    LearnedSinusoidalPosEmb,
    UnLearnedSinusoidalPosEmb,
)
from nemo.collections.multimodal.modules.imagen.diffusionmodules.layers import Downsample
from nemo.collections.multimodal.modules.imagen.diffusionmodules.layers import UpsampleLearnable as Upsample
from nemo.collections.multimodal.modules.imagen.diffusionmodules.layers import linear, normalization, zero_module


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding used for Imagen Base and SR model.

    :param embed_dim: Dimension of embeddings. Also used to calculate the number of channels in ResBlock.
    :param image_size: Input image size. Used to calculate where to inject attention layers in UNet.
    :param channels: Input channel number, defaults to 3.
    :param text_embed_dim: Dimension of conditioned text embedding. Different text encoders and different model versions have different values, defaults to 512
    :param num_res_blocks: Number of ResBlock in each level of UNet, defaults to 3.
    :param channel_mult: Used with embed_dim to calculate the number of channels for each level of UNet, defaults to [1, 2, 3, 4]
    :param num_attn_heads: The number of heads in the attention layer, defaults to 4.
    :param per_head_channels: The number of channels per attention head, defaults to 64.
    :param cond_dim: Dimension of Conditioning projections, defaults to 512.
    :param attention_type: Type of attention layer, defaults to 'fused'.
    :param feature_pooling_type: Type of pooling, defaults to 'attention'.
    :param learned_sinu_pos_emb_dim: Dimension of learned time positional embedding. 0 for unlearned timestep embeddings. Defaults to 16
    :param attention_resolutions: List of resolutions to inject attention layers. Defaults to [8, 16, 32]
    :param dropout: The rate of dropout, defaults to 0.
    :param use_null_token: Whether to create a learned null token for attention, defaults to False.
    :param init_conv_kernel_size: Initial Conv kernel size, defaults to 3.
    :param gradient_checkpointing: Whether to use gradient checkpointing, defaults to False.
    :param scale_shift_norm: Whether to use scale shift norm, defaults to False.
    :param stable_attention: Whether to use numerically-stable attention calculation, defaults to True.
    :param flash_attention: Whether to use flash attention calculation, defaults to False.
    :param resblock_updown: Whether to use ResBlock or Downsample/Upsample, defaults to False.
    :param resample_with_conv: When resblock_updown=False, whether to use conv in addition to Pooling&ConvTranspose. Defaults to True.
    :param low_res_cond: Whether conditioned on low-resolution input, used for SR model. Defaults to False.
    :param noise_cond_aug: Whether to add noise conditioned augmentation with low-resolution input. Defaults to False.
    """

    def __init__(
        self,
        embed_dim,  # Dimension of embeddings. Also used to calculate the number of channels in ResBlock
        image_size,  # Input image size. Used to calculate where to inject attention layers in UNet
        channels=3,  # Input channel number
        text_embed_dim=512,  # Dimension of conditioned text embedding. Different text encoders and different model versions have different values
        num_res_blocks=3,  # Number of ResBlock in each level of UNet
        channel_mult=[1, 2, 3, 4],  # Used with embed_dim to calculate the number of channels for each level of UNet
        num_attn_heads=4,  # The number of heads in the attention layer
        per_head_channels=64,  # The number of channels per attention head
        cond_dim=512,  # Dimension of Conditioning projections
        attention_type='fused',  # Type of attention layer
        feature_pooling_type='attention',  # Type of pooling
        learned_sinu_pos_emb_dim=16,  # Dimension of learned time positional embedding. 0 for unlearned timestep embeddings.
        attention_resolutions=[8, 16, 32],  # List of resolutions to inject attention layers
        dropout=False,  # The rate of dropout
        use_null_token=False,  # Whether to create a learned null token for attention
        init_conv_kernel_size=3,  # Initial Conv kernel size. imagen_pytorch uses 7
        gradient_checkpointing=False,  # Whether to use gradient checkpointing
        scale_shift_norm=True,  # Whether to use scale shift norm
        stable_attention=True,  # Whether to use numerically-stable attention calculation
        flash_attention=False,  # Whether to use flash attention calculation
        resblock_updown=False,  # Whether to use ResBlock or Downsample/Upsample
        resample_with_conv=True,  # When resblock_updown=False, whether to use conv in addition to Pooling&ConvTranspose
        low_res_cond=False,
        noise_cond_aug=False,
    ):
        super().__init__()

        # Attention Class
        if attention_type == 'stacked':
            attention_fn = StackedCrossAttentionBlock
        elif attention_type == 'fused':
            attention_fn = FusedCrossAttentionBlock
        else:
            raise ValueError('Attention {} not defined'.format(attention_type))

        # Time embedding for log(snr) noise from continous version
        time_embed_dim = embed_dim * 4
        assert learned_sinu_pos_emb_dim >= 0
        if learned_sinu_pos_emb_dim > 0:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
            self.time_embed = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(sinu_pos_emb_input_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            # Unlearned Time Embedding
            sinu_pos_emb = UnLearnedSinusoidalPosEmb(embed_dim)
            self.time_embed = nn.Sequential(
                sinu_pos_emb, linear(embed_dim, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)
            )

        # Pooling
        assert feature_pooling_type == 'attention' or feature_pooling_type == 'mean'
        self.feature_pooling_type = feature_pooling_type
        if feature_pooling_type == 'attention':
            self.attention_pooling = nn.Sequential(
                SelfAttentionPooling(input_dim=text_embed_dim),
                nn.LayerNorm(text_embed_dim),
                nn.Linear(text_embed_dim, cond_dim),
            )

        # Context Projections
        self.text_to_cond = linear(text_embed_dim, cond_dim)
        self.to_text_non_attn_cond = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Register for Null Token
        if use_null_token:
            self.null_text_embedding = nn.Parameter(torch.randn(1, 1, cond_dim, dtype=self.text_to_cond.weight.dtype))
        self.use_null_token = use_null_token

        # Converting attention resolutions to downsampling factor
        attention_ds = []
        attention_resolutions = sorted(attention_resolutions)
        self.image_size = image_size
        for res in attention_resolutions:
            attention_ds.append(image_size // int(res))

        self.low_res_cond = low_res_cond
        # Low res noise conditioning augmentation
        self.noise_cond_aug = noise_cond_aug
        if self.noise_cond_aug:
            assert (
                self.low_res_cond
            ), 'noise conditioning augmentation should only be enabled when training with low-res cond'
            if learned_sinu_pos_emb_dim > 0:
                lowres_sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
                lowres_sinu_pos_emb_dim = learned_sinu_pos_emb_dim + 1
            else:
                lowres_sinu_pos_emb = UnLearnedSinusoidalPosEmb(embed_dim)
                lowres_sinu_pos_emb_dim = embed_dim
            self.lowres_time_embed = nn.Sequential(
                lowres_sinu_pos_emb,
                nn.Linear(lowres_sinu_pos_emb_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Initial Convolution
        in_channels = 2 * channels if low_res_cond else channels
        init_dim = embed_dim * channel_mult[0]
        self.init_conv = ConditionalSequential(
            nn.Conv2d(in_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)
        )

        if isinstance(num_res_blocks, int):
            res_blocks_list = [num_res_blocks] * len(channel_mult)
        else:
            res_blocks_list = num_res_blocks
        # UNet Init
        # Downsampling Layers
        # We use Conv2D for UNet
        CONV_DIM = 2
        ch = init_dim
        ds = 1
        self.input_blocks = nn.ModuleList([self.init_conv])
        num_input_block_channels = [ch]
        for level, mult in enumerate(channel_mult):
            num_res_blocks = res_blocks_list[level]
            for _ in range(num_res_blocks):
                out_channels = mult * embed_dim
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=out_channels,
                        dims=CONV_DIM,
                        use_checkpoint=gradient_checkpointing,
                        use_scale_shift_norm=scale_shift_norm,
                        learnable_upsampling=True,
                    )
                ]
                ch = out_channels
                if ds in attention_ds:
                    layers.append(
                        attention_fn(
                            channels=ch,
                            num_heads=num_attn_heads,
                            num_head_channels=per_head_channels,
                            use_checkpoint=gradient_checkpointing,
                            stable_attention=stable_attention,
                            flash_attention=flash_attention,
                            context_dim=cond_dim,
                        )
                    )
                self.input_blocks.append(ConditionalSequential(*layers))
                num_input_block_channels.append(ch)
            is_last_level = level == len(channel_mult) - 1
            if not is_last_level:
                # DownSampling
                self.input_blocks.append(
                    ConditionalSequential(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=ch,
                            dims=CONV_DIM,
                            use_checkpoint=gradient_checkpointing,
                            use_scale_shift_norm=scale_shift_norm,
                            down=True,
                            learnable_upsampling=True,
                        )
                        if resblock_updown
                        else Downsample(channels=ch, use_conv=resample_with_conv, dims=CONV_DIM, out_channels=ch,)
                    )
                )
                num_input_block_channels.append(ch)
                ds *= 2

        # Middle Layers
        self.middle_block = ConditionalSequential(
            # Mid Block 1
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=CONV_DIM,
                use_checkpoint=gradient_checkpointing,
                use_scale_shift_norm=scale_shift_norm,
                learnable_upsampling=True,
            ),
            # Attention Layer
            attention_fn(
                channels=ch,
                num_heads=num_attn_heads,
                num_head_channels=per_head_channels,
                use_checkpoint=gradient_checkpointing,
                stable_attention=stable_attention,
                flash_attention=flash_attention,
                context_dim=cond_dim,
            ),
            # Mid Block 2
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=CONV_DIM,
                use_checkpoint=gradient_checkpointing,
                use_scale_shift_norm=scale_shift_norm,
                learnable_upsampling=True,
            ),
        )

        # Upsampling Layers
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            num_res_blocks = res_blocks_list[level]
            for i in range(num_res_blocks + 1):
                ich = num_input_block_channels.pop()
                out_channels = embed_dim * mult
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=out_channels,
                        dims=CONV_DIM,
                        use_checkpoint=gradient_checkpointing,
                        use_scale_shift_norm=scale_shift_norm,
                        learnable_upsampling=True,
                    )
                ]
                ch = out_channels

                if ds in attention_ds:
                    layers.append(
                        attention_fn(
                            channels=ch,
                            num_heads=-1,  # TODO
                            num_head_channels=per_head_channels,
                            use_checkpoint=gradient_checkpointing,
                            stable_attention=stable_attention,
                            flash_attention=flash_attention,
                            context_dim=cond_dim,
                        )
                    )
                is_last_block = i == num_res_blocks
                if level and is_last_block:
                    layers.append(
                        ResBlock(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=ch,
                            dims=CONV_DIM,
                            use_checkpoint=gradient_checkpointing,
                            use_scale_shift_norm=scale_shift_norm,
                            up=True,
                            learnable_upsampling=True,
                        )
                        if resblock_updown
                        else Upsample(channels=ch, use_conv=resample_with_conv, dims=CONV_DIM, out_channels=ch)
                    )
                    ds //= 2
                self.output_blocks.append(ConditionalSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(init_dim, channels, init_conv_kernel_size, padding=init_conv_kernel_size // 2)),
        )

    def forward(
        self, x, time, text_embed=None, text_mask=None, x_low_res=None, time_low_res=None,
    ):
        if self.low_res_cond:
            assert x_low_res is not None, 'x_low_res cannot be None'
        else:
            assert x_low_res is None, 'x_low_res cannot be presented'
        if self.noise_cond_aug:
            assert time_low_res is not None, 'time_low_res cannot be None when training with noise conditioning aug'
        else:
            assert time_low_res is None, 'time_low_res cannot be presented'
        # Concatenating low resolution images
        if x_low_res is not None:
            if x_low_res.shape != x.shape:
                # Upscale if not done in the trainer
                _, _, new_height, new_width = x.shape
                x_low_res = F.interpolate(x_low_res, (new_height, new_width), mode="bicubic")
            x = torch.cat([x, x_low_res], dim=1)
        batch_size, device = x.shape[0], x.device

        if x.dtype != time.dtype or time.dtype != text_embed.dtype:
            dtype = text_embed.dtype
            x = x.to(dtype=dtype)
            time = time.to(dtype=dtype)
            if x_low_res is not None:
                x_low_res = x_low_res.to(dtype=dtype)
            if time_low_res is not None:
                time_low_res = time_low_res.to(dtype=dtype)
        # Time Conditioning
        t = self.time_embed(time)
        # Add lowres time conditioning
        if self.noise_cond_aug:
            lowres_t = self.lowres_time_embed(time_low_res)
            t += lowres_t
        # Text Conditioning
        text_cond = self.text_to_cond(text_embed)

        # Context Embedding
        # TODO We may want to concat time token here
        if self.use_null_token:
            # Null Context (Helpful when text_embed is drop)
            null_context = self.null_text_embedding.repeat(batch_size, 1, 1)
            context_emb = torch.cat([text_cond, null_context], dim=1)
            context_mask = torch.cat([text_mask, torch.ones(batch_size, 1).to(device)], dim=1)
        else:
            context_emb = text_cond
            context_mask = text_mask

        # Add pooled text embeddings to the diffusion timestep
        # TODO We may only want to calculated the pooled feature based on text token length
        if self.feature_pooling_type == 'mean':
            pooled_text_cond = text_cond.mean(dim=-2)
        elif self.feature_pooling_type == 'attention':
            pooled_text_cond = self.attention_pooling(text_embed)
        text_hiddens = self.to_text_non_attn_cond(pooled_text_cond)
        t += text_hiddens

        h = x
        hs = []
        # UNet Forward
        for module in self.input_blocks:
            h = module(h, t, context_emb, context_mask)
            hs.append(h)
        h = self.middle_block(h, t, context_emb, context_mask)
        for module in self.output_blocks:
            h_prev = hs.pop()
            h = torch.cat([h, h_prev], dim=1)
            h = module(h, t, context_emb, context_mask)
        return self.out(h)

    def forward_with_cond_scale(self, *args, text_embed=None, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, text_embed=text_embed, **kwargs)
        if cond_scale == 1.0:
            return logits
        null_logits = self.forward(*args, text_embed=torch.zeros_like(text_embed), **kwargs)
        return null_logits + (logits - null_logits) * cond_scale


class EfficientUNetModel(nn.Module):
    """
    The full Efficient UNet model with attention and timestep embedding used for Imagen SR model.

    :param embed_dim: Dimension of embeddings. Also used to calculate the number of channels in ResBlock.
    :param image_size: Input image size. Used to calculate where to inject attention layers in UNet.
    :param channels: Input channel number, defaults to 3.
    :param text_embed_dim: Dimension of conditioned text embedding. Different text encoders and different model versions have different values, defaults to 512
    :param channel_mult: Used with embed_dim to calculate the number of channels for each level of UNet, defaults to [1, 1, 2, 4, 8].
    :param num_attn_heads: The number of heads in the attention layer, defaults to 8.
    :param per_head_channels: The number of channels per attention head, defaults to 64.
    :param attention_type: Type of attention layer, defaults to 'fused'.
    :param atnn_enabled_at: Whether to enable attention at each level, defaults to [0, 0, 0, 0, 1].
    :param feature_pooling_type: Type of pooling, defaults to 'attention'.
    :param stride: Stride in ResBlock, defaults to 2.
    :param num_resblocks: Used with num_res_blocks to calculate the number of residual blocks at each level of Efficient-UNet. Defaults to [1, 2, 4, 8, 8].
    :param learned_sinu_pos_emb_dim: Dimension of learned time positional embedding. 0 for unlearned timestep embeddings. Defaults to 16
    :param use_null_token: Whether to create a learned null token for attention, defaults to False.
    :param init_conv_kernel_size: Initial Conv kernel size, defaults to 3.
    :param gradient_checkpointing: Whether to use gradient checkpointing, defaults to False.
    :param scale_shift_norm: Whether to use scale shift norm, defaults to False.
    :param stable_attention: Whether to use numerically-stable attention calculation, defaults to True.
    :param flash_attention: Whether to use flash attention calculation, defaults to False.
    :param skip_connection_scaling: Whether to use 1/sqrt(2) scaling for ResBlock skip connection, defaults to False.
    :param noise_cond_aug: Whether to add noise conditioned augmentation with low-resolution input. Defaults to False.
    """

    def __init__(
        self,
        embed_dim,
        image_size,
        channels=3,
        text_embed_dim=512,  # Dimension of conditioned text embedding. Different text encoders and different model versions have different values
        channel_mult=[
            1,
            1,
            2,
            4,
            8,
        ],  # Used with embed_dim to calculate the number of channels for each level of Efficient-UNet
        num_attn_heads=8,  # The number of heads in the attention layer
        per_head_channels=64,  # The number of channels per attention head
        attention_type='fused',  # Type of attention layer
        atnn_enabled_at=[0, 0, 0, 0, 1],  # Whether to enable attention at each level
        feature_pooling_type='attention',  # Type of pooling
        stride=2,  # Stride in ResBlock
        num_resblocks=[
            1,
            2,
            4,
            8,
            8,
        ],  # Used with num_res_blocks to calculate the number of residual blocks at each level of Efficient-UNet
        learned_sinu_pos_emb_dim=16,  # Dimension of learned time positional embedding. 0 for unlearned timestep embeddings.
        use_null_token=False,  # Whether to create a learned null token for attention
        init_conv_kernel_size=3,  # Initial Conv kernel size. imagen_pytorch uses 7
        gradient_checkpointing=False,  # Whether to use gradient checkpointing
        scale_shift_norm=True,  # Whether to use scale shift norm
        stable_attention=True,  # Whether to use numerically-stable attention calculation
        flash_attention=False,  # Whether to use flash attention calculation
        skip_connection_scaling=False,  # Whether to use 1/sqrt(2) scaling for ResBlock skip connection
        noise_cond_aug=False,
    ):

        super().__init__()

        self.n_levels = len(channel_mult)
        self.image_size = image_size
        # Time embedding for log(snr) noise from continous version
        time_embed_dim = embed_dim * 4
        assert learned_sinu_pos_emb_dim >= 0
        if learned_sinu_pos_emb_dim > 0:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
            sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
            self.time_embed = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(sinu_pos_emb_input_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            # Unlearned Time Embedding
            sinu_pos_emb = UnLearnedSinusoidalPosEmb(embed_dim)
            self.time_embed = nn.Sequential(
                sinu_pos_emb, linear(embed_dim, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)
            )

        self.noise_cond_aug = noise_cond_aug
        if self.noise_cond_aug:
            if learned_sinu_pos_emb_dim > 0:
                lowres_sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
                lowres_sinu_pos_emb_dim = learned_sinu_pos_emb_dim + 1
            else:
                lowres_sinu_pos_emb = UnLearnedSinusoidalPosEmb(embed_dim)
                lowres_sinu_pos_emb_dim = embed_dim
            self.lowres_time_embed = nn.Sequential(
                lowres_sinu_pos_emb,
                nn.Linear(lowres_sinu_pos_emb_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        cond_dim = text_embed_dim  # time_embed_dim
        # Pooling
        assert feature_pooling_type == 'attention' or feature_pooling_type == 'mean'
        self.feature_pooling_type = feature_pooling_type
        if feature_pooling_type == 'attention':
            self.attention_pooling = nn.Sequential(
                SelfAttentionPooling(input_dim=text_embed_dim),
                nn.LayerNorm(text_embed_dim),
                nn.Linear(text_embed_dim, cond_dim),
            )

        # Context Projections
        self.text_to_cond = linear(text_embed_dim, cond_dim)
        self.to_text_non_attn_cond = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # Register for Null Token
        if use_null_token:
            self.null_text_embedding = nn.Parameter(torch.randn(1, 1, cond_dim, dtype=self.text_to_cond.weight.dtype))
        self.use_null_token = use_null_token

        # Initial Convolution
        # Multiply in_channels by 2 because we concatenate with low res inputs.
        in_channels = channels * 2
        init_dim = embed_dim * channel_mult[0]
        self.init_conv = nn.Conv2d(in_channels, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)
        # Efficient-UNet Init
        self.DBlocks = nn.ModuleDict()
        self.UBlocks = nn.ModuleDict()
        ch = init_dim
        for level, mult in enumerate(channel_mult):
            # Different level has different num of res blocks
            num_resblock = num_resblocks[level]
            # Only perform upsample/downsample if it is not the last (deepest) level
            is_last_level = level == len(channel_mult) - 1
            level_attention_type = attention_type if atnn_enabled_at[level] else None

            level_key = str(level)  # TODO Change to more meaningful naming
            self.DBlocks[level_key] = DBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=int(mult * embed_dim),
                use_scale_shift_norm=scale_shift_norm,
                conv_down=not is_last_level,
                stride=stride,
                num_resblocks=num_resblock,
                attention_type=level_attention_type,
                text_embed_dim=cond_dim,
                num_heads=num_attn_heads,
                num_head_channels=per_head_channels,
                use_checkpoint=gradient_checkpointing,
                stable_attention=stable_attention,
                flash_attention=flash_attention,
                skip_connection_scaling=skip_connection_scaling,
            )
            self.UBlocks[level_key] = UBlock(
                channels=int(mult * embed_dim),
                emb_channels=time_embed_dim,
                out_channels=ch,
                use_scale_shift_norm=scale_shift_norm,
                conv_up=not is_last_level,
                stride=stride,
                num_resblocks=num_resblock,
                attention_type=level_attention_type,
                text_embed_dim=cond_dim,
                num_heads=num_attn_heads,
                num_head_channels=per_head_channels,
                use_checkpoint=gradient_checkpointing,
                stable_attention=stable_attention,
                flash_attention=flash_attention,
                skip_connection_scaling=skip_connection_scaling,
            )
            ch = int(mult * embed_dim)
        self.out = nn.Conv2d(channel_mult[0] * embed_dim, channels, 1)

    def forward(
        self, x, time, text_embed, text_mask, x_low_res, time_low_res=None,
    ):
        if self.noise_cond_aug:
            assert time_low_res is not None, 'time_low_res cannot be None when training with noise conditioning aug'
        else:
            assert time_low_res is None, 'time_low_res cannot be presented'

        if x.dtype != time.dtype or time.dtype != text_embed.dtype:
            dtype = text_embed.dtype
            x = x.to(dtype=dtype)
            time = time.to(dtype=dtype)
            if x_low_res is not None:
                x_low_res = x_low_res.to(dtype=dtype)
            if time_low_res is not None:
                time_low_res = time_low_res.to(dtype=dtype)

        batch_size, device = x.shape[0], x.device
        # Time Conditioning
        t = self.time_embed(time)
        # Text Conditioning
        text_cond = self.text_to_cond(text_embed)
        # Concatenating low resolution images
        if x_low_res.shape != x.shape:
            # Upscale if not done in the trainer
            _, _, new_height, new_width = x.shape
            x_low_res = F.interpolate(x_low_res, (new_height, new_width), mode="bicubic")
        x = torch.cat([x, x_low_res], dim=1)

        # Add lowres time conditioning
        if self.noise_cond_aug:
            lowres_t = self.lowres_time_embed(time_low_res)
            t += lowres_t
        # Context Embedding
        # TODO We may want to concat time token here
        if self.use_null_token:
            # Null Context (Helpful when text_embed is drop)
            null_context = self.null_text_embedding.repeat(batch_size, 1, 1)
            context_emb = torch.cat([text_cond, null_context], dim=1)
            context_mask = torch.cat([text_mask, torch.ones(batch_size, 1).to(device)], dim=1)
        else:
            context_emb = text_cond
            context_mask = text_mask

        # Add pooled text embeddings to the diffusion timestep
        # TODO We may only want to calculated the pooled feature based on text token length
        if self.feature_pooling_type == 'mean':
            pooled_text_cond = text_cond.mean(dim=-2)
        elif self.feature_pooling_type == 'attention':
            pooled_text_cond = self.attention_pooling(text_embed)
        text_hiddens = self.to_text_non_attn_cond(pooled_text_cond)
        t += text_hiddens

        # UNet forward
        x = self.init_conv(x)
        feats = dict()
        for level in range(self.n_levels):
            level_key = str(level)
            x = self.DBlocks[level_key](x, t, context_emb, context_mask)
            # Save feats for UBlocks
            if level < self.n_levels - 1:
                feats[level_key] = x
        for level in range(self.n_levels - 1, -1, -1):
            level_key = str(level)
            if level < self.n_levels - 1:
                x = x + feats[level_key]
            x = self.UBlocks[level_key](x, t, context_emb, context_mask)
        return self.out(x)

    def forward_with_cond_scale(self, *args, text_embed=None, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, text_embed=text_embed, **kwargs)
        if cond_scale == 1.0:
            return logits
        null_logits = self.forward(*args, text_embed=torch.zeros_like(text_embed), **kwargs)
        return null_logits + (logits - null_logits) * cond_scale


if __name__ == '__main__':
    model = UNetModel(embed_dim=512, image_size=64,)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    image_batch = torch.rand(4, 3, 64, 64)
    text_cond = torch.rand(4, 88, 512)
    text_mask = torch.ones(4, 88)
    time = torch.ones(4)

    output = model(image_batch, time, text_cond, text_mask,)

    print(output.shape)

    model_sr = EfficientUNetModel(embed_dim=128, image_size=256)
    pytorch_total_params = sum(p.numel() for p in model_sr.parameters())
    print(pytorch_total_params)
    output = model_sr(
        torch.randn(4, 3, 256, 256),
        torch.randn(4, 3, 256, 256),
        torch.ones(4),
        torch.randn(4, 88, 512),
        torch.ones(4, 88),
    )
    print(output.shape)
