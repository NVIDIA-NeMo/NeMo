# *****************************************************************************
# Copyright (C) 2024 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *****************************************************************************
"""The default image and video tokenizer configs."""

from nemo.collections.common.video_tokenizers.modules import (
    ContinuousFormulation,
    Decoder3DType,
    DecoderType,
    DiscreteQuantizer,
    Encoder3DType,
    EncoderType,
)

continuous_image = dict(
    # The attention resolution for res blocks.
    attn_resolutions=[32],
    # The base number of channels.
    channels=128,
    # The channel multipler for each resolution.
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    # The spatial compression ratio.
    spatial_compression=16,
    # The number of layers in each res block.
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    # The output latent dimension (channels).
    latent_channels=16,
    # The encoder output channels just before sampling.
    # Which is also the decoder's input channels.
    z_channels=16,
    # A factor over the z_channels, to get the total channels the encoder should output.
    # For a VAE for instance, we want to output the mean and variance, so we need 2 * z_channels.
    z_factor=1,
    name="CI",
    # What formulation to use, either "AE" or "VAE".
    # Chose VAE here, since the pre-trained ckpt were of a VAE formulation.
    formulation=ContinuousFormulation.AE.name,
    # Specify type of encoder ["Default", "LiteVAE"]
    encoder=EncoderType.Default.name,
    # Specify type of decoder ["Default"]
    decoder=DecoderType.Default.name,
)

discrete_image = dict(
    # The attention resolution for res blocks.
    attn_resolutions=[32],
    # The base number of channels.
    channels=128,
    # The channel multipler for each resolution.
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    # The spatial compression ratio.
    spatial_compression=16,
    # The number of layers in each res block.
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    # The encoder output channels just before sampling.
    z_channels=256,
    # A factor over the z_channels, to get the total channels the encoder should output.
    # for discrete tokenization, often we directly use the vector, so z_factor=1.
    z_factor=1,
    # The quantizer of choice, VQ, LFQ, FSQ, or ResFSQ.
    quantizer=DiscreteQuantizer.FSQ.name,
    # The embedding dimension post-quantization, which is also the input channels of the decoder.
    # Which is also the output
    embedding_dim=6,
    # The number of levels to use for fine-scalar quantization.
    levels=[8, 8, 8, 5, 5, 5],
    # The number of quantizers to use for residual fine-scalar quantization.
    num_quantizers=4,
    name="DI",
    # Specify type of encoder ["Default", "LiteVAE"]
    encoder=EncoderType.Default.name,
    # Specify type of decoder ["Default"]
    decoder=DecoderType.Default.name,
)

continuous_video = dict(
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    latent_channels=16,
    z_channels=16,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    spatial_compression=8,
    temporal_compression=8,
    formulation=ContinuousFormulation.AE.name,
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="CausalCV",
)

discrete_video = dict(
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    z_channels=16,
    z_factor=1,
    num_groups=1,
    legacy_mode=False,
    spatial_compression=16,
    temporal_compression=8,
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=[8, 8, 8, 5, 5, 5],
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="CausalDV",
)
