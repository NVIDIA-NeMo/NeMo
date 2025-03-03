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

from dataclasses import dataclass


@dataclass
class HyenaConfig:
    """Configuration object for Hyena model and operators"""

    tie_projection_weights: bool = False
    """
    Tie projection weights between QKV for attn and hyena (will repeat output 3 times).
    """
    #
    to_upper: str = "normalized_weighted"
    """
    "upper"
    "weighted"
    Whether to convert all text to uppercase.
    """
    #
    lowercase_loss_reweighting: float = 0.1
    # """
    # If to_upper == "weighted"
    # Weight to apply to lowercase tokens in the loss function, 1.0 is no reweighting.
    # """

    use_flashfft: bool = False
    """
    Use flashfftconv instead of torch fft kernel (requires installation of flashfftconv)for hyena
    """

    use_cgcg: bool = False
    """
    Use cgcg (chunked gate-conv-gate) kernel for hyena
    """

    use_cgcg_short: bool = False
    """
    Use cgcg (chunked gate-conv-gate) kernel for hyena short conv
    """

    use_cgcg_mlp: bool = False
    """
    Use cgcg (chunked gate-conv-gate) kernel for hyena mlp
    """

    cgcg_dtype: str = "bfloat16"
    """
    dtype to use within cgcg kernel
    """
    #
    # cgcg_fwd_autotune: bool = False
    # """
    # Whether to autotune cgcg fwd kernel
    #
    # @jeromeku: Note autotuning fwd kernel is unstable,
    # use pre-tuned config for now.
    # """

    cgcg_medium_fwd_kernel_config_chunk_size: int = 128
    """
    cgcg fwd medium conv kernel config chunk size
    """
    cgcg_medium_fwd_kernel_config_block_d: int = 128
    """
    cgcg fwd medium conv kernel config block d tile size
    """

    cgcg_medium_fwd_kernel_config_threadblock_swizzle: str = "row"
    """
    cgcg fwd medium conv kernel config threadblock swizzle type
    """
    cgcg_medium_fwd_kernel_config_chunk_tiles_per_program: int = 3
    """
    cgcg fwd medium conv kernel config chunk tiles per program
    """

    cgcg_medium_fwd_kernel_config_num_warps: int = 4
    """
    cgcg fwd short conv kernel config num warps
    """

    cgcg_medium_fwd_kernel_config_num_stages: int = 3
    """
    cgcg fwd medium conv kernel config num mma pipeline stages
    """

    cgcg_short_fwd_kernel_config_chunk_size: int = 128
    """
    cgcg fwd short conv kernel config chunk size
    """
    cgcg_short_fwd_kernel_config_block_d: int = 128
    """
    cgcg fwd short conv kernel config block d tile size
    """

    cgcg_short_fwd_kernel_config_threadblock_swizzle: str = "row"
    """
    cgcg fwd short conv kernel config threadblock swizzle type
    """
    cgcg_short_fwd_kernel_config_chunk_tiles_per_program: int = 1
    """
    cgcg fwd short conv kernel config chunk tiles per program
    """

    cgcg_short_fwd_kernel_config_num_warps: int = 4
    """
    cgcg fwd short conv kernel config num warps
    """

    cgcg_short_fwd_kernel_config_num_stages: int = 1
    """
    cgcg fwd short conv kernel config num mma pipeline stages
    """

    cgcg_bwd_autotune: bool = True
    """
    Whether to autotune cgcg bwd kernel
    """

    cgcg_fused_bwd: bool = True
    """
    Whether to use fused cgcg bwd kernel
    """

    cgcg_bwd_kernel_config_pre_conv_block_x: int = 128
    """
    cgcg bwd pre_conv kernel config block x tile size
    """

    cgcg_bwd_kernel_config_pre_conv_block_y: int = 128
    """
    cgcg bwd pre_conv kernel config block y tile size
    """

    cgcg_bwd_kernel_config_pre_conv_num_warps: int = 8
    """
    cgcg bwd pre_conv kernel config num warps
    """

    cgcg_bwd_kernel_config_post_conv_block_x: int = 32
    """
    cgcg bwd post conv kernel config block x tile size
    """

    cgcg_bwd_kernel_config_post_conv_block_y: int = 128
    """
    cgcg bwd post conv kernel config block y tile size
    """

    cgcg_bwd_kernel_config_post_conv_num_warps: int = 4
    """
    cgcg bwd post conv kernel config num warps
    """

    short_conv_L: int = 3
    """
    For Hyena models, length of the short convolution.
    """

    use_hyena_filter: bool = False
    """
    Whether to use the Hyena filter.
    """

    normalize_hyena_filters: bool = False

    conv_proj_bias: bool = True  # Maybe this should be false
    """
    Use bias in the short conv1D, needed for model parallel for the short conv.
    """

    use_fast_heads: bool = False
    """
    Use external fast heads in Hyena mixer (reduce BEFORE fftconv)
    """

    use_slow_heads: bool = False
    """
    Use external outer-product heads in Hyena.
    """

    use_long_conv1d: bool = False

    num_groups_hyena: int = None
    """
    Determines number of unique filters to have, for the hyena long filter.
    """

    num_groups_hyena_medium: int = None
    """
    Determines number of unique filters to have, for the hyena medium filter.
    """

    num_groups_hyena_short: int = None
    """
    Determines number of unique filters to have, for the hyena short filter.
    """

    num_groups_hyena_mlp: int = None  # TODO: Possibly remove, only used if is_mlp is True
    """
    Determines number of unique filters to have, for the hyena mlp (filter).
    """

    use_depthwise_short_conv_grouping: bool = True
    """
    Whether to use depthwise convolution grouping for short conv and hyena mlp filters.
    """

    hyena_filter_cls: str = "implicit_modal"
    """
    """

    hyena_width_expansion: float = 1.0
    """
    Factor to expand the projections width within hyena layers.
    """

    hyena_medium_filter_cls: str = 'explicit_single_decay'
    """
    For medium hyena filters specifically, None defaults ot same as hyena_filter_cls (long filters).
    """

    hyena_filter_r_max: float = 0.99  # TODO: Possibly remove, only used in ParallelComplexModalFilter

    hyena_filter_r_min: float = 0.5  # TODO: Possibly remove, only used in ParallelComplexModalFilter

    hyena_filter_emb_dim: int = 33  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    hyena_filter_fast_decay: float = 0.3  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    hyena_filter_slow_decay: float = 1.2  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    hyena_filter_order: int = 16

    hyena_filter_num_inner_mlps: int = 2  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    hyena_filter_w: int = 14  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    hyena_filter_wd: float = 0.0  # TODO: Where to override WD value for filters?

    hyena_filter_omega_0: float = 1  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    hyena_pos_emb: str = "fourier_fixed"  # TODO: Possibly remove, only used in ParallelImplicitFreeformFilter

    explicit_filter_decay_preset: str = "weak"

    modal_residue_factors: int = 3  # TODO: Possibly remove, only used in ImplicitRealModelFilter

    modal_pole_factors: int = 3  # TODO: Possibly remove, only used in ImplicitRealModelFilter

    modal_gamma_min: float = 0.01

    modal_gamma_max: float = 0.1

    use_custom_hyena_short_kernel: bool = False
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    use_custom_hyena_mlp_kernel: bool = False  # TODO: Possibly remove - only relevant if is_mlp is True
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    bidirectional: bool = False
    """
    A bidirectional version of hyena fftconv
    """

    hyena_short_conv_len: int = 7
    """
    Length of the hyena short conv layer, if using
    """

    fast_conv_proj: bool = True
    """
    Use a custom causal conv layer for the hyena projection convs.
    """

    hyena_medium_conv_len: int = 128
    """
    Length of the medium hyena filter.
    """

    fast_conv_mixer: bool = False
    """
    Use a custom causal conv layer for the hyena short conv layer.
    """

    hyena_mlp_len: int = 7  # TODO: Possibly remove, only used if is_mlp is True
    """
    Length of filter used inside the hyena mlp layer. Defaults to hyena_short_conv_len if not provided.
    """

    fast_hyena_mlp_conv: bool = False  # TODO: Possibly remove, only used if is_mlp is True
    """
    Use a custom causal conv layer for the hyena MLP layer.
    """

    hyena_mlp_expansion_factor: float = 1.0  # TODO: Possibly remove, only used if is_mlp is True
    """
    Factor to expand the projections width within hyena MLP layers only.
    """

    hyena_mlp_pregate: bool = True  # TODO: Possibly remove, only used if is_mlp is True
    """
    Use a pre-gate in the hyena MLP layer.
    """

    hyena_mlp_postgate: bool = True  # TODO: Possibly remove, only used if is_mlp is True
    """
    Use a post-gate in the hyena MLP layer.
    """

    hyena_short_conv_pregate: bool = True
    """
    Use a pre-gate in the hyena short conv layer.
    """

    hyena_short_conv_postgate: bool = True
    """
    Use a post-gate in the hyena short conv layer.
    """

    proj_groups: int = 1

    grouped_attention: bool = False

    # mlp_type: str = "regular"  # TODO: In Savanna setting this to 'short_hyena' uses hyena for MLP (is_mlp == True)
    # """
    # Types:
    #     regular: Megatron implementation
    #     llama: LLaMA MLP (SiLU-gated MLP)
    #     short_hyena
    #     identity
    # """
    #
    # make_gated_mlp_multiple_of: int = 16  # TODO: Use this or just have user calculate ffn_size themselves?
    # """
    # Set the ff_dim to be a multiple of this value for llama mlp. Useful for sharding / using model parallel properly.
    # """
