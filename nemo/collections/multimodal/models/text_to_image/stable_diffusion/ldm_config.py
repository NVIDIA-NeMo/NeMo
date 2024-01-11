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
from dataclasses import dataclass, field
from typing import Any, List, Optional

from nemo.core.config import modelPT as model_cfg


@dataclass
class LDMUnetConfig:
    cls: Optional[str] = 'nemo.collections.multimodal.modules.diffusionmodules.openaimodel.UNetModel'
    image_size: Optional[int] = 32  # unused
    in_channels: Optional[int] = 4
    out_channels: Optional[int] = 4
    model_channels: Optional[int] = 320
    attention_resolutions: Optional[List[int]] = field(default_factory=lambda: [4, 2, 1])
    num_res_blocks: Optional[int] = 2
    channel_mult: Optional[List[int]] = field(default_factory=lambda: [1, 2, 4, 4])
    num_heads: Optional[int] = 8
    use_spatial_transformer: Optional[bool] = True
    transformer_depth: Optional[int] = 1
    context_dim: Optional[int] = 768
    use_checkpoint: Optional[bool] = True
    legacy: Optional[bool] = False
    use_flash_attention: Optional[bool] = False


@dataclass
class SchedulerConfig:
    cls: Optional[str] = 'nemo.collections.multimodal.parts.lr_scheduler.LambdaLinearScheduler'
    warm_up_steps: Optional[List[int]] = field(default_factory=lambda: [10000])
    cycle_lengths: Optional[List[int]] = field(
        default_factory=lambda: [10000000000000]
    )  # incredibly large number to prevent corner cases
    f_start: Optional[List[float]] = field(default_factory=lambda: [1.0e-6])
    f_max: Optional[List[float]] = field(default_factory=lambda: [1.0])
    f_min: Optional[List[float]] = field(default_factory=lambda: [1.0])


@dataclass
class CLIPEmbedderConfig:
    cls: Optional[str] = 'nemo.collections.multimodal.modules.encoders.modules.FrozenCLIPEmbedder'
    version: Optional[str] = 'openai/clip-vit-large-patch14'
    device: Optional[str] = 'cuda'
    max_length: Optional[int] = 77


@dataclass
class LDMEncoderConfig:
    double_z: Optional[bool] = True
    z_channels: Optional[int] = 4
    resolution: Optional[int] = 256
    in_channels: Optional[int] = 3
    out_ch: Optional[int] = 3
    ch: Optional[int] = 128
    ch_mult: Optional[List[int]] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: Optional[int] = 2
    attn_resolutions: Optional[List[int]] = field(default_factory=lambda: [])
    dropout: Optional[float] = 0.0


@dataclass
class LDMFirstStageConfig:  # Autoencoder
    cls: Optional[str] = 'nemo.collections.multimodal.models.ldm.autoencoder.AutoencoderKL'
    embed_dim: Optional[int] = 4
    monitor: Optional[str] = 'val/rec_loss'
    ddconfig: Optional[LDMEncoderConfig] = LDMEncoderConfig()


@dataclass
class DDPMDiffusionModelConfig(model_cfg.ModelConfig):
    unet_config: Optional[LDMUnetConfig] = LDMUnetConfig()
    timesteps: Optional[int] = 1000
    beta_schedule: Optional[str] = 'linear'
    loss_type: Optional[str] = 'l2'
    ckpt_path: Optional[str] = None
    ignore_keys: Optional[List[str]] = field(default_factory=list)
    load_only_unet: Optional[bool] = False
    monitor: Optional[str] = 'val/loss'
    use_ema: Optional[bool] = True
    first_stage_key: Optional[str] = 'image'
    image_size: Optional[int] = 256
    channels: Optional[int] = 3
    log_every_t: Optional[int] = 100
    clip_denoised: Optional[bool] = True
    linear_start: Optional[float] = 1e-4
    linear_end: Optional[float] = 2e-2
    cosine_s: Optional[float] = 8e-3
    given_betas: Optional[float] = None
    original_elbo_weight: Optional[float] = 0.0
    v_posterior: Optional[
        float
    ] = 0.0  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
    l_simple_weight: Optional[float] = 1.0
    conditioning_key: Optional[str] = None
    parameterization: Optional[str] = 'eps'  # all assuming fixed variance schedules
    scheduler_config: Optional[Any] = None
    use_positional_encodings: Optional[bool] = False
    learn_logvar: Optional[bool] = False
    logvar_init: Optional[float] = 0.0
    learning_rate: Optional[float] = 1.0e-04


@dataclass
class LatentDiffusionModelConfig(DDPMDiffusionModelConfig):
    # Overrite Default values
    linear_start: Optional[float] = 0.00085
    linear_end: Optional[float] = 0.0120
    num_timesteps_cond: Optional[int] = 1
    log_every_t: Optional[int] = 200
    timesteps: Optional[int] = 1000
    first_stage_key: Optional[str] = 'jpg'
    cond_stage_key: Optional[str] = 'txt'
    image_size: Optional[int] = 64
    channels: Optional[int] = 4
    cond_stage_trainable: Optional[bool] = False
    conditioning_key: Optional[str] = 'crossattn'
    monitor: Optional[str] = 'val/loss_simple_ema'
    scale_factor: Optional[float] = 0.18215
    use_ema: Optional[bool] = False  # TODO
    unet_config: Optional[LDMUnetConfig] = LDMUnetConfig()
    first_stage_config: Optional[LDMFirstStageConfig] = LDMFirstStageConfig()
    scheduler_config: Optional[SchedulerConfig] = SchedulerConfig()
    # New attributes in additon to DDPMDiffusionModel
    concat_mode: Optional[bool] = True
    trainable: Optional[bool] = False
    cond_stage_config: Optional[CLIPEmbedderConfig] = CLIPEmbedderConfig()
    cond_stage_forward: Optional[Any] = None
    scale_by_std: Optional[bool] = False
    text_embedding_dropout_rate: Optional[float] = 0
    fused_opt: Optional[bool] = False
    inductor: Optional[bool] = False
    inductor_cudagraphs: Optional[bool] = False
