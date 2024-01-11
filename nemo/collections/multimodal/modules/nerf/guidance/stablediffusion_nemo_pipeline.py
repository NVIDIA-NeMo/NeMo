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
import os
import tempfile

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import LatentDiffusion
from nemo.collections.multimodal.modules.nerf.guidance.txt2img_guidance_base import Txt2ImgGuidanceBase
from nemo.collections.multimodal.modules.stable_diffusion.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector


class StableDiffusion(Txt2ImgGuidanceBase):
    def __init__(
        self, checkpoint, sampler_type="DDIM", t_range=[0.02, 0.98], precision="16", device=torch.device('cuda')
    ):
        super().__init__()

        self.device = device
        self.checkpoint = checkpoint
        self.sampler_type = sampler_type

        cfg, state_dict = self.load_config_and_state_from_nemo(checkpoint)

        cfg.precision = precision
        cfg.ckpt_path = None
        cfg.unet_config.from_pretrained = None
        cfg.first_stage_config.from_pretrained = None

        self.model = LatentDiffusion(cfg).to(device)

        sd_state_dict = {}
        # Remove Megatron wrapper and inductor
        for key, value in state_dict.items():
            key = key[6:]
            sd_state_dict[key] = value
        self.model.load_state_dict(sd_state_dict)
        self.first_stage_model = self.model.first_stage_model
        self.text_encoder = self.model.cond_stage_model.encode

        self.num_train_timesteps = self.model.num_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.model.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        return self.text_encoder(prompt)

    @torch.autocast(device_type="cuda")
    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(x_start=latents, t=t, noise=noise)
            latent_model_input = torch.cat([latents_noisy] * 2)
            td = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(latent_model_input, td, text_embeddings)

            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        grad = w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        return loss

    def image_encoder(self, x):
        h = self.first_stage_model.encoder(x)
        moments = self.first_stage_model.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.image_encoder(imgs)
        latents = (
            posterior.sample() * self.image_encoder.config.scaling_factor
        )  # self.vae.config.scaling_factor==0.18215

        return latents

    def load_config_and_state_from_nemo(self, nemo_path):
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        save_restore_connector = NLPSaveRestoreConnector()
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)

                # Change current working directory to
                os.chdir(tmpdir)
                config_yaml = os.path.join(tmpdir, save_restore_connector.model_config_yaml)
                cfg = OmegaConf.load(config_yaml)

                model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                state_dict = save_restore_connector._load_state_dict_from_disk(
                    model_weights, map_location=map_location
                )
            finally:
                os.chdir(cwd)

        return cfg, state_dict
