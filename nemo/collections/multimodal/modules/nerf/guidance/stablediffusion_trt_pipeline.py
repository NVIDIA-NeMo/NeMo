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
import logging
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from polygraphy import cuda
from transformers import CLIPTokenizer

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import LatentDiffusion
from nemo.collections.multimodal.modules.nerf.guidance.txt2img_guidance_base import Txt2ImgGuidanceBase
from nemo.collections.multimodal.modules.nerf.utils.trt_engine import Engine, device_view
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import (
    extract_into_tensor,
    make_beta_schedule,
)
from nemo.collections.multimodal.parts.stable_diffusion.utils import default
from nemo.collections.multimodal.parts.utils import randn_like
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector


class LatentDiffusionWrapper(Txt2ImgGuidanceBase):
    def __init__(self, plan_dir, checkpoint):
        super().__init__()
        with open(os.path.join(plan_dir, "conf.yaml"), "rb") as fp:
            config = OmegaConf.load(fp.name)
        max_batch_size = config.batch_size

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.max_length = config.clip.max_length
        self.rng = torch.Generator(device=torch.cuda.current_device(),)

        self.set_beta_schedule()

        stream = cuda.Stream()

        self.image_encoder = self.load_vae_from_checkpoint(checkpoint)

        self.text_encoder = Engine(os.path.join(plan_dir, "clip.plan"))
        shape_dict = {'tokens': config.clip.tokens, 'logits': config.clip.logits}
        self.text_encoder.set_engine(stream, shape_dict)

        self.unet = Engine(os.path.join(plan_dir, "unet.plan"))
        shape_dict = {
            'x': config.unet.x,
            't': (max_batch_size * 2,),
            'context': config.unet.context,
            'logits': config.unet.logits,
        }
        self.unet.set_engine(stream, shape_dict)

    def set_beta_schedule(self):
        betas = make_beta_schedule("linear", 1000, linear_start=0.00085, linear_end=0.0120, cosine_s=0.008)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        alphas_cumprod = torch.tensor(alphas_cumprod)
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(torch.cuda.current_device())
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())))
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: randn_like(x_start, generator=self.rng))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.image_encoder(imgs)
        latents = posterior.sample() * 0.18215
        return latents

    def clip_encode(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to("cuda", non_blocking=True)
        z = self.text_encoder.infer({"tokens": device_view(tokens.type(torch.int32))})['logits'].clone()
        seq_len = (z.shape[1] + 8 - 1) // 8 * 8
        z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
        return z

    def apply_model(self, x, t, cond, return_ids=False):
        self.conditioning_key = "crossattn"
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            # key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            key = 'c_crossattn'
            cond = {key: cond}
        # UNET TRT
        cc = torch.cat(cond['c_crossattn'], 1)  # needs to be changed I think
        out = self.unet.infer(
            {
                "x": device_view(x.contiguous()),
                "t": device_view(t.type(torch.int32).contiguous()),
                "context": device_view(cc.contiguous()),
            }
        )['logits'].clone()
        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def load_vae_from_checkpoint(self, checkpoint):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg, state_dict = self.load_config_and_state_from_nemo(checkpoint)

        if cfg.get('unet_config') and cfg.get('unet_config').get('from_pretrained'):
            cfg.unet_config.from_pretrained = None
        if cfg.get('first_stage_config') and cfg.get('first_stage_config').get('from_pretrained'):
            cfg.first_stage_config.from_pretrained = None

        model = LatentDiffusion(cfg).to(device)

        sd_state_dict = {}
        for key, value in state_dict.items():
            key = key[6:]
            sd_state_dict[key] = value
        model.load_state_dict(sd_state_dict)

        return model.first_stage_model.encode

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


class StableDiffusion(nn.Module):
    def __init__(self, plan_dir, checkpoint, sampler_type="DDIM", t_range=[0.02, 0.98], device=torch.device('cuda')):
        super().__init__()
        logging.info(f'loading stable diffusion...')

        self.device = device
        self.sampler_type = sampler_type
        self.model = LatentDiffusionWrapper(plan_dir, checkpoint)

        self.text_encoder = self.model.clip_encode

        self.num_train_timesteps = self.model.num_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.model.alphas_cumprod.to(self.device)  # for convenience

        logging.info(f'loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        return self.text_encoder(prompt)

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.model.encode_imgs(pred_rgb_512)

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
