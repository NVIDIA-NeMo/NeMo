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
from typing import List

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline

from nemo.collections.multimodal.modules.nerf.guidance.txt2img_guidance_base import Txt2ImgGuidanceBase


class StableDiffusion(Txt2ImgGuidanceBase):
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-2-1-base",
        t_range: List[float] = [0.02, 0.98],
        precision: str = "16",
        device: torch.device = torch.device('cuda'),
    ):
        """
        Initialize StableDiffusion with model_key, t_range, precision and device.

        Parameters:
            model_key (str): Pre-trained model key.
            t_range (List[float]): Range for timesteps.
            precision (str): Model precision ("16", "bf16" or other for float32).
            device (torch.device): Device for torch tensor.
        """
        super().__init__()

        self.device = device
        self.model_key = model_key
        self.precision_t = self._get_precision_type(precision)

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t).to(self.device)
        if self.precision_t in [torch.float16, torch.bfloat16]:
            pipe.unet.to(memory_format=torch.channels_last)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    def _get_precision_type(self, precision: str) -> torch.dtype:
        """
        Map string precision representation to torch dtype.

        Parameters:
            precision (str): String representation of precision.

        Returns:
            torch.dtype: Corresponding torch dtype.
        """
        precision_map = {"16": torch.float16, "bf16": torch.bfloat16}
        return precision_map.get(precision, torch.float32)

    @torch.no_grad()
    def get_text_embeds(self, prompt: str) -> torch.Tensor:
        """
        Get text embeddings from the given prompt.

        Parameters:
            prompt (str): Input text.

        Returns:
            torch.Tensor: Text embeddings tensor [B, 77, 1024].
        """
        inputs = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    # @torch.compile() # TODO(ahmadki)
    def train_step(
        self,
        text_embeddings: torch.Tensor,
        pred_rgb: torch.Tensor,
        guidance_scale: float = 100.0,
        as_latent: bool = False,
    ) -> float:
        """
        Train step function for StableDiffusion.

        Parameters:
            text_embeddings (torch.Tensor): Embeddings tensor [B, 512].
            pred_rgb (torch.Tensor): Predicted RGB tensor [B, 3, 512, 512].
            guidance_scale (float): Guidance scaling factor.
            as_latent (bool): If True, considers pred_rgb as latent.

        Returns:
            float: Loss value.
        """
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)

        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            td = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, td, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        w = 1 - self.alphas[t]
        grad = w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        return loss

    def encode_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode images into latent representations.

        Parameters:
            imgs (torch.Tensor): Image tensor [B, 3, H, W].

        Returns:
            torch.Tensor: Encoded latent tensor.
        """
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents
