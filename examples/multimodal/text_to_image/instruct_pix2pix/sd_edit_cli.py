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
from __future__ import annotations

import math
import os
import random

import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageOps

from nemo.collections.multimodal.models.text_to_image.instruct_pix2pix.ldm.ddpm_edit import MegatronLatentDiffusionEdit
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.k_diffusion import (
    DiscreteEpsDDPMDenoiser,
    sample_euler_ancestral,
)
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner
from nemo.utils import logging


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (n b) ...", n=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (n b) ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        out = out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        return out


@hydra_runner(config_path='conf', config_name='sd_edit')
def main(cfg):
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    with open_dict(cfg):
        edit_cfg = cfg.pop("edit")

    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False

    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusionEdit, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
    )

    # inference use the latent diffusion part of megatron wrapper
    model = megatron_diffusion_model.model
    model_wrap = DiscreteEpsDDPMDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if edit_cfg.seed is None else edit_cfg.seed
    input_image = Image.open(edit_cfg.input).convert("RGB")
    width, height = input_image.size
    factor = edit_cfg.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if edit_cfg.prompt == "":
        input_image.save(edit_cfg.output)
        return

    # get autocast_dtype
    if trainer.precision in ['bf16', 'bf16-mixed']:
        autocast_dtype = torch.bfloat16
    elif trainer.precision in [32, '32', '32-true']:
        autocast_dtype = torch.float
    elif trainer.precision in [16, '16', '16-mixed']:
        autocast_dtype = torch.half
    else:
        raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')

    num_images_per_prompt = edit_cfg.num_images_per_prompt
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):
        cond = {}
        cond["c_crossattn"] = [
            repeat(model.get_learned_conditioning([edit_cfg.prompt]), "1 ... -> n ...", n=num_images_per_prompt)
        ]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").cuda(non_blocking=True)
        cond["c_concat"] = [
            repeat(model.encode_first_stage(input_image).mode(), "1 ... -> n ...", n=num_images_per_prompt)
        ]

        uncond = {}
        uncond["c_crossattn"] = [repeat(null_token, "1 ... -> n ...", n=num_images_per_prompt)]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(edit_cfg.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": edit_cfg.cfg_text,
            "image_cfg_scale": edit_cfg.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0])
        z = z * sigmas[0]
        z = sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "n c h w -> n h w c")

    os.makedirs(edit_cfg.outpath, exist_ok=True)
    if edit_cfg.get("combine_images") is None:
        for idx, image in enumerate(x):
            edited_image = Image.fromarray(image.type(torch.uint8).cpu().numpy())
            save_path = os.path.join(
                edit_cfg.outpath,
                f'{edit_cfg.prompt.replace(" ", "_")}_{edit_cfg.cfg_text}_{edit_cfg.cfg_image}_{seed}_{idx}.jpg',
            )
            edited_image.save(save_path)
            logging.info(f"Edited image saved to: {save_path}")
    else:
        row, column = edit_cfg.combine_images
        width, height = x.size(2), x.size(1)
        total_width, total_height = width * column, height * row
        edited_image = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        y_offset = 0
        for idx, image in enumerate(x):
            image = Image.fromarray(image.type(torch.uint8).cpu().numpy())
            edited_image.paste(image, (x_offset, y_offset))
            x_offset += image.size[0]
            if (idx + 1) % column == 0:
                x_offset = 0
                y_offset += height
        save_path = os.path.join(
            edit_cfg.outpath,
            f'{edit_cfg.prompt.replace(" ", "_")}_{edit_cfg.cfg_text}_{edit_cfg.cfg_image}_{seed}_combine.jpg',
        )
        edited_image.save(save_path)
        logging.info(f"Edited image saved to: {save_path}")


if __name__ == "__main__":
    main()
