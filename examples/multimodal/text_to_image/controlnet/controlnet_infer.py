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
import time

import cv2
import einops
import torch
from PIL import Image

from nemo.collections.multimodal.models.text_to_image.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.ddim import DDIMSampler
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.plms import PLMSSampler
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner


def get_control_input(image_path, batch_size, hint_image_size, control_image_preprocess=None):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (hint_image_size, hint_image_size))
    control = torch.from_numpy(image).float() / 255.0
    control = torch.stack([control for _ in range(batch_size)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w')
    return control


def encode_prompt(cond_stage_model, prompt, unconditional_guidance_scale, batch_size):
    c = cond_stage_model.encode(batch_size * [prompt])
    if unconditional_guidance_scale != 1.0:
        uc = cond_stage_model.encode(batch_size * [""])
    else:
        uc = None
    return c, uc


def initialize_sampler(model, sampler_type):
    if sampler_type == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_type == 'PLMS':
        sampler = PLMSSampler(model)
    else:
        raise ValueError(f'Sampler {sampler_type} is not supported for {cls.__name__}')
    return sampler


def decode_images(model, samples):
    images = model.decode_first_stage(samples)

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

    return images


def torch_to_numpy(images):
    numpy_images = [x.float().cpu().permute(0, 2, 3, 1).numpy() for x in images]
    return numpy_images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def pipeline(model, cfg, rng=None, verbose=True):
    # setup default values for inference configs
    unconditional_guidance_scale = cfg.infer.get("unconditional_guidance_scale", 7.5)
    batch_size = cfg.infer.get('num_images_per_prompt', 1)
    prompts = cfg.infer.get('prompts', [])
    control = cfg.infer.get('control', [])
    height = cfg.infer.get('height', 512)
    width = cfg.infer.get('width', 512)
    downsampling_factor = cfg.infer.get('down_factor', 8)
    sampler_type = cfg.infer.get('sampler_type', 'DDIM')
    inference_steps = cfg.infer.get('inference_steps', 50)
    output_type = cfg.infer.get('output_type', 'pil')
    save_to_file = cfg.infer.get('save_to_file', True)
    out_path = cfg.infer.get('out_path', '')
    eta = cfg.infer.get('eta', 0)
    guess_mode = cfg.model.get('guess_mode', False)
    hint_image_size = cfg.infer.get('hint_image_size', 512)
    control_image_preprocess = cfg.infer.get('control_image_preprocess', None)

    # get autocast_dtype
    if cfg.trainer.precision in ['bf16', 'bf16-mixed']:
        autocast_dtype = torch.bfloat16
    elif cfg.trainer.precision in [32, '32', '32-true']:
        autocast_dtype = torch.float
    elif cfg.trainer.precision in [16, '16', '16-mixed']:
        autocast_dtype = torch.half
    else:
        raise ValueError('precision must be in [32, 16, "bf16"]')

    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=autocast_dtype in (torch.half, torch.bfloat16), dtype=autocast_dtype,
    ):

        in_channels = model.model.diffusion_model.in_channels

        sampler = initialize_sampler(model, sampler_type.upper())

        output = []
        throughput = []

        if isinstance(prompts, str):
            prompts = [prompts]

        assert len(prompts) == len(control)

        for control, prompt in zip(control, prompts):
            tic = time.perf_counter()
            tic_total = tic
            txt_cond, txt_u_cond = encode_prompt(
                model.cond_stage_model, prompt, unconditional_guidance_scale, batch_size
            )

            control = get_control_input(control, batch_size, hint_image_size, control_image_preprocess).to(
                torch.cuda.current_device(), dtype=autocast_dtype
            )

            cond = {"c_concat": control, "c_crossattn": txt_cond}
            u_cond = {"c_concat": None if guess_mode else control, "c_crossattn": txt_u_cond}

            toc = time.perf_counter()
            conditioning_time = toc - tic

            latent_shape = [batch_size, height // downsampling_factor, width // downsampling_factor]
            latents = torch.randn(
                [batch_size, in_channels, height // downsampling_factor, width // downsampling_factor], generator=rng
            ).to(torch.cuda.current_device())

            tic = time.perf_counter()
            samples, intermediates = sampler.sample(
                S=inference_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=latent_shape,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=u_cond,
                eta=eta,
                x_T=latents,
            )
            toc = time.perf_counter()
            sampling_time = toc - tic

            tic = time.perf_counter()
            images = decode_images(model, samples)
            toc = time.perf_counter()
            decode_time = toc - tic

            toc_total = time.perf_counter()
            total_time = toc_total - tic_total
            output.append(images)

            throughput.append(
                {
                    'text-conditioning-time': conditioning_time,
                    'sampling-time': sampling_time,
                    'decode-time': decode_time,
                    'total-time': total_time,
                    'sampling-steps': inference_steps,
                }
            )

        # Convert output type and save to disk
        if output_type == 'torch':
            output = torch.cat(output, dim=0)
        else:
            output = torch_to_numpy(output)
            if output_type == 'pil':
                output = [numpy_to_pil(x) for x in output]

        if save_to_file:
            os.makedirs(out_path, exist_ok=True)
            # Saving control map
            control_image = control[0].float().cpu().permute(1, 2, 0).numpy()
            control_image = Image.fromarray((control_image * 255).round().astype("uint8"))
            control_image.save(os.path.join(out_path, f'{prompt[:50]}_control.png'))
            if output_type == 'pil':
                for text_prompt, pils in zip(prompts, output):
                    for idx, image in enumerate(pils):
                        image.save(os.path.join(out_path, f'{text_prompt[:50]}_{idx}.png'))
            else:
                with open(os.path.join(out_path, 'output.pkl'), 'wb') as f:
                    pickle.dump(output, f)
        else:
            return output

        ave_metrics = {}
        for key in throughput[0].keys():
            ave_metrics[f'avg-{key}'] = sum([dicts[key] for dicts in throughput]) / len(throughput)
        if verbose:
            print(ave_metrics)


@hydra_runner(config_path='conf', config_name='controlnet_infer')
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.control_stage_config.from_pretrained_unet = None
        model_cfg.channels_last = True
        model_cfg.capture_cudagraph_iters = -1

    torch.backends.cuda.matmul.allow_tf32 = True
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronControlNet, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )
    model = megatron_diffusion_model.model
    model.cuda().eval()

    guess_mode = cfg.model.guess_mode
    model.contol_scales = (
        [cfg.model.strength * (0.825 ** float(12 - i)) for i in range(13)]
        if guess_mode
        else ([cfg.model.strength] * 13)
    )

    rng = torch.Generator().manual_seed(cfg.infer.seed)
    pipeline(model, cfg, rng=rng)


if __name__ == "__main__":
    main()
