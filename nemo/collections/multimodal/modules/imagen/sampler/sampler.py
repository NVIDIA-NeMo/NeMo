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
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from nemo.collections.multimodal.modules.imagen.sampler.batch_ops import batch_div, batch_mul
from nemo.collections.multimodal.modules.imagen.sampler.continuous_ddpm import GaussianDiffusionContinuousTimes


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def thresholding_x0(x0, method='dynamic', th=0.995):
    if method is None:
        return x0
    elif method == 'static':
        return x0.clamp(-1.0, 1.0)
    elif method == 'dynamic':
        # torch.quantile only suppoprt either float or double dtype
        # we need to manual cast it if running in FP16/AMP mode
        original_dtype = x0.dtype
        if original_dtype not in [torch.float, torch.double]:
            x0 = x0.float()
        s = torch.quantile(rearrange(x0, 'b ... -> b (...)').abs(), th, dim=-1)  # From Figure A.10 (b)
        s.clamp_(min=1.0)
        s = right_pad_dims_to(x0, s)
        x0 = x0.clamp(-s, s) / s
        return x0.type(original_dtype)
    else:
        raise RuntimeError(f'Thresholding method: {method} not supported.')


def thresholding_derivative(x, t, d, thresholding_method='dynamic'):
    x0 = x - batch_mul(d, t)
    corrected_x0 = thresholding_x0(x0, thresholding_method)
    corrected_d = batch_div(x - corrected_x0, t)
    return corrected_d


class Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, model_kwargs, shape, z=None):
        pass


class DDPMSampler(Sampler):
    def __init__(self, unet_type, denoiser):
        super().__init__()
        self.unet_type = unet_type
        self.noise_scheduler = denoiser
        self.pred_objective = 'noise'

    def p_mean_variance(
        self, unet, x, t, t_next, text_embeds, text_mask, x_low_res=None, cond_scale=1.0, thresholding_method='dynamic'
    ):

        if self.unet_type == 'base':
            pred = unet.forward_with_cond_scale(
                x=x, time=t, text_embed=text_embeds, text_mask=text_mask, cond_scale=cond_scale
            )
        elif self.unet_type == 'sr':
            pred = unet.forward_with_cond_scale(
                x=x, x_low_res=x_low_res, time=t, text_embed=text_embeds, text_mask=text_mask, cond_scale=cond_scale
            )

        if self.pred_objective == 'noise':
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        elif self.pred_objective == 'x_start':
            x_start = pred
        elif self.pred_objective == 'v':
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        else:
            raise ValueError(f'unknown objective {self.pred_objective}')

        x_start = thresholding_x0(x_start, method=thresholding_method)
        mean_and_variance = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t, t_next=t_next)
        return mean_and_variance, x_start

    @torch.no_grad()
    def p_sample(
        self, unet, x, t, t_next, text_embeds, text_mask, x_low_res=None, cond_scale=1.0, thresholding_method='dynamic'
    ):
        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(
            unet=unet,
            x=x,
            t=t,
            t_next=t_next,
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_scale=cond_scale,
            x_low_res=x_low_res,
            thresholding_method=thresholding_method,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        b = x.shape[0]
        is_last_sampling_timestep = (
            (t_next == 0) if isinstance(self.noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        )
        nonzero_mask = (1 - is_last_sampling_timestep.type_as(x)).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    def forward(
        self,
        model,
        noise_map,
        text_encoding,
        text_mask,
        x_low_res=None,
        cond_scale=1.0,
        sampling_steps=None,
        thresholding_method='dynamic',
    ):
        batch = noise_map.shape[0]
        device = noise_map.device
        dtype = noise_map.dtype
        original_steps = self.noise_scheduler.num_timesteps
        if sampling_steps:
            self.noise_scheduler.num_timesteps = sampling_steps
        timesteps = self.noise_scheduler.get_sampling_timesteps(batch, device=device)
        img = noise_map
        for times, times_next in tqdm(timesteps, total=len(timesteps)):
            img, x_start = self.p_sample(
                unet=model,
                x=img.type(dtype),
                t=times.type(dtype),
                t_next=times_next.type(dtype),
                text_embeds=text_encoding,
                text_mask=text_mask,
                cond_scale=cond_scale,
                x_low_res=x_low_res.type(dtype) if x_low_res is not None else None,
                thresholding_method=thresholding_method,
            )
        self.noise_scheduler.num_timesteps = original_steps
        return img


class EDMSampler(Sampler):
    def __init__(
        self,
        unet_type,
        num_steps=50,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
    ):
        super().__init__()
        self.unet_type = unet_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.num_steps = num_steps

    def forward(
        self,
        unet,
        noise_map,
        text_encoding,
        text_mask,
        x_low_res=None,
        cond_scale=1.0,
        sampling_steps=None,
        thresholding_method='dynamic',
    ):
        if self.unet_type == 'base':
            assert x_low_res is None
        elif self.unet_type == 'sr':
            assert x_low_res is not None
        low_res_cond = {'x_low_res': x_low_res} if x_low_res is not None else {}
        thresholding_method = 'dynamic'
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        print(f'Sampling with sigma in [{sigma_min}, {sigma_max}], cfg={cond_scale}')
        # Time step discretization
        num_steps = sampling_steps if sampling_steps else self.num_steps
        step_indices = torch.arange(num_steps, device=noise_map.device)
        # Table 1: Sampling - Time steps
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = noise_map * t_steps[0]
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])), total=len(t_steps[:-1])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = (t_cur + gamma * t_cur).to(x_cur.device)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = unet.forward_with_cond_scale(
                x=x_hat.to(torch.float32),
                time=t_hat.to(torch.float32),
                text_embed=text_encoding,
                text_mask=text_mask,
                cond_scale=cond_scale,
                **low_res_cond,
            )
            d_cur = (x_hat - denoised) / t_hat
            d_cur = thresholding_derivative(x_hat, t_hat, d_cur, thresholding_method=thresholding_method)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = unet.forward_with_cond_scale(
                    x=x_next.to(torch.float32),
                    time=t_next.to(torch.float32),
                    text_embed=text_encoding,
                    text_mask=text_mask,
                    cond_scale=cond_scale,
                    **low_res_cond,
                )
                d_prime = (x_next - denoised) / t_next
                d_prime = thresholding_derivative(x_next, t_next, d_prime, thresholding_method=thresholding_method)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next
