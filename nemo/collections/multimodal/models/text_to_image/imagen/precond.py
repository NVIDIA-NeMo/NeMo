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
import torch.nn.functional as F

from nemo.collections.multimodal.modules.imagen.sampler.batch_ops import batch_mul
from nemo.collections.multimodal.modules.imagen.sampler.continuous_ddpm import GaussianDiffusionContinuousTimes
from nemo.collections.multimodal.parts.utils import randn_like


class PrecondModel(torch.nn.Module):
    def __init__(self, unet, loss_type):
        super().__init__()
        self.unet = unet
        self.rng = None
        self.inference = False
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        elif loss_type == 'huber':
            self.loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f'{loss_type} loss is not supported')

    def set_inference_mode(self, value):
        self.inference = value

    def forward(self, **model_kwargs):
        return self.unet(**model_kwargs)

    def forward_with_cond_scale(self, *args, text_embed=None, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, text_embed=text_embed, **kwargs)
        if cond_scale == 1.0:
            return logits
        null_logits = self.forward(*args, text_embed=torch.zeros_like(text_embed), **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def set_rng(self, generator):
        self.rng = generator


class ContinousDDPMPrecond(PrecondModel):
    def __init__(
        self,
        unet,
        loss_type='l2',
        pred_objective='noise',
        noise_schedule='cosine',
        timesteps=1000,
        noise_cond_aug=False,
    ):
        super().__init__(unet, loss_type)
        self.scheduler = GaussianDiffusionContinuousTimes(noise_schedule=noise_schedule, timesteps=timesteps)
        self.pred_objective = pred_objective
        assert noise_cond_aug == False, 'noise cond aug currently not supported for DDPM'

    def sample_time(self, batch_size, device=None):
        return self.scheduler.sample_random_times(batch_size=batch_size, device=device)

    def get_xt(self, x0, t=None, epsilon=None):
        if epsilon is None:
            epsilon = randn_like(x0, generator=self.rng)
        if t is None:
            t = self.sample_time(batch_size=x0.shape[0], device=x0.device)
        x_noisy, log_snr, alpha, sigma = self.scheduler.q_sample(x_start=x0, t=t, noise=epsilon,)
        return x_noisy, t, epsilon

    def forward(self, x, time, text_embed, text_mask, **model_kwargs):
        # Convert time to FP32 for calculating time embedding due to FP16 overflow
        time = time.float()
        time = self.scheduler.get_condition(time)
        time = time.type_as(x)

        return self.unet(x=x, time=time, text_embed=text_embed, text_mask=text_mask, **model_kwargs)

    def compute_loss(self, x0, text_embed, text_mask, time=None, noise=None, **model_kwargs):
        x_noisy, time, noise = self.get_xt(x0=x0, t=time, epsilon=noise)
        pred = self.forward(x_noisy, time, text_embed, text_mask, **model_kwargs)
        # Determine target
        if self.pred_objective == 'noise':
            target = noise
        elif self.pred_objective == 'x_start':
            target = x0
        else:
            raise ValueError(f'unknown objective {self.pred_objective}')
        return self.loss_fn(pred, target)

    def set_rng(self, generator):
        self.scheduler.rng = generator
        self.rng = generator


class EDMPrecond(PrecondModel):
    def __init__(
        self,
        unet,  # Underlying model.
        loss_type='l2',
        sigma_data=0.5,  # Expected standard deviation of the training data.
        p_mean=-1.2,
        p_std=1.2,
        noise_cond_aug=False,
    ):
        super().__init__(unet, loss_type)
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_cond_aug = noise_cond_aug

    def forward(self, x, time, text_embed, text_mask, **model_kwargs):
        bs = x.shape[0]
        assert time.ndim <= 1, 'time should be in shape of either [bs] or scalar'
        sigma = time
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        if c_noise.ndim < 1:
            c_noise = c_noise.repeat(bs,)

        if self.noise_cond_aug:
            # Applying noise conditioning augmentation
            assert 'x_low_res' in model_kwargs, 'x_low_res does not exist when attemping to apply noise augmentation'
            x_low_res = model_kwargs['x_low_res']
            if self.inference:
                batch_size = x_low_res.shape[0]
                time_low_res = torch.ones(batch_size, device=x_low_res.device) * 0.002
                x_low_res_noisy, time_low_res = self.get_xt(x0=x_low_res, t=time_low_res, epsilon=None)
            else:
                x_low_res_noisy, time_low_res = self.get_xt(x0=x_low_res, t=None, epsilon=None)
            c_in_noise = 1 / (self.sigma_data ** 2 + time_low_res ** 2).sqrt()
            c_noise_noise = time_low_res.log() / 4
            model_kwargs['x_low_res'] = batch_mul(c_in_noise, x_low_res_noisy)
            model_kwargs['time_low_res'] = c_noise_noise

        F_x = self.unet(batch_mul(c_in, x), c_noise, text_embed, text_mask, **model_kwargs)
        D_x = batch_mul(c_skip, x) + batch_mul(c_out, F_x)
        return D_x

    def sample_time(self, batch_size, device=None):
        return (torch.randn(batch_size, device=device, generator=self.rng) * self.p_std + self.p_mean).exp()

    def get_xt(self, x0, t=None, epsilon=None):
        if epsilon is None:
            epsilon = randn_like(x0, generator=self.rng)
        assert epsilon.shape == x0.shape
        if t is None:
            t = self.sample_time(batch_size=x0.shape[0], device=x0.device)
        sigma = t
        noise = batch_mul(epsilon, sigma)
        return x0 + noise, sigma

    def compute_loss(self, x0, text_embed, text_mask, time=None, noise=None, **model_kwargs):
        x_noisy, time = self.get_xt(x0=x0, t=None, epsilon=noise)
        pred = self.forward(x_noisy, time, text_embed, text_mask, **model_kwargs)
        sigma = time
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2).sqrt()
        target = x0
        return self.loss_fn(batch_mul(weight, target), batch_mul(weight, pred),)

    def set_rng(self, generator):
        self.rng = generator
