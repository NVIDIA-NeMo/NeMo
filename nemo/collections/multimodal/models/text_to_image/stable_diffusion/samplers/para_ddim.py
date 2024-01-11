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

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers import Sampler
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.base_sampler import AbstractBaseSampler
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import noise_like


class ParaDDIMSampler(AbstractBaseSampler):
    """ Parallel version of DDIM sampler. Utilizes Parallel Sampling (https://arxiv.org/abs/2305.16317).
        It reduces the latency of a model, but the total compute cost is increased.

        The main three parameters that affect the performance of the algorithm are:
            Parallelism (int): Defines the maximal size of the window. That many diffusion steps can happen in
                parallel.
            Tolerance (float): Sets the maximal error tolerance defined as a ratio between drift of the trajectory
                and noise. The larger the tolerance the faster the method is. The smaller the tolerance the better
                quality output is achieved.
            Number of GPUs (int): Number of GPUs utilizing DataParallel parallelism to compute diffusion steps in
                parallel.

        Different combination of these parameters values can result in different latency-quality-compute trade-off.
        For more details please refer to the Parallel Sampling paper (https://arxiv.org/abs/2305.16317).
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, sampler=Sampler.PARA_DDIM, **kwargs)

    @torch.no_grad()
    def p_sampling_fn(self):
        pass

    @torch.no_grad()
    def para_ddim_sampling_fn(
        self,
        cond: torch.tensor,
        batch_size: int,
        per_latent_shape: Tuple[int, ...],
        x_T: torch.tensor = None,
        steps: int = 50,
        parallelism: int = 8,
        tolerance: float = 0.1,
        temperature: float = 0.0,
        noise_dropout: float = 0.0,
        quantize_denoised: bool = False,
        unconditional_guidance_scale: float = 1.0,
        unconditional_conditioning: torch.tensor = None,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        print(
            f"Running {self.sampler.name} with {steps} timesteps, "
            f"parallelism={parallelism}, "
            f"and tolerance={tolerance}"
        )

        device = self.model.betas.device
        size = (batch_size, *per_latent_shape)
        x_T = torch.randn(size, generator=self.model.rng, device=device) if x_T is None else x_T
        time_range = np.flip(self.ddim_timesteps).copy()  # Make a copy to resolve issue with negative strides

        # Processing window of timesteps [window_start, window_end) in parallel
        window_start = 0
        window_size = min(parallelism, steps)
        window_end = window_size

        # Store the whole trajectory in memory; it will be iteratively improved
        latents = torch.stack([x_T] * (steps + 1))

        # Pre-computing noises to ensure noise is sampled once per diffusion step
        noises = torch.zeros_like(latents)
        for i in range(steps - 1, -1, -1):
            gaussian_noise = torch.randn_like(x_T)
            noise = (self.ddim_variance[i] ** 0.5) * gaussian_noise
            noises[i] = noise.clone()

        # Store inverse of the variance to avoid division at every iteration
        variance = [self.ddim_variance[i] for i in range(steps - 1, -1, -1)] + [0]
        inverse_variance = 1.0 / torch.tensor(variance).to(noises.device)
        latent_dim = noises[0, 0].numel()
        inverse_variance_norm = inverse_variance[:, None] / latent_dim

        scaled_tolerance = tolerance ** 2

        with tqdm(total=steps) as progress_bar:
            while window_start < steps:
                window_size = window_end - window_start

                # Prepare the input to the model. Model will perform window_size noise predictions in parallel
                window_cond = torch.stack([cond] * window_size)
                window_uncond_cond = torch.stack([unconditional_conditioning] * window_size)
                window_latents = latents[window_start:window_end]
                window_timesteps = torch.tensor(time_range[window_start:window_end], device=device).repeat(
                    1, batch_size
                )

                # Reshape (w, b, ...) -> (w * b, ...)
                latents_input = window_latents.flatten(0, 1)
                timesteps_input = window_timesteps.flatten(0, 1)
                cond_input = window_cond.flatten(0, 1)
                uncond_cond_input = window_uncond_cond.flatten(0, 1)

                # Model call
                e_t, _ = self._get_model_output(
                    latents_input,
                    timesteps_input,
                    uncond_cond_input,
                    unconditional_guidance_scale,
                    score_corrector,
                    cond_input,
                    corrector_kwargs,
                )
                # Reshape back (w * b, ...) -> (w, b, ...)
                e_t = e_t.reshape(window_size, batch_size, *per_latent_shape)

                # Perform Picard iteration
                window_latents_picard_iteration = self._get_x_prev(
                    batch_size=batch_size,
                    steps=steps,
                    x=window_latents,
                    e_t=e_t,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    quantize_denoised=quantize_denoised,
                    window_start=window_start,
                    window_end=window_end,
                    device=device,
                ).reshape(window_latents.shape)

                # Calculate cumulative drift
                delta = window_latents_picard_iteration - window_latents
                delta_cum = torch.cumsum(delta, dim=0)
                block_latents_new = latents[window_start][None,] + delta_cum

                # Calculate the error
                error = torch.linalg.norm(
                    (block_latents_new - latents[window_start + 1 : window_end + 1]).reshape(
                        window_size, batch_size, -1
                    ),
                    dim=-1,
                ).pow(2)

                # Calculate error magnitude
                error_magnitude = error * inverse_variance_norm[window_start + 1 : window_end + 1]
                # Pad so at least one value exceeds tolerance
                error_magnitude = nn.functional.pad(error_magnitude, (0, 0, 0, 1), value=1e9)
                error_exceeding = torch.max(error_magnitude > scaled_tolerance, dim=1).values.int()

                # Find how many diffusion steps have error below given threshold tolerance and shift the window
                ind = torch.argmax(error_exceeding).item()
                new_window_start = window_start + min(1 + ind, window_size)
                new_window_end = min(new_window_start + window_size, steps)

                # Update the trajectory
                latents[window_start + 1 : window_end + 1] = block_latents_new
                latents[window_end : new_window_end + 1] = latents[window_end][
                    None,
                ]

                progress_bar.update(new_window_start - window_start)
                window_start = new_window_start
                window_end = new_window_end

        intermediates = {"x_inter": [latents[i] for i in range(steps)]}
        return latents[-1], intermediates

    def _get_x_prev(
        self,
        batch_size: int,
        steps: int,
        x: torch.tensor,
        e_t: torch.tensor,
        temperature: float,
        noise_dropout: float,
        quantize_denoised: bool,
        window_start: int,
        window_end: int,
        device: Any,
    ):
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        window_size = window_end - window_start

        def prepare_tensor(x):
            x = torch.tensor(x, device=device).flip(dims=[0])
            x = x.unsqueeze(1).repeat(1, batch_size).reshape(window_size, batch_size, 1, 1, 1)
            return x

        # Select parameters corresponding to the currently considered timesteps. Note that index_end < index_start,
        # because during diffusion the time is reversed (we go from timestep step to 0)
        index_start = steps - window_start
        index_end = steps - window_end
        a_t = prepare_tensor(alphas[index_end:index_start])
        a_prev = prepare_tensor(alphas_prev[index_end:index_start])
        sigma_t = prepare_tensor(sigmas[index_end:index_start])
        sqrt_one_minus_at = prepare_tensor(sqrt_one_minus_alphas[index_end:index_start])

        # Current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        # Direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev
