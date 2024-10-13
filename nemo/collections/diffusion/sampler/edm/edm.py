# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from statistics import NormalDist
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class EDMScaling:
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class EDMSDE:
    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
    ):
        self.gaussian_dist = NormalDist(mu=p_mean, sigma=p_std)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self._generator = np.random

    def sample_t(self, batch_size: int) -> torch.Tensor:
        cdf_vals = self._generator.uniform(size=(batch_size))
        samples_interval_gaussian = [self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]
        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        return torch.exp(log_sigma)

    def marginal_prob(self, x0: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x0, sigma


class EDMSampler(nn.Module):
    """
    Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
    # https://github.com/NVlabs/edm/blob/62072d2612c7da05165d6233d13d17d71f213fee/generate.py#L25

    Attributes:
        None

    Methods:
        forward(x0_fn: Callable, x_sigma_max: torch.Tensor, num_steps: int = 35, sigma_min: float = 0.002,
                sigma_max: float = 80, rho: float = 7, S_churn: float = 0, S_min: float = 0,
                S_max: float = float("inf"), S_noise: float = 1) -> torch.Tensor:
            Performs the forward pass for the EDM sampling process.

            Parameters:
                x0_fn (Callable): A function that takes in a tensor and returns a denoised tensor.
                x_sigma_max (torch.Tensor): The initial noise level tensor.
                num_steps (int, optional): The number of sampling steps. Default is 35.
                sigma_min (float, optional): The minimum noise level. Default is 0.002.
                sigma_max (float, optional): The maximum noise level. Default is 80.
                rho (float, optional): The rho parameter for time step discretization. Default is 7.
                S_churn (float, optional): The churn parameter for noise increase. Default is 0.
                S_min (float, optional): The minimum value for the churn parameter. Default is 0.
                S_max (float, optional): The maximum value for the churn parameter. Default is float("inf").
                S_noise (float, optional): The noise scale for the churn parameter. Default is 1.

            Returns:
                torch.Tensor: The sampled tensor after the EDM process.
    """

    @torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
    ) -> torch.Tensor:
        # Time step discretization.
        in_dtype = x_sigma_max.dtype
        _ones = torch.ones(x_sigma_max.shape[0], dtype=in_dtype, device=x_sigma_max.device)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=x_sigma_max.device)
        t_steps = (
            sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = x_sigma_max.to(torch.float64)
        for i, (t_cur, t_next) in enumerate(
            tqdm(zip(t_steps[:-1], t_steps[1:], strict=False), total=len(t_steps) - 1)
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = x0_fn(x_hat.to(in_dtype), t_hat.to(in_dtype) * _ones).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = x0_fn(x_hat.to(in_dtype), t_hat.to(in_dtype) * _ones).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(in_dtype)
