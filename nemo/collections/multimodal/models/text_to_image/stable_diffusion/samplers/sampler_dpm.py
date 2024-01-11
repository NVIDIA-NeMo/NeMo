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
"""SAMPLING ONLY."""

import torch

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers import Sampler
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.samplers.base_sampler import AbstractBaseSampler
from .dpmsolver import DPMSolver, NoiseScheduleVP, model_wrapper

MODEL_TYPES = {"eps": "noise", "v": "v"}


class DPMSolverSampler(AbstractBaseSampler):
    def __init__(self, model, **kwargs):

        super().__init__(model, sampler=Sampler.DPM, **kwargs)

        def to_torch(x, model):
            x_copy = x.clone()
            x_detached = x_copy.detach()
            x_float32 = x_detached.to(torch.float32)
            x_device = x_float32.to(model.betas.device)
            return x_device

        self.register_buffer("alphas_cumprod", to_torch(model.alphas_cumprod, model))

    @torch.no_grad()
    def p_sampling_fn(self):
        pass

    @torch.no_grad()
    def dpm_sampling_fn(
        self,
        shape,
        steps,
        conditioning=None,
        unconditional_conditioning=None,
        unconditional_guidance_scale=1.0,
        x_T=None,
    ):

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(shape, generator=self.model.rng, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type=MODEL_TYPES[self.model.parameterization],
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )
        dpm_solver = DPMSolver(model_fn, ns, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(
            img, steps=steps, skip_type="time_uniform", method="multistep", order=2, lower_order_final=True,
        )

        return x.to(device), None
