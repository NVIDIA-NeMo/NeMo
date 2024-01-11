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


class PLMSSampler(AbstractBaseSampler):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__(model, sampler=Sampler.PLMS, schedule="linear", **kwargs)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=False):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        super().make_schedule(ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=False)

    @torch.no_grad()
    def p_sampling_fn(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device
        e_t, model_output = self._get_model_output(
            x, t, unconditional_conditioning, unconditional_guidance_scale, score_corrector, c, corrector_kwargs
        )
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = self._get_x_prev_and_pred_x0(
                use_original_steps,
                b,
                index,
                device,
                x,
                t,
                model_output,
                e_t,
                quantize_denoised,
                repeat_noise,
                temperature,
                noise_dropout,
            )
            e_t_next, model_output = self._get_model_output(
                x_prev,
                t_next,
                unconditional_conditioning,
                unconditional_guidance_scale,
                score_corrector,
                c,
                corrector_kwargs,
            )
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = self._get_x_prev_and_pred_x0(
            use_original_steps,
            b,
            index,
            device,
            x,
            t,
            model_output,
            e_t_prime,
            quantize_denoised,
            repeat_noise,
            temperature,
            noise_dropout,
        )

        return x_prev, pred_x0, e_t
