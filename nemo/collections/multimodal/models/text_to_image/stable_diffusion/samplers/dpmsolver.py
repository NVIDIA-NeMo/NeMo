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
import math

import torch

from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import expand_dims, interpolate_fn


class NoiseScheduleVP:
    def __init__(
        self, schedule="discrete", betas=None, alphas_cumprod=None, continuous_beta_0=0.1, continuous_beta_1=20.0,
    ):
        """Create a wrapper class for the forward SDE."""

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule
                )
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape((1, -1,))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            if schedule == "cosine":
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device),
            ).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":

            def log_alpha_fn(s):
                return torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))

            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))

            def t_fn(log_alpha_t):
                return (
                    torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0))
                    * 2.0
                    * (1.0 + self.cosine_s)
                    / math.pi
                    - self.cosine_s
                )

            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model."""

    def get_model_input_time(t_continuous):
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = (
                noise_schedule.marginal_alpha(t_continuous),
                noise_schedule.marginal_std(t_continuous),
            )
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t, sigma_t = (
                noise_schedule.marginal_alpha(t_continuous),
                noise_schedule.marginal_std(t_continuous),
            )
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, dims=cond_grad.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPMSolver:
    def __init__(
        self, model_fn, noise_schedule, predict_x0=False, thresholding=False, max_val=1.0,
    ):
        """Construct a DPM-Solver."""
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = (
            self.noise_schedule.marginal_alpha(t),
            self.noise_schedule.marginal_std(t),
        )
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims)
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.predict_x0:
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = expand_dims(sigma_t / sigma_s, dims) * x - expand_dims(alpha_t * phi_1, dims) * model_s
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_s), dims) * x
                - expand_dims(sigma_t * phi_1, dims) * model_s
            )
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpm_solver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.
        """
        if solver_type not in ["dpm_solver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpm_solver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        dims = x.dim()
        model_prev_1, model_prev_0 = model_prev_list
        t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = (
            ns.marginal_log_mean_coeff(t_prev_0),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = expand_dims(1.0 / r0, dims) * (model_prev_0 - model_prev_1)
        if self.predict_x0:
            if solver_type == "dpm_solver":
                x_t = (
                    expand_dims(sigma_t / sigma_prev_0, dims) * x
                    - expand_dims(alpha_t * (torch.exp(-h) - 1.0), dims) * model_prev_0
                    - 0.5 * expand_dims(alpha_t * (torch.exp(-h) - 1.0), dims) * D1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    expand_dims(sigma_t / sigma_prev_0, dims) * x
                    - expand_dims(alpha_t * (torch.exp(-h) - 1.0), dims) * model_prev_0
                    + expand_dims(alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0), dims) * D1_0
                )
        else:
            if solver_type == "dpm_solver":
                x_t = (
                    expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                    - expand_dims(sigma_t * (torch.exp(h) - 1.0), dims) * model_prev_0
                    - 0.5 * expand_dims(sigma_t * (torch.exp(h) - 1.0), dims) * D1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                    - expand_dims(sigma_t * (torch.exp(h) - 1.0), dims) * model_prev_0
                    - expand_dims(sigma_t * ((torch.exp(h) - 1.0) / h - 1.0), dims) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpm_solver"):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = (
            ns.marginal_log_mean_coeff(t_prev_0),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = expand_dims(1.0 / r0, dims) * (model_prev_0 - model_prev_1)
        D1_1 = expand_dims(1.0 / r1, dims) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + expand_dims(r0 / (r0 + r1), dims) * (D1_0 - D1_1)
        D2 = expand_dims(1.0 / (r0 + r1), dims) * (D1_0 - D1_1)
        if self.predict_x0:
            x_t = (
                expand_dims(sigma_t / sigma_prev_0, dims) * x
                - expand_dims(alpha_t * (torch.exp(-h) - 1.0), dims) * model_prev_0
                + expand_dims(alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0), dims) * D1
                - expand_dims(alpha_t * ((torch.exp(-h) - 1.0 + h) / h ** 2 - 0.5), dims) * D2
            )
        else:
            x_t = (
                expand_dims(torch.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * (torch.exp(h) - 1.0), dims) * model_prev_0
                - expand_dims(sigma_t * ((torch.exp(h) - 1.0) / h - 1.0), dims) * D1
                - expand_dims(sigma_t * ((torch.exp(h) - 1.0 - h) / h ** 2 - 0.5), dims) * D2
            )
        return x_t

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpm_solver"):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpm_solver",
        atol=0.0078,
        rtol=0.05,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device

        if method == "multistep":
            assert steps >= order
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
            with torch.no_grad():
                vec_t = timesteps[0].expand((x.shape[0]))
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
                # Init the first `order` values by lower order multistep DPM-Solver.
                for init_order in range(1, order):
                    vec_t = timesteps[init_order].expand(x.shape[0])
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, vec_t, init_order, solver_type=solver_type,
                    )
                    model_prev_list.append(self.model_fn(x, vec_t))
                    t_prev_list.append(vec_t)
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    vec_t = timesteps[step].expand(x.shape[0])
                    if lower_order_final and steps < 15:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, vec_t, step_order, solver_type=solver_type,
                    )
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = vec_t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, vec_t)
        if denoise_to_zero:
            x = self.denoise_to_zero_fn(x, torch.ones((x.shape[0],)).to(device) * t_0)
        return x
