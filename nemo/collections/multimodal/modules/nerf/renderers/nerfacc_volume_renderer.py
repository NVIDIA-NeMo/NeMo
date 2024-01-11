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
import collections
from typing import Optional

import torch
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import accumulate_along_rays_, render_weight_from_density, rendering

from nemo.collections.multimodal.modules.renderer.base_renderer import BaseRenderer

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def render_image_with_occgrid(
    # scene
    nerf: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    # TODO(ahmadki): optimize, cache result between sigma_fn and rgb_sigma_fn
    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        sigmas = nerf.density(positions)['sigma']
        return sigmas

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        sigmas, rgbs, normal = nerf(
            positions=positions, view_dirs=None, light_dirs=t_dirs
        )  # TODO(ahmadki): t_dirs is incorrect
        return rgbs, sigmas

    results = []
    chunk = torch.iinfo(torch.int32).max if nerf.training else test_chunk_size

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=nerf.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )

        weight = extras["weights"]
        alpha = extras["alphas"]

        chunk_results = [rgb, opacity, depth, weight, alpha, len(t_starts)]
        results.append(chunk_results)

    colors, opacities, depths, weights, alphas, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r for r in zip(*results)
    ]

    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        weights,
        alphas,
        sum(n_rendering_samples),
    )


@torch.no_grad()
def render_image_with_occgrid_test(
    max_samples: int,
    # scene
    nerf: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0
        sigmas, rgbs, normal = nerf(
            positions=positions, view_dirs=None, light_dirs=t_dirs
        )  # TODO(ahmadki): t_dirs is incorrect ?
        return rgbs, sigmas

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, 3, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(0, n_grids * 2, device=t_mins.device, dtype=torch.int64).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:

        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # the number of samples to add on each ray
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # volume rendering using native cuda scan
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )

        accumulate_along_rays_(
            weights, values=rgbs, ray_indices=ray_indices, outputs=rgb,
        )
        accumulate_along_rays_(
            weights, values=None, ray_indices=ray_indices, outputs=opacity,
        )
        accumulate_along_rays_(
            weights, values=(t_starts + t_ends)[..., None] / 2.0, ray_indices=ray_indices, outputs=depth,
        )
        # update near_planes using termination planes
        near_planes = termination_planes
        # update rays status
        ray_mask = torch.logical_and(
            # early stopping
            opacity.view(-1) <= opc_thre,
            # remove rays that have reached the far plane
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    if render_bkgd is not None:
        rgb = rgb + render_bkgd * (1.0 - opacity)

    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)

    return (
        rgb.view((*rays_shape[:-1], -1)),
        opacity.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        weights,
        alphas,
        total_samples,
    )


class NerfaccVolumeBaseRenderer(BaseRenderer):
    def __init__(
        self,
        bound,
        grid_resolution,
        grid_levels,
        render_step_size=1e-3,
        near_plane=0.2,
        cone_angle=0.004,
        alpha_thre=1e-2,
    ):

        super().__init__(bound)

        self.grid_resolution = grid_resolution
        self.grid_levels = grid_levels
        self.render_step_size = render_step_size
        self.near_plane = near_plane
        self.cone_angle = cone_angle
        self.alpha_thre = alpha_thre
        self.nerf = None

        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=self.grid_resolution, levels=self.grid_levels)

    @torch.no_grad()  # TODO(ahmadki)
    def update_step(
        self,
        epoch: int,
        global_step: int,
        update_interval: int = 16,
        decay: float = 0.95,
        occ_thre: float = 0.01,
        warmup_steps: int = 256,
        **kwargs
    ):
        def occ_eval_fn(x):
            density = self.nerf.forward_density(x)
            return density * self.render_step_size

        self.estimator.update_every_n_steps(
            step=global_step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=occ_thre,
            ema_decay=decay,
            warmup_steps=warmup_steps,
            n=update_interval,
        )

    def forward(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, step=None, **kwargs):
        return self._render(rays_o=rays_o, rays_d=rays_d, step=step, **kwargs)

    def _render(
        self,
        rays_o,
        rays_d,
        light_d=None,
        ambient_ratio=1.0,
        shading='albedo',
        bg_color=None,
        perturb=False,
        T_thresh=1e-4,
        binarize=False,
        step=None,
        **kwargs
    ):
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact

        rays = Rays(origins=rays_o, viewdirs=rays_d)

        if self.training:
            rgb, acc, depth, weights, alphas, n_rendering_samples = render_image_with_occgrid(
                nerf=self.nerf,
                estimator=self.estimator,
                rays=rays,
                near_plane=self.near_plane,
                render_step_size=self.render_step_size,
                render_bkgd=bg_color,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
            )
        else:
            rgb, acc, depth, weights, alphas, n_rendering_samples = render_image_with_occgrid_test(
                max_samples=1024,
                nerf=self.nerf,
                estimator=self.estimator,
                rays=rays,
                near_plane=self.near_plane,
                render_step_size=self.render_step_size,
                render_bkgd=bg_color,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
            )

        results = {}
        results['weights'] = weights
        results['image'] = rgb.view(1, -1, 3)
        results['depth'] = depth.view(1, -1)
        results['weights_sum'] = acc.view(1, -1)

        return results
