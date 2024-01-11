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
import torch.nn.functional as F

import nemo.collections.multimodal.modules.nerf.utils.torch_ngp.raymarching as raymarching
from nemo.collections.multimodal.modules.nerf.materials.materials_base import ShadingEnum
from nemo.collections.multimodal.modules.nerf.renderers.base_renderer import BaseRenderer


class TorchNGPVolumeRenderer(BaseRenderer):
    def __init__(self, bound, update_interval, grid_resolution, density_thresh, max_steps, dt_gamma):

        super().__init__(bound, update_interval)

        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_resolution = grid_resolution
        self.density_thresh = density_thresh
        self.dt_gamma = dt_gamma
        self.max_steps = max_steps

        # density grid
        # TODO(ahmadki): needs rework
        density_grid = torch.zeros([self.cascade, self.grid_resolution ** 3])  # [CAS, H * H * H]
        density_bitfield = torch.zeros(
            self.cascade * self.grid_resolution ** 3 // 8, dtype=torch.uint8
        )  # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0

        # TODO(ahmadki): needs rework
        self.nerf = None
        self.material = None
        self.background = None

    @torch.no_grad()
    def update_step(self, epoch: int, global_step: int, decay: float = 0.95, S: int = 128, **kwargs):
        if global_step % self.update_interval != 0:
            return

        ### update density grid
        tmp_grid = -torch.ones_like(self.density_grid)

        X = torch.arange(self.grid_resolution, dtype=torch.int32, device=self.aabb.device).split(S)
        Y = torch.arange(self.grid_resolution, dtype=torch.int32, device=self.aabb.device).split(S)
        Z = torch.arange(self.grid_resolution, dtype=torch.int32, device=self.aabb.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    coords = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1
                    )  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    xyzs = 2 * coords.float() / (self.grid_resolution - 1) - 1  # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_resolution = bound / self.grid_resolution
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_resolution)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_resolution
                        # query density
                        density = self.nerf.forward_density(cas_xyzs).reshape(-1).detach()
                        # assign
                        tmp_grid[cas, indices] = density
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

    def forward(
        self,
        rays_o,
        rays_d,
        light_d=None,
        ambient_ratio=1.0,
        shading_type=None,
        return_normal_image=False,
        return_normal_perturb=False,
        **kwargs
    ):
        return self._render(
            rays_o=rays_o,
            rays_d=rays_d,
            light_d=light_d,
            ambient_ratio=ambient_ratio,
            shading_type=shading_type,
            return_normal_image=return_normal_image,
            return_normal_perturb=return_normal_perturb,
            **kwargs
        )

    # TODO(ahmadki): return_normal_image is always False ?
    def _render(
        self,
        rays_o,
        rays_d,
        light_d=None,
        ambient_ratio=1.0,
        shading_type=None,
        return_normal_image=False,
        return_normal_perturb=False,
        perturb=False,
        T_thresh=1e-4,
        binarize=False,
        **kwargs
    ):
        # rays_o, rays_d: [B, H, W, 3]
        B, H, W, _ = rays_o.shape

        # group all rays into a single batch
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        num_rays = rays_o.shape[0]  # num_rays = B * H * W

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb)

        # random sample light_d if not provided
        # TODO(ahmadki): move to dataset
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = rays_o + torch.randn(3, device=rays_o.device)
            light_d = F.normalize(light_d)

        normal_image = None
        normals_perturb = None
        weights = None

        if self.training:
            positions, dirs, ts, rays = raymarching.march_rays_train(
                rays_o,
                rays_d,
                self.bound,
                self.density_bitfield,
                self.cascade,
                self.grid_resolution,
                nears,
                fars,
                perturb,
                self.dt_gamma,
                self.max_steps,
            )
            dirs = F.normalize(dirs)

            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, positions.shape[0]).long()
                light_d = light_d[flatten_rays]

            return_normal = (shading_type is not None) or return_normal_image
            sigmas, albedo, normals = self.nerf(positions=positions, return_normal=return_normal)

            fg_color = self.material(
                albedo=albedo, normals=normals, light_d=light_d, ambient_ratio=ambient_ratio, shading_type=shading_type
            )

            weights, opacity, depth, image = raymarching.composite_rays_train(
                sigmas, fg_color, ts, rays, T_thresh, binarize
            )

            if return_normal_image and normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(
                    sigmas.detach(), (normals + 1) / 2, ts, rays, T_thresh, binarize
                )

            if return_normal_perturb:
                perturb_positions = positions + torch.randn_like(positions) * 1e-2
                normals_perturb = self.normal(positions=perturb_positions)

        else:
            # allocate tensors
            image = torch.zeros(num_rays, 3, device=rays_o.device)
            depth = torch.zeros(num_rays, device=rays_o.device)
            opacity = torch.zeros(num_rays, device=rays_o.device)

            n_alive = num_rays
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=rays_o.device)
            rays_t = nears.clone()

            step = 0

            while step < self.max_steps:  # hard coded max step
                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(num_rays // n_alive, 8), 1)

                positions, dirs, ts = raymarching.march_rays(
                    n_alive,
                    n_step,
                    rays_alive,
                    rays_t,
                    rays_o,
                    rays_d,
                    self.bound,
                    self.density_bitfield,
                    self.cascade,
                    self.grid_resolution,
                    nears,
                    fars,
                    perturb if step == 0 else False,
                    self.dt_gamma,
                    self.max_steps,
                )
                dirs = F.normalize(dirs)

                return_normal = shading_type not in [None, ShadingEnum.TEXTURELESS]
                sigmas, albedo, normals = self.nerf(positions=positions, return_normal=return_normal)

                fg_color = self.material(
                    albedo=albedo,
                    normals=normals,
                    light_d=light_d,
                    ambient_ratio=ambient_ratio,
                    shading_type=shading_type,
                )
                raymarching.composite_rays(
                    n_alive,
                    n_step,
                    rays_alive,
                    rays_t,
                    sigmas,
                    fg_color,
                    ts,
                    opacity,
                    depth,
                    image,
                    T_thresh,
                    binarize,
                )

                # TODO(ahmadki): add optoin to return normal_image, like in training

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

        # mix background color
        bg_color = self.background(rays_d)  # [N, 3]
        image = image + (1 - opacity).unsqueeze(-1) * bg_color

        results = {
            "image": image.view(B, H, W, 3),
            "depth": depth.view(B, H, W, 1),
            "opacity": opacity.view(B, H, W, 1),
            "dirs": dirs,
        }
        if normals is not None:
            results["normals"] = normals
        if weights is not None:
            results["weights"] = weights
        if normal_image is not None:
            results["normal_image"] = normal_image.view(B, H, W, 3)
        if normals_perturb is not None:
            results["normal_perturb"] = normals_perturb

        return results
