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

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from nemo.collections.multimodal.modules.nerf.geometry.dmtet import DeepMarchingTetrahedra
from nemo.collections.multimodal.modules.nerf.geometry.nerf_base import DensityActivationEnum
from nemo.collections.multimodal.modules.nerf.renderers.base_renderer import BaseRenderer


# TODO: self.density_thresh, self.mean_density need a rework, they can be infered at run time
# and shouldn't be loaded from the checkpoint
class NVDiffRastRenderer(BaseRenderer):
    def __init__(self, bound, update_interval, grid_resolution, density_thresh, quartet_file):

        super().__init__(bound, update_interval)

        self.grid_resolution = grid_resolution
        self.density_thresh = density_thresh
        self.quartet_file = quartet_file

        self.cascade = 1 + math.ceil(math.log2(bound))
        density_grid = torch.zeros([self.cascade, self.grid_resolution ** 3])  # [CAS, H * H * H]
        density_bitfield = torch.zeros(
            self.cascade * self.grid_resolution ** 3 // 8, dtype=torch.uint8
        )  # [CAS * H * H * H // 8]
        self.register_buffer('density_grid', density_grid)
        self.register_buffer('density_bitfield', density_bitfield)
        self.mean_density = 0
        self.iter_density = 0

        # load dmtet vertices
        # TODO(ahmadki): hard coded devices
        tets = np.load(quartet_file)
        self.verts = -torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * 2  # covers [-1, 1]
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.tet_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
        self.dmtet = DeepMarchingTetrahedra(device='cuda')

        # vert sdf and deform
        sdf = torch.nn.Parameter(torch.zeros_like(self.verts[..., 0]), requires_grad=True)
        self.register_parameter('sdf', sdf)
        deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', deform)

        edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cuda"
        )  # six edges for each tetrahedron.
        all_edges = self.indices[:, edges].reshape(-1, 2)  # [M * 6, 2]
        all_edges_sorted = torch.sort(all_edges, dim=1)[0]
        self.all_edges = torch.unique(all_edges_sorted, dim=0)

        self.initialized = False  # TODO(ahmadki): not a good approach

        self.glctx = dr.RasterizeCudaContext()

        # TODO(ahmadki): not a good approach
        self.nerf = None
        self.material = None
        self.background = None

    # TODO(ahmkadi): doesn't look good to me !!
    @torch.no_grad()
    def update_step(self, epoch: int, global_step: int, decay: float = 0.95, S: int = 128, **kwargs):
        pass

    @torch.no_grad()
    def init_tet(self):
        # TODO(ahmadki): a better approach would be to have a global nerf representation (mesh) that
        # we can init the tets from. this would work with checkpoints.

        # TODO(ahmadki): a placeholder, but it works for now
        self.mean_density = 300
        density_thresh = min(self.mean_density, self.density_thresh)

        if self.nerf.density_activation == DensityActivationEnum.SOFTPLUS:
            density_thresh = density_thresh * 25

        # Get initial sigma
        sigma = self.nerf.forward_density(positions=self.verts)
        mask = sigma > density_thresh
        valid_verts = self.verts[mask]
        self.tet_scale = valid_verts.abs().amax(dim=0) + 1e-1

        # Scale vertices
        self.verts = self.verts * self.tet_scale

        # get sigma using the scaled vertices
        sigma = self.nerf.forward_density(positions=self.verts)
        self.sdf.data += (sigma - density_thresh).clamp(-1, 1)

    def forward(
        self,
        rays_o,
        rays_d,
        mvp,
        light_d=None,
        ambient_ratio=1.0,
        shading_type=None,
        return_normal_image=False,
        return_vertices=False,
        return_faces=False,
        return_faces_normals=False,
        **kwargs
    ):
        if not self.initialized:
            self.init_tet()
            self.initialized = True
        return self._render(
            rays_o=rays_o,
            rays_d=rays_d,
            mvp=mvp,
            light_d=light_d,
            ambient_ratio=ambient_ratio,
            shading_type=shading_type,
            return_normal_image=return_normal_image,
            return_vertices=return_vertices,
            return_faces=return_faces,
            return_faces_normals=return_faces_normals,
            **kwargs
        )

    def _render(
        self,
        rays_o,
        rays_d,
        mvp,
        light_d=None,
        ambient_ratio=1.0,
        shading_type=None,
        return_normal_image=False,
        return_vertices=False,
        return_faces=False,
        return_faces_normals=False,
        **kwargs
    ):
        # mvp: [B, 4, 4]
        B, H, W, _ = rays_o.shape

        # TODO(ahmadki): move to dataset
        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = rays_o + torch.randn(3, device=rays_o.device)
            light_d = F.normalize(light_d)

        results = {}

        # get mesh
        deform = torch.tanh(self.deform) / self.grid_resolution

        verts, faces = self.dmtet(self.verts + deform, self.sdf, self.indices)

        # get normals
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

        faces = faces.int()

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = F.normalize(face_normals)

        vn = torch.zeros_like(verts)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        vn = torch.where(
            torch.sum(vn * vn, -1, keepdim=True) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )

        # rasterization
        verts_clip = torch.bmm(
            F.pad(verts, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1),
            mvp.permute(0, 2, 1),
        ).float()  # [B, N, 4]
        rast, _ = dr.rasterize(self.glctx, verts_clip, faces, (H, W))

        alpha = (rast[..., 3:] > 0).float()
        xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces)  # [B, H, W, 3]
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = F.normalize(normal)

        xyzs = xyzs.view(-1, 3)
        mask = (rast[..., 3:] > 0).view(-1).detach()

        # do the lighting here since we have normal from mesh now.
        albedo = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            masked_albedo = self.nerf.forward_features(positions=xyzs[mask])
            albedo[mask] = masked_albedo.float()
        albedo = albedo.view(B, H, W, 3)
        fg_color = self.material(
            albedo=albedo, normals=normal, light_d=light_d, ambient_ratio=ambient_ratio, shading_type=shading_type
        )

        fg_color = dr.antialias(fg_color, rast, verts_clip, faces).clamp(0, 1)  # [B, H, W, 3]
        alpha = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1)  # [B, H, W, 1]

        # mix background color
        bg_color = self.background(rays_d=rays_d)  # [N, 3]

        depth = rast[:, :, :, [2]]  # [B, H, W]
        color = fg_color + (1 - alpha) * bg_color

        results['depth'] = depth
        results['image'] = color
        if return_normal_image:
            results['normal_image'] = dr.antialias((normal + 1) / 2, rast, verts_clip, faces).clamp(
                0, 1
            )  # [B, H, W, 3]
        if return_vertices:
            results['vertices'] = verts
        if return_faces:
            results['faces'] = faces
        if return_faces_normals:
            results['face_normals'] = face_normals
        return results
