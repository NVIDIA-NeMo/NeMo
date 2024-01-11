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
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

# lazy building:
# `import raymarching` will not immediately build the extension, only if you actually call any functions.

BACKEND = None


def get_backend():
    global BACKEND

    if BACKEND is None:
        try:
            import _raymarching as _backend
        except ImportError:
            from .backend import _backend

        BACKEND = _backend

    return BACKEND


# ----------------------------------------
# utils
# ----------------------------------------


class _near_far_from_aabb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda:
            rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # num rays

        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)

        get_backend().near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)

        return nears, fars


near_far_from_aabb = _near_far_from_aabb.apply


class _sph_from_ray(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, radius):
        ''' sph_from_ray, CUDA implementation
        get spherical coordinate on the background sphere from rays.
        Assume rays_o are inside the Sphere(radius).
        Args:
            rays_o: [N, 3]
            rays_d: [N, 3]
            radius: scalar, float
        Return:
            coords: [N, 2], in [-1, 1], theta and phi on a sphere. (further-surface)
        '''
        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda:
            rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # num rays

        coords = torch.empty(N, 2, dtype=rays_o.dtype, device=rays_o.device)

        get_backend().sph_from_ray(rays_o, rays_d, radius, N, coords)

        return coords


sph_from_ray = _sph_from_ray.apply


class _morton3D(Function):
    @staticmethod
    def forward(ctx, coords):
        ''' morton3D, CUDA implementation
        Args:
            coords: [N, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...)
            TODO: check if the coord range is valid! (current 128 is safe)
        Returns:
            indices: [N], int32, in [0, 128^3)

        '''
        if not coords.is_cuda:
            coords = coords.cuda()

        N = coords.shape[0]

        indices = torch.empty(N, dtype=torch.int32, device=coords.device)

        get_backend().morton3D(coords.int(), N, indices)

        return indices


morton3D = _morton3D.apply


class _morton3D_invert(Function):
    @staticmethod
    def forward(ctx, indices):
        ''' morton3D_invert, CUDA implementation
        Args:
            indices: [N], int32, in [0, 128^3)
        Returns:
            coords: [N, 3], int32, in [0, 128)

        '''
        if not indices.is_cuda:
            indices = indices.cuda()

        N = indices.shape[0]

        coords = torch.empty(N, 3, dtype=torch.int32, device=indices.device)

        get_backend().morton3D_invert(indices.int(), N, coords)

        return coords


morton3D_invert = _morton3D_invert.apply


class _packbits(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, thresh, bitfield=None):
        ''' packbits, CUDA implementation
        Pack up the density grid into a bit field to accelerate ray marching.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [C, H * H * H / 8]
        '''
        if not grid.is_cuda:
            grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        N = C * H3 // 8

        if bitfield is None:
            bitfield = torch.empty(N, dtype=torch.uint8, device=grid.device)

        get_backend().packbits(grid, N, thresh, bitfield)

        return bitfield


packbits = _packbits.apply


class _flatten_rays(Function):
    @staticmethod
    def forward(ctx, rays, M):
        ''' flatten rays
        Args:
            rays: [N, 2], all rays' (point_offset, point_count),
            M: scalar, int, count of points (we cannot get this info from rays unfortunately...)
        Returns:
            res: [M], flattened ray index.
        '''
        if not rays.is_cuda:
            rays = rays.cuda()
        rays = rays.contiguous()

        N = rays.shape[0]

        res = torch.zeros(M, dtype=torch.int, device=rays.device)

        get_backend().flatten_rays(rays, N, M, res)

        return res


flatten_rays = _flatten_rays.apply

# ----------------------------------------
# train functions
# ----------------------------------------


class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        rays_o,
        rays_d,
        bound,
        density_bitfield,
        C,
        H,
        nears,
        fars,
        perturb=False,
        dt_gamma=0,
        max_steps=1024,
        contract=False,
    ):
        ''' march rays to generate points (forward only)
        Args:
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            step_counter: int32, (2), used to count the actual number of generated points.
            mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
            perturb: bool
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs.
            ts: float, [M, 2], all generated points' ts.
            rays: int32, [N, 2], all rays' (point_offset, point_count), e.g., xyzs[rays[i, 0]:(rays[i, 0] + rays[i, 1])] --> points belonging to rays[i, 0]
        '''

        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda:
            rays_d = rays_d.cuda()
        if not density_bitfield.is_cuda:
            density_bitfield = density_bitfield.cuda()

        rays_o = rays_o.float().contiguous().view(-1, 3)
        rays_d = rays_d.float().contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()

        N = rays_o.shape[0]  # num rays

        step_counter = torch.zeros(1, dtype=torch.int32, device=rays_o.device)  # point counter, ray counter

        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)

        # first pass: write rays, get total number of points M to render
        rays = torch.empty(N, 2, dtype=torch.int32, device=rays_o.device)  # id, offset, num_steps
        get_backend().march_rays_train(
            rays_o,
            rays_d,
            density_bitfield,
            bound,
            contract,
            dt_gamma,
            max_steps,
            N,
            C,
            H,
            nears,
            fars,
            None,
            None,
            None,
            rays,
            step_counter,
            noises,
        )

        # allocate based on M
        M = step_counter.item()
        # print(M, N)
        # print(rays[:, 0].max())

        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        ts = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)

        # second pass: write outputs
        get_backend().march_rays_train(
            rays_o,
            rays_d,
            density_bitfield,
            bound,
            contract,
            dt_gamma,
            max_steps,
            N,
            C,
            H,
            nears,
            fars,
            xyzs,
            dirs,
            ts,
            rays,
            step_counter,
            noises,
        )

        return xyzs, dirs, ts, rays


march_rays_train = _march_rays_train.apply


class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, ts, rays, T_thresh=1e-4, binarize=False):
        ''' composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            ts: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights: float, [M]
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''

        sigmas = sigmas.float().contiguous()
        rgbs = rgbs.float().contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights = torch.zeros(M, dtype=sigmas.dtype, device=sigmas.device)  # may leave unmodified, so init with 0
        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)

        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)

        get_backend().composite_rays_train_forward(
            sigmas, rgbs, ts, rays, M, N, T_thresh, binarize, weights, weights_sum, depth, image
        )

        ctx.save_for_backward(sigmas, rgbs, ts, rays, weights_sum, depth, image)
        ctx.dims = [M, N, T_thresh, binarize]

        return weights, weights_sum, depth, image

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights, grad_weights_sum, grad_depth, grad_image):

        grad_weights = grad_weights.contiguous()
        grad_weights_sum = grad_weights_sum.contiguous()
        grad_depth = grad_depth.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, ts, rays, weights_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh, binarize = ctx.dims

        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)

        get_backend().composite_rays_train_backward(
            grad_weights,
            grad_weights_sum,
            grad_depth,
            grad_image,
            sigmas,
            rgbs,
            ts,
            rays,
            weights_sum,
            depth,
            image,
            M,
            N,
            T_thresh,
            binarize,
            grad_sigmas,
            grad_rgbs,
        )

        return grad_sigmas, grad_rgbs, None, None, None, None


composite_rays_train = _composite_rays_train.apply

# ----------------------------------------
# infer functions
# ----------------------------------------


class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx,
        n_alive,
        n_step,
        rays_alive,
        rays_t,
        rays_o,
        rays_d,
        bound,
        density_bitfield,
        C,
        H,
        near,
        far,
        perturb=False,
        dt_gamma=0,
        max_steps=1024,
        contract=False,
    ):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            perturb: bool/int, int > 0 is used as the random seed.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            ts: float, [n_alive * n_step, 2], all generated points' ts
        '''

        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda:
            rays_d = rays_d.cuda()

        rays_o = rays_o.float().contiguous().view(-1, 3)
        rays_d = rays_d.float().contiguous().view(-1, 3)

        M = n_alive * n_step

        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        ts = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)  # 2 vals, one for rgb, one for depth

        if perturb:
            # torch.manual_seed(perturb) # test_gui uses spp index as seed
            noises = torch.rand(n_alive, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(n_alive, dtype=rays_o.dtype, device=rays_o.device)

        get_backend().march_rays(
            n_alive,
            n_step,
            rays_alive,
            rays_t,
            rays_o,
            rays_d,
            bound,
            contract,
            dt_gamma,
            max_steps,
            C,
            H,
            density_bitfield,
            near,
            far,
            xyzs,
            dirs,
            ts,
            noises,
        )

        return xyzs, dirs, ts


march_rays = _march_rays.apply


class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # need to cast sigmas & rgbs to float
    def forward(
        ctx,
        n_alive,
        n_step,
        rays_alive,
        rays_t,
        sigmas,
        rgbs,
        ts,
        weights_sum,
        depth,
        image,
        T_thresh=1e-2,
        binarize=False,
    ):
        ''' composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            ts: float, [n_alive * n_step, 2]
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        '''
        sigmas = sigmas.float().contiguous()
        rgbs = rgbs.float().contiguous()
        get_backend().composite_rays(
            n_alive, n_step, T_thresh, binarize, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image
        )
        return tuple()


composite_rays = _composite_rays.apply
