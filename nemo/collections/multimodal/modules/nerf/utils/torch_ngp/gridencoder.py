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

import _gridencoder as _backend
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}


class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        inputs,
        embeddings,
        offsets,
        per_level_scale,
        base_resolution,
        calc_grad_inputs=False,
        gridtype=0,
        align_corners=False,
        interpolation=0,
        max_level=None,
    ):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape  # batch size, coord dim
        L = offsets.shape[0] - 1  # level
        C = embeddings.shape[1]  # embedding dim for each level
        S = np.log2(per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution  # base resolution

        max_level = L if max_level is None else max(min(int(math.ceil(max_level * L)), L), 1)

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        # zero init if we only calculate partial levels
        if max_level < L:
            outputs.zero_()

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
            if max_level < L:
                dy_dx.zero_()
        else:
            dy_dx = None

        _backend.grid_encode_forward(
            inputs,
            embeddings,
            offsets,
            outputs,
            B,
            D,
            C,
            L,
            max_level,
            S,
            H,
            dy_dx,
            gridtype,
            align_corners,
            interpolation,
        )

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation, max_level]
        ctx.align_corners = align_corners

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation, max_level = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.grid_encode_backward(
            grad,
            inputs,
            embeddings,
            offsets,
            grad_embeddings,
            B,
            D,
            C,
            L,
            max_level,
            S,
            H,
            dy_dx,
            grad_inputs,
            gridtype,
            align_corners,
            interpolation,
        )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        num_levels=16,
        level_dim=2,
        per_level_scale=2,
        base_resolution=16,
        log2_hashmap_size=19,
        desired_resolution=None,
        gridtype='hash',
        align_corners=False,
        interpolation='linear',
    ):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim  # coord dims, 2 or 3
        self.num_levels = num_levels  # num levels, each level multiply resolution by 2
        self.level_dim = level_dim  # encode channels per level
        self.per_level_scale = per_level_scale  # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype]  # "tiled" or "hash"
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]  # "linear" or "smoothstep"
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution) ** input_dim)  # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)

        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, bound=1, max_level=None):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # max_level: only calculate first max_level levels (None will use all levels)
        # return: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound)  # map to [0, 1]

        # print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(
            inputs,
            self.embeddings,
            self.offsets,
            self.per_level_scale,
            self.base_resolution,
            inputs.requires_grad,
            self.gridtype_id,
            self.align_corners,
            self.interp_id,
            max_level,
        )
        outputs = outputs.view(prefix_shape + [self.output_dim])

        # print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.

        D = self.input_dim
        C = self.embeddings.shape[1]  # embedding dim for each level
        L = self.offsets.shape[0] - 1  # level
        S = np.log2(self.per_level_scale)  # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = self.base_resolution  # base resolution

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound)  # map to [0, 1]
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]

        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')

        _backend.grad_total_variation(
            inputs,
            self.embeddings,
            self.embeddings.grad,
            self.offsets,
            weight,
            B,
            D,
            C,
            L,
            S,
            H,
            self.gridtype_id,
            self.align_corners,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def grad_weight_decay(self, weight=0.1):
        # level-wise meaned weight decay (ref: zip-nerf)

        B = self.embeddings.shape[0]  # size of embedding
        C = self.embeddings.shape[1]  # embedding dim for each level
        L = self.offsets.shape[0] - 1  # level

        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')

        _backend.grad_weight_decay(self.embeddings, self.embeddings.grad, self.offsets, weight, B, C, L)
