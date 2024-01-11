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
import _shencoder as _backend
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _sh_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # force float32 for better precision
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape  # batch size, coord dim
        output_dim = degree ** 2

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = None

        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)

        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        inputs, dy_dx = ctx.saved_tensors

        if dy_dx is not None:
            grad = grad.contiguous()
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None


sh_encode = _sh_encoder.apply


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super().__init__()

        self.input_dim = input_dim  # coord dims, must be 3
        self.degree = degree  # 0 ~ 4
        self.output_dim = degree ** 2

        assert self.input_dim == 3, "SH encoder only support input dim == 3"
        assert self.degree > 0 and self.degree <= 8, "SH encoder only supports degree in [1, 8]"

    def __repr__(self):
        return f"SHEncoder: input_dim={self.input_dim} degree={self.degree}"

    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., degree^2]

        inputs = inputs / size  # [-1, 1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)

        outputs = sh_encode(inputs, self.degree, inputs.requires_grad)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        return outputs
