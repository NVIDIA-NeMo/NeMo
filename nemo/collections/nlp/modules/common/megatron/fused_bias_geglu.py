# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import bias_gelu, bias_gelu_back

try:
    from apex._autocast_utils import _cast_if_autocast_enabled

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


@torch.jit.script
def bias_geglu(bias, y, bias_2, y_2):
    x_2 = bias_2 + y_2
    return bias_gelu(bias, y) * x_2


@torch.jit.script
def bias_geglu_back(g, bias, y, bias_2, y_2):
    x_2 = bias_2 + y_2
    return bias_gelu_back(g, bias, y) * x_2, bias_gelu(bias, y) * g


class GeGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias and bias_2 are optional arguments
    def forward(ctx, input, bias, input_2, bias_2):
        ctx.save_for_backward(input, bias, input_2, bias_2)
        return bias_geglu(bias, input, bias_2, input_2)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, input_2, bias_2 = ctx.saved_tensors
        tmp, tmp2 = bias_geglu_back(grad_output, bias, input, bias_2, input_2)
        return tmp, tmp, tmp2, tmp2


def fused_bias_geglu(input, bias, input_2, bias_2):
    args = _cast_if_autocast_enabled(input, bias, input_2, bias_2)
    with torch.cuda.amp.autocast(enabled=False):
        return GeGLUFunction.apply(*args)
