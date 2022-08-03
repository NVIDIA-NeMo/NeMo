# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Most of the code here has been copied from:
# https://github.com/microsoft/mup

'''
Initializer functions mirroring those of `torch.nn.init`. They serve as
drop-in replacements after the user has called `set_base_shapes` on their
model.

All of the initializers here are designed to 1) behave exactly the same
as the torch versions when the model shapes are equal to their base shapes,
and 2) to scale with width correctly (according to μP), when the model shapes
differ from the base shapes. In general, this means deviating from the
torch version behaviors.
'''
import math
import warnings

import torch
from torch.nn.init import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    _no_grad_fill_,
    _no_grad_normal_,
    _no_grad_uniform_,
    calculate_gain,
)


def constant_std_init_(tensor, sampler_):
    assert hasattr(tensor, 'infshape'), 'Please call set_base_shapes(...)'
    if tensor.infshape.ninf() <= 1:
        sampler_(tensor)
    elif tensor.infshape.ninf() == 2:
        sampler_(tensor, scale=tensor.infshape.width_mult() ** -0.5)
    else:
        raise NotImplementedError()
    return tensor


def uniform_(tensor, a=0, b=1):
    '''Drop-in replacement of `torch.nn.init.uniform_`.
    Note:
        -  if using this function, ensure `a` and `b` do not depend on fan-in,
           fan-out, or other notions of width, e.g. if a = 0, b = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    assert hasattr(tensor, 'infshape'), 'Please call set_base_shapes(...)'
    if a != -b:
        assert tensor.infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'

    def sampler_(tensor, scale=1):
        _no_grad_uniform_(tensor, a * scale, b * scale)

    return constant_std_init_(tensor, sampler_)


def normal_(tensor, mean=0, std=1):
    '''Drop-in replacement of `torch.nn.init.normal_`.
    Note:
        -  if using this function, ensure `mean` and `std` do not depend on
           fan-in, fan-out, or other notions of width, e.g. if mean = 0, std =
           1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    if mean != 0:
        assert tensor.infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'

    def sampler_(tensor, scale=1):
        _no_grad_normal_(tensor, mean=mean * scale, std=std * scale)

    return constant_std_init_(tensor, sampler_)


def ones_(tensor):
    '''Same as `torch.nn.init.ones_`.
    Note:
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    assert tensor.infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'

    def sampler_(tensor, scale=1):
        _no_grad_fill_(tensor, scale)

    return constant_std_init_(tensor, sampler_)


def eye_(tensor):
    '''Same as `torch.nn.init.eye_`.
    Note:
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    assert tensor.infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'
    return torch.nn.init.eye_(tensor)


def _inf_fan_adjust_xavier(scale, tensor):
    fan_out, fan_in = tensor.infshape[:2]
    # following are needed to accomodate SP models where all infshapes are finite so base_dims are Nones
    fan_out_base_dim = fan_out.base_dim or fan_out.dim
    fan_in_base_dim = fan_in.base_dim or fan_in.dim
    scale *= math.sqrt((fan_out.dim + fan_in.dim) / (fan_out_base_dim + fan_in_base_dim))
    if tensor.infshape.ninf() <= 1:
        # should have fixed scale
        pass
    elif tensor.infshape.ninf() == 2:
        # should scale like fanin
        assert fan_out.isinf() and fan_in.isinf()
        scale /= math.sqrt(fan_in.width_mult())
    else:
        raise NotImplementedError('can only handle 2 inf dimensions currently')
    return scale


def xavier_uniform_(tensor, gain=1.0):
    '''Drop-in replacement of `torch.nn.init.xavier_uniform_`.
    Note:
        -  if using this function, ensure `gain` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if gain = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    std = _inf_fan_adjust_xavier(std, tensor)
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(tensor, gain=1.0):
    '''Drop-in replacement of `torch.nn.init.xavier_normal_`.
    Note:
        -  if using this function, ensure `gain` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if gain = 1.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    std = _inf_fan_adjust_xavier(std, tensor)
    return _no_grad_normal_(tensor, 0.0, std)


def _inf_fan_adjust_kaiming(scale, tensor, mode):
    fan_out, fan_in = tensor.infshape[:2]
    if tensor.infshape.ninf() == 0:
        return scale
    elif tensor.infshape.ninf() == 1:
        # should have fixed scale
        if mode == 'fan_in' and fan_in.isinf():
            scale *= fan_in.width_mult() ** 0.5
        elif mode == 'fan_out' and fan_out.isinf():
            scale *= fan_out.width_mult() ** 0.5
    elif tensor.infshape.ninf() == 2:
        # should scale like fanin
        assert fan_out.isinf() and fan_in.isinf()
        if mode == 'fan_out':
            scale *= math.sqrt(fan_out.width_mult() / fan_in.width_mult())
    else:
        raise NotImplementedError('can only handle <=2 inf dimensions currently')
    return scale


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''Drop-in replacement of `torch.nn.init.kaiming_normal_`.
    Note:
        -  if using this function, ensure `a` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if a = 0.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = _inf_fan_adjust_kaiming(gain / math.sqrt(fan), tensor, mode)
    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''Drop-in replacement of `torch.nn.init.kaiming_uniform_`.
    Note:
        -  if using this function, ensure `a` does not depend on fan-in,
           fan-out, or other notions of width, e.g. if a = 0.
        - `tensor` should have `infshape` attribute set by `set_base_shapes`.
    '''
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = _inf_fan_adjust_kaiming(gain / math.sqrt(fan), tensor, mode)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


try:
    from torch.nn.init import _no_grad_trunc_normal_

    def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
        '''Drop-in replacement of `torch.nn.init.trunc_normal_`.
        Note:
            -  if using this function, ensure `mean`, `std`, `a`, `b` do not
               depend on fan-in, fan-out, or other notions of width, e.g. if
               mean = 0, std = 1, a = -2, b = 2.
            - `tensor` should have `infshape` attribute set by
              `set_base_shapes`.
        '''
        if mean != 0 or a != -b:
            assert tensor.infshape.ninf() == 1, 'Sampler for (inf, inf) tensors should have mean 0'

        def sampler_(tensor, scale=1):
            _no_grad_trunc_normal_(tensor, mean=mean * scale, std=std * scale, a=a * scale, b=b * scale)

        return constant_std_init_(tensor, sampler_)


except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'Failed to import _no_grad_trunc_normal_ from torch.nn.init; '
        'you might be running an older version of torch. trunc_normal_ will not work.'
    )

    def trunc_normal_(tensor, mean=0, std=1, a=-2, b=2):
        warnings.warn('Please upgrade your Pytorch version before using truncated normal.')
