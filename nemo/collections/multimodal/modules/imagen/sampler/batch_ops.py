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
# Functions for performing operations with broadcasting to the right axis
#
# Example
# input1: tensor of size (N1, N2)
# input2: tensor of size (N1, N2, N3, N4)
# batch_mul(input1, input2) = input1[:, :, None, None] * input2
#
# If the common dimensions don't match, we raise an assertion error.


def common_broadcast(x, y):
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], 'Dimensions not equal at axis {}'.format(axis)

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_add(x, y):
    x, y = common_broadcast(x, y)
    return x + y


def batch_mul(x, y):
    x, y = common_broadcast(x, y)
    return x * y


def batch_sub(x, y):
    x, y = common_broadcast(x, y)
    return x - y


def batch_div(x, y):
    x, y = common_broadcast(x, y)
    return x / y
