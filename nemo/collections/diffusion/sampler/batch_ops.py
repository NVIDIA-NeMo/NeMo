# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from torch import Tensor


def common_broadcast(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """
    Broadcasts two tensors to have the same shape by adding singleton dimensions where necessary.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the two tensors with broadcasted shapes.

    Raises:
        AssertionError: If the dimensions of the tensors do not match at any axis within their common dimensions.
    """
    ndims1 = x.ndim
    ndims2 = y.ndim

    common_ndims = min(ndims1, ndims2)
    for axis in range(common_ndims):
        assert x.shape[axis] == y.shape[axis], "Dimensions not equal at axis {}".format(axis)

    if ndims1 < ndims2:
        x = x.reshape(x.shape + (1,) * (ndims2 - ndims1))
    elif ndims2 < ndims1:
        y = y.reshape(y.shape + (1,) * (ndims1 - ndims2))

    return x, y


def batch_add(x: Tensor, y: Tensor) -> Tensor:
    """
    Adds two tensors element-wise after broadcasting them to a common shape.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        Tensor: The element-wise sum of the input tensors after broadcasting.
    """
    x, y = common_broadcast(x, y)
    return x + y


def batch_mul(x: Tensor, y: Tensor) -> Tensor:
    """
    Multiplies two tensors element-wise after broadcasting them to a common shape.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        Tensor: The element-wise product of the input tensors after broadcasting.
    """
    x, y = common_broadcast(x, y)
    return x * y


def batch_sub(x: Tensor, y: Tensor) -> Tensor:
    """
    Subtracts two tensors element-wise after broadcasting them to a common shape.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        Tensor: The result of element-wise subtraction of the input tensors.
    """
    x, y = common_broadcast(x, y)
    return x - y


def batch_div(x: Tensor, y: Tensor) -> Tensor:
    """
    Divides two tensors element-wise after broadcasting them to a common shape.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        Tensor: The result of element-wise division of `x` by `y` after broadcasting.
    """
    x, y = common_broadcast(x, y)
    return x / y
