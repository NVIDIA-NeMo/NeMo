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

# The MIT License (MIT)
# Copyright (c) 2020, nicolas deutschmann
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch
import torch.nn.functional as F


def piecewise_linear_transform(x, q_tilde, compute_jacobian=True, outlier_passthru=True):
    """Apply an element-wise piecewise-linear transformation to some variables

    Args:
        x : torch.Tensor
            a tensor with shape (N,k) where N is the batch dimension while k is the dimension of the variable space. This variable span the k-dimensional unit
        hypercube

        q_tilde: torch.Tensor
                is a tensor with shape (N,k,b) where b is the number of bins.
                This contains the un-normalized heights of the bins of the piecewise-constant PDF for dimension k,
                i.e. q_tilde lives in all of R and we don't impose a constraint on their sum yet.
                Normalization is imposed in this function using softmax.

        compute_jacobian : bool, optional
                            determines whether the jacobian should be compute or None is returned

    Returns:
        tuple of torch.Tensor
            pair `(y,h)`.
            - `y` is a tensor with shape (N,k) living in the k-dimensional unit hypercube
            - `j` is the jacobian of the transformation with shape (N,) if compute_jacobian==True, else None.
    """
    logj = None

    third_dimension_softmax = torch.nn.Softmax(dim=2)

    # Compute the bin width w
    N, k, b = q_tilde.shape
    Nx, kx = x.shape
    assert N == Nx and k == kx, "Shape mismatch"

    w = 1.0 / b

    # Compute normalized bin heights with softmax function on bin dimension
    q = 1.0 / w * third_dimension_softmax(q_tilde)
    # x is in the mx-th bin: x \in [0,1],
    # mx \in [[0,b-1]], so we clamp away the case x == 1
    mx = torch.clamp(torch.floor(b * x), 0, b - 1).to(torch.long)
    # Need special error handling because trying to index with mx
    # if it contains nans will lock the GPU. (device-side assert triggered)
    if torch.any(torch.isnan(mx)).item() or torch.any(mx < 0) or torch.any(mx >= b):
        raise AvertedCUDARuntimeError("NaN detected in PWLinear bin indexing")

    # We compute the output variable in-place
    out = x - mx * w  # alpha (element of [0.,w], the position of x in its bin

    # Multiply by the slope
    # q has shape (N,k,b), mxu = mx.unsqueeze(-1) has shape (N,k) with entries that are a b-index
    # gather defines slope[i, j, k] = q[i, j, mxu[i, j, k]] with k taking only 0 as a value
    # i.e. we say slope[i, j] = q[i, j, mx [i, j]]
    slopes = torch.gather(q, 2, mx.unsqueeze(-1)).squeeze(-1)
    out = out * slopes
    # The jacobian is the product of the slopes in all dimensions

    # Compute the integral over the left-bins.
    # 1. Compute all integrals: cumulative sum of bin height * bin weight.
    # We want that index i contains the cumsum *strictly to the left* so we shift by 1
    # leaving the first entry null, which is achieved with a roll and assignment
    q_left_integrals = torch.roll(torch.cumsum(q, 2) * w, 1, 2)
    q_left_integrals[:, :, 0] = 0

    # 2. Access the correct index to get the left integral of each point and add it to our transformation
    out = out + torch.gather(q_left_integrals, 2, mx.unsqueeze(-1)).squeeze(-1)

    # Regularization: points must be strictly within the unit hypercube
    # Use the dtype information from pytorch
    eps = torch.finfo(out.dtype).eps
    out = out.clamp(min=eps, max=1.0 - eps)
    oob_mask = torch.logical_or(x < 0.0, x > 1.0).detach().float()
    if outlier_passthru:
        out = out * (1 - oob_mask) + x * oob_mask
        slopes = slopes * (1 - oob_mask) + oob_mask

    if compute_jacobian:
        # logj = torch.log(torch.prod(slopes.float(), 1))
        logj = torch.sum(torch.log(slopes), 1)
    del slopes

    return out, logj


def piecewise_linear_inverse_transform(y, q_tilde, compute_jacobian=True, outlier_passthru=True):
    """
    Apply inverse of an element-wise piecewise-linear transformation to some
    variables

    Args:
        y : torch.Tensor
            a tensor with shape (N,k) where N is the batch dimension while k is the dimension of the variable space. This variable span the k-dimensional unit hypercube

        q_tilde: torch.Tensor
                is a tensor with shape (N,k,b) where b is the number of bins. This contains the un-normalized heights of the bins of the piecewise-constant PDF for dimension k, i.e. q_tilde lives in all of R and we don't impose a constraint on their sum yet. Normalization is imposed in this function using softmax.

        compute_jacobian : bool, optional
                            determines whether the jacobian should be compute or None is returned

    Returns:
        tuple of torch.Tensor
            pair `(x,h)`.
            - `x` is a tensor with shape (N,k) living in the k-dimensional unit hypercube
            - `j` is the jacobian of the transformation with shape (N,) if compute_jacobian==True, else None.
    """

    third_dimension_softmax = torch.nn.Softmax(dim=2)
    # Compute the bin width w
    N, k, b = q_tilde.shape
    Ny, ky = y.shape
    assert N == Ny and k == ky, "Shape mismatch"

    w = 1.0 / b

    # Compute normalized bin heights with softmax function on the bin dimension
    q = 1.0 / w * third_dimension_softmax(q_tilde)

    # Compute the integral over the left-bins in the forward transform.
    # 1. Compute all integrals: cumulative sum of bin height * bin weight.
    # We want that index i contains the cumsum *strictly to the left*,
    # so we shift by 1 leaving the first entry null,
    # which is achieved with a roll and assignment
    q_left_integrals = torch.roll(torch.cumsum(q.float(), 2) * w, 1, 2)
    q_left_integrals[:, :, 0] = 0

    # Find which bin each y belongs to by finding the smallest bin such that
    # y - q_left_integral is positive

    edges = (y.unsqueeze(-1) - q_left_integrals).detach()
    # y and q_left_integrals are between 0 and 1,
    # so that their difference is at most 1.
    # By setting the negative values to 2., we know that the
    # smallest value left is the smallest positive
    edges[edges < 0] = 2.0
    edges = torch.clamp(torch.argmin(edges, dim=2), 0, b - 1).to(torch.long)

    # Need special error handling because trying to index with mx
    # if it contains nans will lock the GPU. (device-side assert triggered)
    if torch.any(torch.isnan(edges)).item() or torch.any(edges < 0) or torch.any(edges >= b):
        raise AvertedCUDARuntimeError("NaN detected in PWLinear bin indexing")

    # Gather the left integrals at each edge. See comment about gathering in q_left_integrals
    # for the unsqueeze
    q_left_integrals = q_left_integrals.gather(2, edges.unsqueeze(-1)).squeeze(-1)

    # Gather the slope at each edge.
    q = q.gather(2, edges.unsqueeze(-1)).squeeze(-1)

    # Build the output
    x = (y - q_left_integrals) / q + edges * w

    # Regularization: points must be strictly within the unit hypercube
    # Use the dtype information from pytorch
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(min=eps, max=1.0 - eps)
    oob_mask = torch.logical_or(y < 0.0, y > 1.0).detach().float()
    if outlier_passthru:
        x = x * (1 - oob_mask) + y * oob_mask
        q = q * (1 - oob_mask) + oob_mask

    # Prepare the jacobian
    logj = None
    if compute_jacobian:
        # logj = - torch.log(torch.prod(q, 1))
        logj = -torch.sum(torch.log(q.float()), 1)
    return x.detach(), logj


def unbounded_piecewise_quadratic_transform(x, w_tilde, v_tilde, upper=1, lower=0, inverse=False):
    assert upper > lower
    _range = upper - lower
    inside_interval_mask = (x >= lower) & (x < upper)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(x)
    log_j = torch.zeros_like(x)

    outputs[outside_interval_mask] = x[outside_interval_mask]
    log_j[outside_interval_mask] = 0

    output, _log_j = piecewise_quadratic_transform(
        (x[inside_interval_mask] - lower) / _range,
        w_tilde[inside_interval_mask, :],
        v_tilde[inside_interval_mask, :],
        inverse=inverse,
    )
    outputs[inside_interval_mask] = output * _range + lower
    if not inverse:
        # the before and after transformation cancel out, so the log_j would be just as it is.
        log_j[inside_interval_mask] = _log_j
    else:
        log_j = None
    return outputs, log_j


def weighted_softmax(v, w):
    # to avoid NaN...
    v = v - torch.max(v, dim=-1, keepdim=True)[0]
    v = torch.exp(v) + 1e-8  # to avoid NaN...
    v_sum = torch.sum((v[..., :-1] + v[..., 1:]) / 2 * w, dim=-1, keepdim=True)
    return v / v_sum


def piecewise_quadratic_transform(x, w_tilde, v_tilde, inverse=False):
    """Element-wise piecewise-quadratic transformation
    Args:
        x : torch.Tensor
            *, The variable spans the D-dim unit hypercube ([0,1))
        w_tilde : torch.Tensor
            * x K defined in the paper
        v_tilde : torch.Tensor
            * x (K+1) defined in the paper
        inverse : bool
            forward or inverse
    Returns:
        c : torch.Tensor
            *, transformed value
        log_j : torch.Tensor
            *, log determinant of the Jacobian matrix
    """
    w = torch.softmax(w_tilde, dim=-1)
    v = weighted_softmax(v_tilde, w)
    w_cumsum = torch.cumsum(w, dim=-1)
    # force sum = 1
    w_cumsum[..., -1] = 1.0
    w_cumsum_shift = F.pad(w_cumsum, (1, 0), 'constant', 0)
    cdf = torch.cumsum((v[..., 1:] + v[..., :-1]) / 2 * w, dim=-1)
    # force sum = 1
    cdf[..., -1] = 1.0
    cdf_shift = F.pad(cdf, (1, 0), 'constant', 0)

    if not inverse:
        # * x D x 1, (w_cumsum[idx-1] < x <= w_cumsum[idx])
        bin_index = torch.searchsorted(w_cumsum, x.unsqueeze(-1))
    else:
        # * x D x 1, (cdf[idx-1] < x <= cdf[idx])
        bin_index = torch.searchsorted(cdf, x.unsqueeze(-1))

    w_b = torch.gather(w, -1, bin_index).squeeze(-1)
    w_bn1 = torch.gather(w_cumsum_shift, -1, bin_index).squeeze(-1)
    v_b = torch.gather(v, -1, bin_index).squeeze(-1)
    v_bp1 = torch.gather(v, -1, bin_index + 1).squeeze(-1)
    cdf_bn1 = torch.gather(cdf_shift, -1, bin_index).squeeze(-1)

    if not inverse:
        alpha = (x - w_bn1) / w_b.clamp(min=torch.finfo(w_b.dtype).eps)
        c = (alpha ** 2) / 2 * (v_bp1 - v_b) * w_b + alpha * v_b * w_b + cdf_bn1

        # just sum of log pdfs
        log_j = torch.lerp(v_b, v_bp1, alpha).clamp(min=torch.finfo(c.dtype).eps).log()

        # make sure it falls into [0,1)
        c = c.clamp(min=torch.finfo(c.dtype).eps, max=1.0 - torch.finfo(c.dtype).eps)
        return c, log_j
    else:
        # quadratic equation for alpha
        # alpha should fall into (0, 1]. Since a, b > 0, the symmetry axis -b/2a < 0 and we should pick the larger root
        # skip calculating the log_j in inverse since we don't need it
        a = (v_bp1 - v_b) * w_b / 2
        b = v_b * w_b
        c = cdf_bn1 - x
        alpha = (-b + torch.sqrt((b ** 2) - 4 * a * c)) / (2 * a)
        inv = alpha * w_b + w_bn1

        # make sure it falls into [0,1)
        inv = inv.clamp(min=torch.finfo(c.dtype).eps, max=1.0 - torch.finfo(inv.dtype).eps)
        return inv, None
