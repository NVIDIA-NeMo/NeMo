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

# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang)
# See ../../../LICENSE for clarification regarding multiple authors
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

# This script was copied from https://github.com/k2-fsa/k2/blob/master/k2/python/k2/sparse/autograd.py
# with minor changes fixing uncoalesced gradients.

import torch


class _AbsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the `abs` of a sparse tensor.
        Args:
          sparse_tensor:
            A sparse tensor. It has to satisfy::
                assert sparse_tensor.is_coalesced()
        Returns:
          The absolute value of the sparse tensor.
          The `abs` operation is applied element-wise.
        """
        assert sparse_tensor.is_sparse
        assert sparse_tensor.is_coalesced()

        indices = sparse_tensor.indices().clone()
        values = sparse_tensor.values()
        size = sparse_tensor.size()

        values_abs = values.abs()

        ans = torch.sparse_coo_tensor(
            indices=indices, values=values_abs, size=size, dtype=sparse_tensor.dtype, device=sparse_tensor.device,
        )

        ctx.save_for_backward(sparse_tensor)
        return ans

    @staticmethod
    def backward(ctx, ans_grad: torch.Tensor) -> torch.Tensor:
        (sparse_tensor,) = ctx.saved_tensors

        indices = sparse_tensor.indices().clone()
        values = sparse_tensor.values()
        size = sparse_tensor.size()

        sparse_tensor_grad_values = ans_grad.coalesce().values() * values.sign()

        sparse_tensor_grad = torch.sparse_coo_tensor(
            indices=indices,
            values=sparse_tensor_grad_values,
            size=size,
            dtype=sparse_tensor.dtype,
            device=sparse_tensor.device,
        )

        return sparse_tensor_grad


def sparse_abs(sparse_tensor: torch.Tensor) -> torch.Tensor:
    """Compute the `abs` of a sparse tensor.
    It supports autograd.
    Args:
      sparse_tensor:
        A sparse tensor. It has to satisfy::
            assert sparse_tensor.is_coalesced()
    Returns:
      The absolute value of the sparse tensor.
      The `abs` operation is applied element-wise.
    """
    return _AbsFunction.apply(sparse_tensor)
