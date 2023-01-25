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


import numba
import numpy as np
import torch


def maximum_path(neg_cent, mask):
    """ Numba version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


@numba.jit(nopython=True, boundscheck=False, parallel=True)
def maximum_path_each(path, value, t_y: int, t_x: int, max_neg_val=-1e9):
    """
    Args:
        path: int32[:, :]
        value: float32[:, :]
        t_y: int
        t_x: int
        max_neg_val: float
    """
    index: int = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@numba.jit(nopython=True, boundscheck=False, parallel=True)
def maximum_path_c(paths, values, t_ys, t_xs):
    """
    Args:
      paths: int32[:, :, :]
      values: float32[:, :, :]
      t_ys: int[:]
      t_xs: int[:]
    """
    b: int = paths.shape[0]
    for i in numba.prange(b):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


if __name__ == '__main__':
    pass
