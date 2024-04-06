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


from typing import Dict

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch


def torch_to_numpy_with_dtype(tensor, dtype=trt.float16):
    """Converts a torch tensor to numpy array with the dtype."""
    if dtype == trt.float16:
        torch_dtype = torch.float16
    elif dtype == trt.float32:
        torch_dtype = torch.float32
    elif dtype == trt.bfloat16:
        torch_dtype = torch.bfloat16
    else:
        assert False, f"{dtype} not supported"
    return tensorrt_llm._utils.torch_to_numpy(tensor.detach().to(torch_dtype))


def split(v, tp_size, idx, dim=0):
    """Splits the np tensor v on dim and return the idx's slice."""
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def get_tensor_parallel_group(tensor_parallel: int):
    """Returns the tensor_parallel_group config based on tensor_parallel."""
    from mpi4py import MPI

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    offset = mpi_rank - mpi_rank % tensor_parallel
    tp_group = list(range(offset, offset + tensor_parallel))
    return None if tensor_parallel == 1 else tp_group


def get_tensor_from_dict(weights_dict: Dict[str, np.ndarray], name: str) -> np.array:
    """Loads tensor from the weights_dict."""
    return weights_dict.get(f"model.{name}.bin", None)
