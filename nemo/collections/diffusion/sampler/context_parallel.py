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

import torch
from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, get_world_size


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenates tensors from multiple processes along a specified dimension.

    This function gathers tensors from all processes in the given process group
    and concatenates them along the specified dimension.

    Args:
        x (Tensor): The input tensor to be gathered and concatenated.
        seq_dim (int): The dimension along which to concatenate the gathered tensors.
        cp_group (ProcessGroup): The process group containing all the processes involved in the gathering.

    Returns:
        Tensor: A tensor resulting from the concatenation of tensors from all processes.

    Raises:
        RuntimeError: If the gathering of tensors fails.
    """
    # Number of processes in the group
    world_size = get_world_size(cp_group)

    # List to hold tensors from each rank
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Attempt to gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError(f"Gathering failed: {e}")

    # Concatenate tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)
