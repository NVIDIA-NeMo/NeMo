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

from contextlib import nullcontext

import torch

__all__ = ['avoid_bfloat16_autocast_context', 'avoid_float16_autocast_context']


def avoid_bfloat16_autocast_context():
    """
    If the current autocast context is bfloat16,
    cast it to float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
        return torch.cuda.amp.autocast(dtype=torch.float32)
    else:
        return nullcontext()


def avoid_float16_autocast_context():
    """
    If the current autocast context is float16, cast it to bfloat16
    if available or float32
    """

    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.cuda.is_bf16_supported():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            return torch.cuda.amp.autocast(dtype=torch.float32)
    else:
        return nullcontext()
