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
import random
from typing import Tuple

import torch
import torch.nn as nn


class RandomBackground(nn.Module):
    def __init__(self, base_background: Tuple, random_ratio: float) -> None:
        super().__init__()
        self.random_ratio = random_ratio
        self.num_output_dims = len(base_background)
        self.register_buffer("base_background", torch.tensor(base_background))

    def forward(self, rays_d: torch.Tensor) -> torch.Tensor:
        if random.random() < self.random_ratio:
            return torch.rand(rays_d.shape[0], self.num_output_dims).to(rays_d)
        else:
            return self.base_background.to(rays_d).expand(rays_d.shape[0], -1)
