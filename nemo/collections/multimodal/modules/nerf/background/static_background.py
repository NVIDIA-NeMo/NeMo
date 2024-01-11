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
from typing import Tuple

import torch
import torch.nn as nn


class StaticBackground(nn.Module):
    def __init__(self, background: Tuple) -> None:
        super().__init__()
        self.register_buffer("background", torch.tensor(background))

    def forward(self, rays_d: torch.Tensor) -> torch.Tensor:
        background = self.background.to(rays_d)
        return background.expand(rays_d.shape[0], -1)
