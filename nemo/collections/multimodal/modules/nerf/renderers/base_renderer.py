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
import torch
import torch.nn as nn

# TODO(ahmadki): make abstract
class BaseRenderer(nn.Module):
    def __init__(self, bound, update_interval):
        super().__init__()
        self.bound = bound
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)
        self.update_interval = update_interval

    @torch.no_grad()
    def update_step(self, epoch: int, global_step: int, decay: float = 0.95, **kwargs):
        raise NotImplementedError

    def forward(self, rays_o, rays_d, return_normal_image=False, return_normal_perturb=False, **kwargs):
        raise NotImplementedError
