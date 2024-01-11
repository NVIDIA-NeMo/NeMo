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

# TODO(ahmadki): abstract class
class NeRFBackgroundBase(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, rays_d: torch.Tensor) -> torch.Tensor:
        """
        positions = [B*N, 3]
        """
        raise NotImplementedError

    def forward_net(self, rays_d_encoding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, rays_d: torch.Tensor) -> torch.Tensor:
        rays_d_encoding = self.encode(rays_d)
        features = self.forward_net(rays_d_encoding)
        features = torch.sigmoid(features)
        return features
