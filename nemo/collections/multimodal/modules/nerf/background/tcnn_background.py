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
from typing import Dict

import numpy as np
import tinycudann as tcnn
import torch

from nemo.collections.multimodal.modules.nerf.background.nerf_background_base import NeRFBackgroundBase


class TCNNBackground(NeRFBackgroundBase):
    def __init__(
        self,
        bound: int,
        encoder_num_input_dims: int,
        encoder_cfg: Dict,
        background_net_num_output_dims: int,
        background_net_cfg: Dict,
    ):
        super().__init__()
        self.bound = bound
        if encoder_cfg.get('per_level_scale') is None:
            encoder_cfg['per_level_scale'] = np.exp2(np.log2(2048 * self.bound / 16) / (16 - 1))
        self.encoder = tcnn.Encoding(n_input_dims=encoder_num_input_dims, encoding_config=dict(encoder_cfg))
        self.background_net = tcnn.Network(
            self.encoder.n_output_dims, background_net_num_output_dims, network_config=dict(background_net_cfg)
        )

    def encode(self, rays_d: torch.Tensor) -> torch.Tensor:
        return self.encoder(rays_d)

    def forward_net(self, rays_d_encoding: torch.Tensor) -> torch.Tensor:
        return self.background_net(rays_d_encoding)
