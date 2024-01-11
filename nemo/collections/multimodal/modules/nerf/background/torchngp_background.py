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

import torch

from nemo.collections.multimodal.modules.nerf.background.nerf_background_base import NeRFBackgroundBase
from nemo.collections.multimodal.modules.nerf.geometry.layers import MLP
from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.encoding import get_encoder


class TorchNGPBackground(NeRFBackgroundBase):
    def __init__(
        self, encoder_type: str, encoder_input_dims: int, encoder_multi_res: int, num_output_dims: int, net_cfg: Dict
    ):
        super().__init__()

        self.encoder, self.encoder_output_dims = get_encoder(
            encoder_type, input_dim=encoder_input_dims, multires=encoder_multi_res
        )
        self.background_net = MLP(
            num_input_dims=self.encoder_output_dims,
            num_output_dims=num_output_dims,
            num_hidden_dims=net_cfg.num_hidden_dims,
            num_layers=net_cfg.num_layers,
            bias=net_cfg.bias,
        )

    def encode(self, rays_d: torch.Tensor) -> torch.Tensor:
        return self.encoder(rays_d)

    def forward_net(self, rays_d_encoding: torch.Tensor) -> torch.Tensor:
        return self.background_net(rays_d_encoding)
