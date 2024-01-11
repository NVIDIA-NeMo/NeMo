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
from typing import Dict, Optional

import numpy as np
import tinycudann as tcnn
import torch

from nemo.collections.multimodal.modules.nerf.geometry.nerf_base import DensityActivationEnum, NeRFBase, NormalTypeEnum


# Don't fuse sigma_net with features_net:
# 1. performance benefit is questionable, especially that we sometimes require only density or features
# 2. we sacrifice generality
class TCNNNerf(NeRFBase):
    """
    NeRF model with TCNN encoding and MLPs for sigma and features.

    Args:
        num_input_dims (int): Number of input dimensions.
        bound (torch.Tensor): The bounding box tensor.
        density_activation (DensityActivationEnum): Activation function for density.
        blob_radius (float): Radius for the blob.
        blob_density (float): Density for the blob.
        normal_type (Optional[NormalTypeEnum]): Method to compute normals.
        encoder_cfg (Dict): Configuration for the TCNN encoder.
        sigma_net_num_output_dims (int): Number of output dimensions for the sigma network.
        sigma_net_cfg (Dict): Configuration for the sigma network.
        features_net_num_output_dims (int): Number of output dimensions for the features network.
        features_net_cfg (Optional[Dict]): Configuration for the features network.
    """

    def __init__(
        self,
        num_input_dims: int,
        bound: torch.Tensor,
        density_activation: DensityActivationEnum,
        blob_radius: float,
        blob_density: float,
        normal_type: Optional[NormalTypeEnum],
        encoder_cfg: Dict,
        sigma_net_num_output_dims: int,
        sigma_net_cfg: Dict,
        features_net_num_output_dims: int,
        features_net_cfg: Optional[Dict],
    ) -> None:
        super().__init__(
            num_input_dims=num_input_dims,
            bound=bound,
            density_activation=density_activation,
            blob_radius=blob_radius,
            blob_density=blob_density,
            normal_type=normal_type,
        )

        # Set per_level_scale if not set
        if encoder_cfg.get('per_level_scale') is None:
            encoder_cfg['per_level_scale'] = np.exp2(np.log2(2048 * self.bound / 16) / (16 - 1))
        # Build the TCNN encoder
        self.encoder = tcnn.Encoding(n_input_dims=num_input_dims, encoding_config=dict(encoder_cfg))

        # Build the sigma network
        assert sigma_net_num_output_dims == 1, "sigma_net_num_output_dims!=1 is not supported"
        self.sigma_tcnn = tcnn.Network(
            self.encoder.n_output_dims, sigma_net_num_output_dims, network_config=dict(sigma_net_cfg)
        )

        # Build the features network
        self.features_tcnn = None
        if features_net_cfg is not None:
            self.features_tcnn = tcnn.Network(
                self.encoder.n_output_dims, features_net_num_output_dims, network_config=dict(features_net_cfg)
            )

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode the positions using the TCNN encoder.

        Args:
            positions (torch.Tensor): The positions tensor.

        Returns:
            torch.Tensor: The encoded positions tensor.
        """
        # TODO(ahmadki): is it safe to do with FP16 ?
        return self.encoder((positions + self.bound) / (2 * self.bound))

    def sigma_net(self, positions_encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigma using the TCNN network.

        Args:
            positions_encoding (torch.Tensor): The encoded positions tensor.

        Returns:
            torch.Tensor: The sigma tensor.
        """
        return self.sigma_tcnn(positions_encoding).squeeze()

    def features_net(self, positions_encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute the features using the TCNN network.

        Args:
            positions_encoding (torch.Tensor): The encoded positions tensor.

        Returns:
            torch.Tensor: The features tensor.
        """
        return self.features_tcnn(positions_encoding)
