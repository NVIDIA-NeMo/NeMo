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

import torch

from nemo.collections.multimodal.modules.nerf.geometry.layers import MLP
from nemo.collections.multimodal.modules.nerf.geometry.nerf_base import DensityActivationEnum, NeRFBase, NormalTypeEnum
from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.encoding import get_encoder


# Don't fuse sigma_net with features_net:
# 1. performance benefit is questionable, especially that we sometimes require only density or features
# 2. we sacrifice generality
class TorchNGPNerf(NeRFBase):
    """
    NeRF model with Torch-NGP encoding and MLPs for sigma and features.

    Args:
        num_input_dims (int): Number of input dimensions.
        bound (torch.Tensor): The bounding box tensor.
        density_activation (DensityActivationEnum): Activation function for density.
        blob_radius (float): Radius for the blob.
        blob_density (float): Density for the blob.
        normal_type (Optional[NormalTypeEnum]): Method to compute normals.
        encoder_type (str): Type of the encoder.
        encoder_max_level (int): Maximum level of the encoder.
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
    ):
        super().__init__(
            num_input_dims=num_input_dims,
            bound=bound,
            density_activation=density_activation,
            blob_radius=blob_radius,
            blob_density=blob_density,
            normal_type=normal_type,
        )

        # Build the Torch-NGP encoder
        self.encoder_max_level = encoder_cfg.get('encoder_max_level', None)
        self.encoder, self.encoder_output_dims = get_encoder(input_dim=num_input_dims, **encoder_cfg)

        # Build the sigma network
        assert sigma_net_num_output_dims == 1, "sigma_net_num_output_dims must be equal to 1"
        self.sigma_mlp = MLP(
            num_input_dims=self.encoder_output_dims,
            num_output_dims=sigma_net_num_output_dims,
            num_hidden_dims=sigma_net_cfg.num_hidden_dims,
            num_layers=sigma_net_cfg.num_layers,
            bias=sigma_net_cfg.bias,
        )

        # Build the features network
        self.features_mlp = None
        if features_net_cfg is not None:
            self.features_mlp = MLP(
                num_input_dims=self.encoder_output_dims,
                num_output_dims=features_net_num_output_dims,
                num_hidden_dims=features_net_cfg.num_hidden_dims,
                num_layers=features_net_cfg.num_layers,
                bias=features_net_cfg.bias,
            )

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode the positions.

        Args:
            positions (torch.Tensor): The positions tensor.

        Returns:
            torch.Tensor: The encoded positions tensor.
        """
        return self.encoder(positions, bound=self.bound, max_level=self.encoder_max_level)

    def sigma_net(self, positions_encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigma using the sigma network.

        Args:
            positions_encoding (torch.Tensor): The encoded positions tensor.

        Returns:
            torch.Tensor: The sigma tensor.
        """
        return self.sigma_mlp(positions_encoding).squeeze()

    def features_net(self, positions_encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute the features using the features network.

        Args:
            positions_encoding (torch.Tensor): The encoded positions tensor.

        Returns:
            torch.Tensor: The features tensor.
        """
        return self.features_mlp(positions_encoding)
