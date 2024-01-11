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

from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from nemo.collections.multimodal.data.nerf.cameras import PinholeCamera
from nemo.collections.multimodal.data.nerf.utils import (
    compute_look_at_vectors,
    construct_poses,
    get_rays,
    get_view_direction,
)


def circle_poses(
    radius: torch.Tensor = torch.tensor([3.2]),
    theta: torch.Tensor = torch.tensor([60]),
    phi: torch.Tensor = torch.tensor([0]),
    angle_overhead: float = 30,
    angle_front: float = 60,
    return_dirs: bool = False,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """
    Generate camera poses based on a circular arrangement.

    Parameters:
        radius: torch.Tensor - Radii for the camera positions.
        theta: torch.Tensor - Theta angles for the camera positions.
        phi: torch.Tensor - Phi angles for the camera positions.
        angle_overhead: float - Angle range of the overhead view.
        angle_front: float - Angle range of the front view.
        return_dirs: bool - Whether to return the view directions.
        device: str - The device to allocate the tensor on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple: Contains the following:
            - poses (torch.Tensor): Generated poses, shape [size, 4, 4].
            - dirs (torch.Tensor, optional): View directions, if requested.
    """
    # Convert degrees to radians for theta and phi
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # Calculate camera centers in Cartesian coordinates
    centers = torch.stack(
        [
            radius * torch.sin(theta) * torch.sin(phi),
            radius * torch.cos(theta),
            radius * torch.sin(theta) * torch.cos(phi),
        ],
        dim=-1,
    )  # [B, 3]

    # Compute camera look-at matrix
    forward_vector, up_vector, right_vector = compute_look_at_vectors(centers=centers, device=device)

    # Construct the 4x4 pose matrices
    poses = construct_poses(
        centers=centers, right_vector=right_vector, up_vector=up_vector, forward_vector=forward_vector, device=device
    )

    dirs = get_view_direction(theta, phi, angle_overhead, angle_front) if return_dirs else None

    return poses, dirs


class CirclePosesDataset(Dataset):
    """
    A dataset class to generate circle poses.
    """

    def __init__(
        self,
        size: int = 100,
        height: int = 256,
        width: int = 256,
        default_fovx: float = 20.0,
        default_fovy: float = 20.0,
        default_radius: float = 3.2,
        default_polar: float = 90.0,
        default_azimuth: float = 0.0,
        angle_overhead: float = 30.0,
        angle_front: float = 60.0,
        near: float = 0.01,
        far: float = 1000.0,
        device: torch.device = 'cpu',
    ) -> None:
        """
        Initializes a new CirclePosesDataset instance.

        Parameters:
            size (int): Number of samples in the dataset.
            height (int): Height of the image.
            width (int): Width of the image.
            default_fovx (float): Default field of view in x-direction.
            default_fovy (float): Default field of view in y-direction.
            default_radius (float): Default radius of the circle.
            default_polar (float): Default polar angle.
            default_azimuth (float): Default azimuth angle.
            angle_overhead (float): Overhead angle.
            angle_front (float): Frontal angle.
            near (float): Near clipping distance.
            far (float): Far clipping distance.
            device (torch.device): Device to generate data on.
        """
        super().__init__()
        self.size = size
        self.height = height
        self.width = width

        self.default_fovx = default_fovx
        self.default_fovy = default_fovy
        self.default_radius = default_radius
        self.default_polar = default_polar
        self.default_azimuth = default_azimuth

        self.angle_overhead = angle_overhead
        self.angle_front = angle_front
        self.near = near
        self.far = far

        self.device = device

        # TODO(ahmadki): make camera type a parameter
        self.camera = PinholeCamera(
            width=self.width, height=self.height, near=self.near, far=self.far, device=self.device
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, torch.Tensor]]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Data dictionary containing the following:
            - height (int): Height of the image.
            - width (int): Width of the image.
            - rays_o (torch.Tensor): Ray origins, shape [height, width, 3].
            - rays_d (torch.Tensor): Ray directions, shape [height, width, 3].
            - dir (torch.Tensor): View direction, shape [3].
            - mvp (torch.Tensor): Model-view-projection matrix, shape [4, 4].
            - azimuth (torch.Tensor): Azimuth angle, shape [1].
        """
        # Initialize circle pose parameters
        thetas = torch.FloatTensor([self.default_polar]).to(self.device)
        phis = torch.FloatTensor([(idx / self.size) * 360]).to(self.device)
        radius = torch.FloatTensor([self.default_radius]).to(self.device)

        # Generate circle poses and directions
        poses, dirs = circle_poses(
            radius=radius,
            theta=thetas,
            phi=phis,
            angle_overhead=self.angle_overhead,
            angle_front=self.angle_front,
            return_dirs=True,
            device=self.device,
        )

        # Compute camera intrinsics
        intrinsics = self.camera.compute_intrinsics(fovx=self.default_fovx, fovy=self.default_fovy)

        # Compute projection matrix
        projection = self.camera.compute_projection_matrix(focal_x=intrinsics[0], focal_y=intrinsics[1])
        mvp = projection @ torch.inverse(poses)  # [1, 4, 4]

        # Sample rays
        rays_o, rays_d = get_rays(
            poses=poses, intrinsics=intrinsics, height=self.height, width=self.width, device=poses.device
        )

        # Compute azimuth delta
        delta_azimuth = phis - self.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]

        data = {
            'height': self.height,
            'width': self.width,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'dir': dirs,
            'mvp': mvp,
            'azimuth': delta_azimuth,
        }

        return data

    def collate_fn(self, batch: list) -> Dict[str, Union[int, torch.Tensor]]:
        """Collate function to combine multiple data points into batches.

        Args:
            batch (list): List of data dictionaries.

        Returns:
            dict: Collated data.
        """
        return {
            'height': self.height,
            'width': self.width,
            'rays_o': torch.cat([item['rays_o'] for item in batch], dim=0),
            'rays_d': torch.cat([item['rays_d'] for item in batch], dim=0),
            'mvp': torch.cat([item['mvp'] for item in batch], dim=0),
            'dir': torch.cat([item['dir'] for item in batch], dim=0),
            'azimuth': torch.cat([item['azimuth'] for item in batch], dim=0),
        }
