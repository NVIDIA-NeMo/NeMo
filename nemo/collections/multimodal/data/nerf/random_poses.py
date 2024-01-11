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
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from nemo.collections.multimodal.data.nerf.cameras import PinholeCamera
from nemo.collections.multimodal.data.nerf.utils import (
    compute_look_at_vectors,
    construct_poses,
    get_rays,
    get_view_direction,
)


def linear_normalization(x: float, lower_bound: float, upper_bound: float) -> float:
    """
    Linearly normalize a value between lower_bound and upper_bound to a value between 0 and 1.

    Parameters:
        x: The value to normalize.
        lower_bound: The lower bound of the range of x.
        upper_bound: The upper bound of the range of x.

    Returns:
        The normalized value between 0 and 1.
    """
    return min(1, max(0, (x - lower_bound) / (upper_bound - lower_bound)))


def rand_poses(
    size: int,
    radius_range: List[float] = [1, 1.5],
    theta_range: List[float] = [0, 120],
    phi_range: List[float] = [0, 360],
    angle_overhead: float = 30,
    angle_front: float = 60,
    uniform_sphere_rate: float = 0.5,
    jitter: bool = False,
    jitter_center: float = 0.2,
    jitter_target: float = 0.2,
    jitter_up: float = 0.02,
    return_dirs: bool = False,
    device: torch.device = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate random poses from an orbit camera.

    Args:
        size (int): Number of poses to generate.
        radius_range (List[float]): Min and max radii for camera [min, max].
        theta_range (List[float]): Elevation angle range in degrees [min, max].
        phi_range (List[float]): Azimuth angle range in degrees [min, max].
        angle_overhead (float): Overhead angle in degrees.
        angle_front (float): Front angle in degrees.
        uniform_sphere_rate (float): The probability of sampling from a uniform sphere.
        jitter (bool): Whether to add noise to the poses.
        jitter_center (float): Noise range for the camera center.
        jitter_target (float): Noise range for the camera target.
        jitter_up (float): Noise range for the camera up vector.
        return_dirs (bool): Whether to return the view directions.
        device (torch.device): The device on which to allocate tensors.

    Returns:
        Tuple: Contains the following:
            - poses (torch.Tensor): Generated poses, shape [size, 4, 4].
            - thetas (torch.Tensor): Elevation angles in degrees, shape [size].
            - phis (torch.Tensor): Azimuth angles in degrees, shape [size].
            - radius (torch.Tensor): Radii of the camera orbits, shape [size].
            - dirs (torch.Tensor, optional): View directions, if requested.
    """

    # Convert angles from degrees to radians
    theta_range = np.radians(theta_range)
    phi_range = np.radians(phi_range)
    angle_overhead = np.radians(angle_overhead)
    angle_front = np.radians(angle_front)

    # Generate radius for each pose
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    # Generate camera center positions
    if random.random() < uniform_sphere_rate:
        centers, thetas, phis = sample_uniform_sphere(size=size, radius=radius, device=device)
    else:
        centers, thetas, phis = sample_orbit(
            size=size, radius=radius, theta_range=theta_range, phi_range=phi_range, device=device
        )

    # Initialize targets to 0 (assuming 0 is a point in 3D space that cameras are looking at)
    targets = torch.zeros_like(centers)

    # Apply jitter
    if jitter:
        centers += torch.rand_like(centers) * jitter_center - jitter_center / 2.0
        targets = torch.randn_like(centers) * jitter_target

    # Compute camera look-at matrix
    forward_vector, up_vector, right_vector = compute_look_at_vectors(
        centers=centers - targets, jitter_up=jitter_up if jitter else 0, device=device
    )

    # Construct the 4x4 pose matrices
    poses = construct_poses(
        centers=centers, right_vector=right_vector, up_vector=up_vector, forward_vector=forward_vector, device=device
    )

    # Optionally compute view directions
    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front) if return_dirs else None

    # Convert back to degrees for thetas and phis
    thetas, phis = torch.rad2deg(thetas), torch.rad2deg(phis)

    return poses, thetas, phis, radius, dirs


def sample_uniform_sphere(
    size: int, radius: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points uniformly on a sphere.

    Args:
        size (int): Number of points to sample.
        device (torch.device): Device to allocate tensors on.
        radius (torch.Tensor): Radii for the points.

    Returns:
        Tuple: Contains the following:
            - centers (torch.Tensor): The Cartesian coordinates of the sampled points.
            - thetas (torch.Tensor): Elevation angles in radians.
            - phis (torch.Tensor): Azimuth angles in radians.
    """
    # Generate unit vectors
    unit_centers = F.normalize(
        torch.stack(
            [
                torch.randn(size, device=device),
                torch.abs(torch.randn(size, device=device)),
                torch.randn(size, device=device),
            ],
            dim=-1,
        ),
        p=2,
        dim=1,
    )
    # Generate radii and scale unit vectors
    centers = unit_centers * radius.unsqueeze(-1)
    # Calculate spherical coordinates
    thetas = torch.acos(unit_centers[:, 1])
    phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
    phis[phis < 0] += 2 * np.pi

    return centers, thetas, phis


def sample_orbit(
    size: int, radius: torch.Tensor, theta_range: np.ndarray, phi_range: np.ndarray, device: torch.device = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample points on a spherical orbit.

    Args:
        size (int): Number of points to sample.
        radius (torch.Tensor): Radii for the points.
        theta_range (np.ndarray): Elevation angle range in radians [min, max].
        phi_range (np.ndarray): Azimuth angle range in radians [min, max].
        device (torch.device): Device to allocate tensors on.

    Returns:
        Tuple: Contains the following:
            - centers (torch.Tensor): The Cartesian coordinates of the sampled points.
            - thetas (torch.Tensor): Elevation angles in radians.
            - phis (torch.Tensor): Azimuth angles in radians.
    """
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
    phis[phis < 0] += 2 * np.pi

    x = radius * torch.sin(thetas) * torch.sin(phis)
    y = radius * torch.cos(thetas)
    z = radius * torch.sin(thetas) * torch.cos(phis)

    centers = torch.stack([x, y, z], dim=-1)

    return centers, thetas, phis


class RandomPosesDataset(IterableDataset):
    """
    A dataset class to generate random poses.
    """

    def __init__(
        self,
        internal_batch_size: int = 100,
        height: int = 256,
        width: int = 256,
        radius_range: Tuple[float, float] = [3.0, 3.5],
        theta_range: Tuple[float, float] = [45.0, 105.0],
        phi_range: Tuple[float, float] = [-180.0, 180.0],
        fovx_range: Tuple[float, float] = [10.0, 30.0],
        default_fovx: float = 20.0,
        fovy_range: Tuple[float, float] = [10.0, 30.0],
        default_fovy: float = 20.0,
        default_radius: float = 3.2,
        default_polar: float = 90.0,
        default_azimuth: float = 0.0,
        jitter: bool = False,
        jitter_center: float = 0.2,
        jitter_target: float = 0.2,
        jitter_up: float = 0.02,
        angle_overhead: float = 30.0,
        angle_front: float = 60.0,
        uniform_sphere_rate: float = 0.0,
        near: float = 0.01,
        far: float = 1000.0,
        device: torch.device = 'cpu',
    ) -> None:
        """
        Initializes a new RandomPosesDataset instance.

        Parameters:
            internal_batch_size (int): Number of samples to pre-generate internally.
            height (int): Height of the image.
            width (int): Width of the image.
            radius_range (Tuple[float, float]): Range of generated radii.
            theta_range (Tuple[float, float]): Range of generated theta angles.
            phi_range (Tuple[float, float]): Range of generated phi angles.
            fovx_range (Tuple[float, float]): Range of generated field of view in x-direction.
            default_fovx (float): Default field of view in x-direction.
            fovy_range (Tuple[float, float]): Range of generated field of view angles in y-direction.
            default_fovy (float): Default field of view in y-direction.
            default_radius (float): Default radius of the circle.
            default_polar (float): Default polar angle.
            default_azimuth (float): Default azimuth angle.
            jitter (bool): Whether to jitter the poses.
            jitter_center (float): Jittering center range.
            jitter_target (float): Jittering target range.
            jitter_up (float): Jittering up range.
            angle_overhead (float): Overhead angle.
            angle_front (float): Frontal angle.
            uniform_sphere_rate (float): Rate of sampling uniformly on a sphere.
            near (float): Near clipping distance.
            far (float): Far clipping distance.
            device (torch.device): Device to generate data on.
        """

        super().__init__()
        self.height = height
        self.width = width
        self.internal_batch_size = internal_batch_size

        # TODO(ahmadki): expose for models other than dreamfusion
        self.progressive_view = False
        self.progressive_view_start_step = 0
        self.progressive_view_end_step = 500

        self.default_fovx = default_fovx
        self.default_fovy = default_fovy
        self.default_radius = default_radius
        self.default_polar = default_polar
        self.default_azimuth = default_azimuth
        self.same_fov_random = True

        self.radius_range = radius_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.fovx_range = fovx_range
        self.fovy_range = fovy_range

        self.current_radius_range = radius_range
        self.current_theta_range = theta_range
        self.current_phi_range = phi_range
        self.current_fovx_range = fovx_range
        self.current_fovy_range = fovy_range

        self.angle_overhead = angle_overhead
        self.angle_front = angle_front
        self.uniform_sphere_rate = uniform_sphere_rate
        self.jitter = jitter
        self.jitter_center = jitter_center
        self.jitter_target = jitter_target
        self.jitter_up = jitter_up

        self.near = near
        self.far = far

        self.device = device

        # TODO(ahmadki): make camera type a parameter
        self.camera = PinholeCamera(
            width=self.width, height=self.height, near=self.near, far=self.far, device=self.device
        )

    def update_step(self, epoch: int, global_step: int) -> None:
        """
        Update the dataset at the beginning of each epoch.

        Parameters:
            epoch (int): Current epoch.
            global_step (int): Current global step.

        """
        if self.progressive_view:
            self.progressive_view_update_step(global_step=global_step)

    def progressive_view_update_step(self, global_step: int) -> None:
        """
        progressively relaxing view range

        Parameters:
            global_step (int): Current global step.
        """
        # TODO(ahmadki): support non-linear progressive_views
        r = linear_normalization(
            x=global_step, lower_bound=self.progressive_view_start_step, upper_bound=self.progressive_view_end_step
        )
        self.current_phi_range = [
            (1 - r) * self.default_azimuth + r * self.phi_range[0],
            (1 - r) * self.default_azimuth + r * self.phi_range[1],
        ]
        self.current_theta_range = [
            (1 - r) * self.default_polar + r * self.theta_range[0],
            (1 - r) * self.default_polar + r * self.theta_range[1],
        ]
        self.current_radius_range = [
            (1 - r) * self.default_radius + r * self.radius_range[0],
            (1 - r) * self.default_radius + r * self.radius_range[1],
        ]
        self.current_fovy_range = [
            (1 - r) * self.default_fovy + r * self.fovy_range[0],
            (1 - r) * self.default_fovy + r * self.fovy_range[1],
        ]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Returns an iterator over the dataset.

        Returns:
            Iterator: An iterator over the dataset.

        """
        while True:
            # Generate samples
            rays_o, rays_d, dirs, mvp, delta_azimuth = self.generate_samples()
            for i in range(self.internal_batch_size):
                # Yield one sample at a time from the internal batch
                yield {
                    'height': self.height,
                    'width': self.width,
                    'rays_o': rays_o[i].unsqueeze(0),
                    'rays_d': rays_d[i].unsqueeze(0),
                    'dir': dirs[i].unsqueeze(0),
                    'mvp': mvp[i].unsqueeze(0),
                    'azimuth': delta_azimuth[i].unsqueeze(0),
                }

    def generate_samples(self):
        """
        Generate a batch of random poses.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                A tuple containing:
                    - rays (Dict[str, torch.Tensor]): A dictionary containing the origin and direction of the rays.
                    - dirs (torch.Tensor): A tensor containing the directions of the rays.
                    - mvp (torch.Tensor): A tensor containing the model-view-projection matrix.
                    - azimuth (torch.Tensor): A A tensor containing the azimuth angle.
        """
        # Generate random poses and directions
        poses, dirs, thetas, phis, radius = rand_poses(
            size=self.internal_batch_size,
            radius_range=self.current_radius_range,
            theta_range=self.current_theta_range,
            phi_range=self.current_phi_range,
            angle_overhead=self.angle_overhead,
            angle_front=self.angle_front,
            uniform_sphere_rate=self.uniform_sphere_rate,
            jitter=self.jitter,
            jitter_center=self.jitter_center,
            jitter_target=self.jitter_target,
            jitter_up=self.jitter_up,
            return_dirs=True,
            device=self.device,
        )

        # random focal
        if self.same_fov_random:
            fovx_random = random.random()
            fovy_random = fovx_random
        else:
            fovx_random = random.random()
            fovy_random = random.random()
        fovx = fovx_random * (self.current_fovx_range[1] - self.current_fovx_range[0]) + self.current_fovx_range[0]
        fovy = fovy_random * (self.current_fovy_range[1] - self.current_fovy_range[0]) + self.current_fovy_range[0]

        # Compute camera intrinsics
        intrinsics = self.camera.compute_intrinsics(fovx=fovx, fovy=fovy)

        # Compute projection matrix
        projection = self.camera.compute_projection_matrix(focal_x=intrinsics[0], focal_y=intrinsics[1])
        mvp = projection @ torch.inverse(poses)  # [internal batch size, 4, 4]

        # Sample rays
        rays_o, rays_d = get_rays(
            poses=poses, intrinsics=intrinsics, height=self.height, width=self.width, device=poses.device
        )

        # Compute azimuth delta
        delta_azimuth = phis - self.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360  # range in [-180, 180]

        return rays_o, rays_d, dirs, mvp, delta_azimuth

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function to bundle multiple samples into a single batch.

        Args:
            batch (List[Dict]): List of samples to collate.

        Returns:
            Dict: A dictionary containing the collated batch.
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
