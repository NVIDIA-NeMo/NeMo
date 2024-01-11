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
import torch
import torch.nn.functional as F


def get_view_direction(thetas: torch.Tensor, phis: torch.Tensor, overhead: float, front: float) -> torch.Tensor:
    """
    Get the view direction based on given theta and phi values.

    Parameters:
    - thetas (torch.Tensor): Array of theta values with shape [B,]
    - phis (torch.Tensor): Array of phi values with shape [B,]
    - overhead (float): Threshold for determining top and bottom views.
    - front (float): Threshold for determining front, back and side views.

    Returns:
    - torch.Tensor: Array of view directions. Values can be:
        0: front
        1: side (camera left)
        2: back
        3: side (camera right)
        4: top
        5: bottom

    Notes:
    - Phi and theta values are assumed to be in radians.
    """

    num_samples = thetas.shape[0]
    res = torch.zeros(num_samples, dtype=torch.long)

    # Normalize phis values to [0, 2*pi]
    phis = phis % (2 * np.pi)

    # Determine direction based on phis
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3

    # Override directions based on thetas for top and bottom views
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5

    return res


def compute_look_at_vectors(centers: torch.Tensor, jitter_up: Optional[float] = None, device: torch.device = "cuda"):
    """
    Compute the look-at vectors for camera poses.

    Parameters:
        centers: The centers of the cameras.
        jitter_up: The noise range for the up vector of the camera.
        device: Device to allocate the output tensor.

    Returns:
        Tuple: Contains the following:
            - forward_vector: The forward vectors of the cameras, shape [B, 3].
            - up_vector: The up vectors of the cameras, shape [B, 3].
            - right_vector: The right vectors of the cameras, shape [B, 3].
    """
    forward_vector = F.normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = F.normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_noise = torch.randn_like(up_vector) * jitter_up if jitter_up is not None else 0
    up_vector = F.normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    return forward_vector, up_vector, right_vector


def construct_poses(
    centers: torch.Tensor,
    right_vector: torch.Tensor,
    up_vector: torch.Tensor,
    forward_vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Construct the 4x4 pose matrices.

    Args:
        size (int): Number of pose matrices to construct.
        centers (torch.Tensor): The Cartesian coordinates of the camera centers.
        right_vector (torch.Tensor): The right vectors of the cameras.
        up_vector (torch.Tensor): The up vectors of the cameras.
        forward_vector (torch.Tensor): The forward vectors of the cameras.
        device (torch.device): Device to allocate tensors on.

    Returns:
        torch.Tensor: The pose matrices, shape [size, 4, 4].
    """
    poses = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack([right_vector, up_vector, forward_vector], dim=-1)
    poses[:, :3, 3] = centers

    return poses


@torch.cuda.amp.autocast(enabled=False)
def get_rays(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    num_samples: Optional[int] = None,
    error_map: Optional[torch.Tensor] = None,
    device: torch.device = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Generates rays from camera poses and intrinsics.

    Args:
        poses (torch.Tensor): Camera poses, shape [B, 4, 4] (cam2world).
        intrinsics (torch.Tensor): Intrinsic camera parameters [fx, fy, cx, cy].
        height (int): Height of the image.
        width (int): Width of the image.
        num_samples: Number of rays to sample, default is None for all rays.
        error_map: Optional tensor to use for non-uniform sampling of rays.
        device (torch.device): Device on which to generate the rays.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the following keys:
            - 'rays_o': Origin of the rays, shape [B, N, 3]
            - 'rays_d': Directions of the rays, shape [B, N, 3]
            - 'inds': Indices of the rays, shape [B, N] (if N > 0)
            - 'inds_coarse': Coarse indices of the rays, shape [B, N] (if error_map is not None)
    """

    batch_size = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=device),
        torch.linspace(0, height - 1, height, device=device),
        indexing='ij',
    )
    i = i.t().reshape([1, height * width]).expand([batch_size, height * width]) + 0.5
    j = j.t().reshape([1, height * width]).expand([batch_size, height * width]) + 0.5

    results = {}

    if num_samples is not None:
        num_samples = min(num_samples, height * width)

        if error_map is None:
            sampled_indices = torch.randint(0, height * width, size=[num_samples], device=device)
            sampled_indices = sampled_indices.expand([batch_size, num_samples])
        else:
            sampled_indices, sampled_indices_coarse = non_uniform_sampling(
                error_map=error_map, num_samples=num_samples, height=height, width=width, device=device
            )
            results['sampled_indices_coarse'] = sampled_indices_coarse

        i = torch.gather(i, -1, sampled_indices)
        j = torch.gather(j, -1, sampled_indices)
        results['sampled_indices'] = sampled_indices
    else:
        sampled_indices = torch.arange(height * width, device=device).expand([batch_size, height * width])

    zs = torch.full_like(i, -1.0)
    xs = -(i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)

    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)
    rays_o = poses[..., :3, 3].unsqueeze(-2).expand_as(rays_d)

    rays_o = rays_o.view(-1, height, width, 3)
    rays_d = rays_d.view(-1, height, width, 3)

    return rays_o, rays_d


def non_uniform_sampling(
    error_map: torch.Tensor, batch_size: int, num_samples: int, height: int, width: int, device: torch.device = "cuda"
) -> torch.Tensor:
    """
    Perform non-uniform sampling based on the provided error_map.

    Parameters:
        error_map: The error map for non-uniform sampling.
        batch_size (int): Batch size of the generated samples.
        num_samples (int): Number of samples to pick.
        height (int): Height of the image.
        width (int): Width of the image.
        device: Device on which tensors are stored.

    Returns:
        A tensor containing the sampled indices.
    """

    sampled_indices_coarse = torch.multinomial(error_map.to(device), num_samples, replacement=False)
    inds_x, inds_y = sampled_indices_coarse // 128, sampled_indices_coarse % 128
    sx, sy = height / 128, width / 128

    inds_x = (inds_x * sx + torch.rand(batch_size, num_samples, device=device) * sx).long().clamp(max=height - 1)
    inds_y = (inds_y * sy + torch.rand(batch_size, num_samples, device=device) * sy).long().clamp(max=width - 1)
    sampled_indices = inds_x * width + inds_y

    return sampled_indices, sampled_indices_coarse
