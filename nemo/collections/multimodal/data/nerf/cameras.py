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

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch


class Camera(ABC):
    """
    Abstract base class for Camera models.
    """

    def __init__(self, width: int, height: int, device: torch.device = 'cuda') -> None:
        """
        Initializes the Camera instance with given dimensions and device.

        Parameters:
            width: int - Width of the camera frame.
            height: int - Height of the camera frame.
            device: torch.device - The device where tensor computations will be performed.
        """
        self.width = width
        self.height = height
        self.device = device

    @abstractmethod
    def compute_intrinsics(self) -> None:
        """
        Abstract method to compute camera intrinsics.
        """
        pass

    @abstractmethod
    def compute_projection_matrix(self) -> None:
        """
        Abstract method to compute the projection matrix.
        """
        pass


class OrthographicCamera(Camera):
    """
    Class for Orthographic Camera models.
    """

    def compute_projection_matrix(self) -> torch.Tensor:
        """
        Computes the projection matrix for an Orthographic camera.

        Returns:
            torch.Tensor: The projection matrix.
        """
        projection = torch.tensor(
            [[2 / self.width, 0, 0, 0], [0, -2 / self.height, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return projection


class PinholeCamera(Camera):
    """
    Class for Pinhole Camera models.
    """

    def __init__(self, width: int, height: int, near: float, far: float, device: torch.device = 'cuda') -> None:
        """
        Initializes the Pinhole Camera instance with given parameters.

        Parameters:
            width: int - Width of the camera frame.
            height: int - Height of the camera frame.
            near: float - Near clipping plane.
            far: float - Far clipping plane.
            device: torch.device - The device where tensor computations will be performed.
        """
        super().__init__(width, height, device)
        self.near = near
        self.far = far

    def compute_intrinsics(self, fovx: float, fovy: float) -> np.ndarray:
        """
        Computes the intrinsic matrix for the camera based on field of views.

        Parameters:
            fovx: float - Field of view in X direction.
            fovy: float - Field of view in Y direction.

        Returns:
            np.ndarray: The intrinsic matrix.
        """
        focal_x = self.width / (2 * np.tan(np.deg2rad(fovx) / 2))
        focal_y = self.height / (2 * np.tan(np.deg2rad(fovy) / 2))
        cx, cy = self.width / 2, self.height / 2
        return np.array([focal_x, focal_y, cx, cy])

    def compute_projection_matrix(self, focal_x: float, focal_y: float) -> torch.Tensor:
        """
        Computes the projection matrix for the camera.

        Parameters:
            focal_x: float - Focal length in X direction.
            focal_y: float - Focal length in Y direction.

        Returns:
            torch.Tensor: The projection matrix.
        """
        projection = torch.tensor(
            [
                [2 * focal_x / self.width, 0, 0, 0],
                [0, -2 * focal_y / self.height, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return projection


class CubeCamera(Camera):
    """
    Class for Cube Camera models, which is essentially six pinhole cameras.
    """

    def __init__(
        self, width: int, height: int, near: float = 0.01, far: float = 1000, device: torch.device = 'cuda'
    ) -> None:
        """
        Initializes the Cube Camera instance with given parameters.

        Parameters:
            width: int - Width of each camera face.
            height: int - Height of each camera face.
            near: float - Near clipping plane.
            far: float - Far clipping plane.
            device: torch.device - The device where tensor computations will be performed.
        """
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.device = device

    def compute_intrinsics(self) -> List[np.ndarray]:
        """
        Computes the intrinsic matrices for the six faces of the cube using a Pinhole camera model.

        Returns:
            List[np.ndarray]: List of 6 intrinsic matrices, one for each face.
        """
        # Similar to Pinhole but repeated six times for six faces of the cube
        return [
            PinholeCamera(
                width=self.width, height=self.height, near=self.near, far=self.far, device=self.device
            ).compute_intrinsics(90, 90)
            for _ in range(6)
        ]

    def compute_projection_matrix(self) -> List[torch.Tensor]:
        """
        Computes the projection matrices for the six faces of the cube using a Pinhole camera model.

        Returns:
            List[torch.Tensor]: List of 6 projection matrices, one for each face.
        """
        # Similar to Pinhole but repeated six times for six faces of the cube
        return [
            PinholeCamera(
                width=self.width, height=self.height, near=self.near, far=self.far, device=self.device
            ).compute_projection_matrix(1, 1)
            for _ in range(6)
        ]
