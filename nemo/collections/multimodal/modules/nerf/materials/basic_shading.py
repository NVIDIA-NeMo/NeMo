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
from typing import Optional

import torch

from nemo.collections.multimodal.modules.nerf.materials.materials_base import MaterialsBase, ShadingEnum


class BasicShading(MaterialsBase):
    """
    Material model for handling various shading types.
    """

    def __init__(self):
        super(BasicShading, self).__init__()
        self.specular = torch.nn.Parameter(torch.rand(3))
        self.shininess = torch.nn.Parameter(torch.rand(1))

    def forward(
        self,
        albedo: torch.Tensor,
        normals: torch.Tensor,
        light_d: torch.Tensor,
        ambient_ratio: float,
        shading_type: Optional[ShadingEnum] = None,
    ) -> torch.Tensor:
        """
        Apply material and shading to the input RGB tensor.

        Args:
            albedo (Tensor): Base albedo values.
            normals (Tensor): Normal vectors at each ray intersection.
            light_d (Tensor): Light direction.
            ambient_ratio (float): Ratio for ambient lighting.
            shading_type (ShadingEnum): The type of shading to apply

        Returns:
            Tensor: The output RGB tensor after applying material and shading.
        """
        if shading_type is None:
            return albedo
        elif shading_type == ShadingEnum.TEXTURELESS:
            return torch.ones_like(albedo) * ambient_ratio
        elif shading_type == ShadingEnum.NORMAL:
            return (normals + 1) / 2  # Map normals from [-1, 1] to [0, 1]
        elif shading_type in [ShadingEnum.LAMBERTIAN, ShadingEnum.PHONG]:
            # Ambient light
            ambient_light = ambient_ratio * albedo
            # Dot product between light direction and normals
            dot_product = torch.sum(normals * light_d, dim=1, keepdim=True)
            # Lambertian term
            diffuse_term = albedo * torch.clamp(dot_product, min=0)

            if shading_type == ShadingEnum.LAMBERTIAN:
                return ambient_light + diffuse_term
            elif shading_type == ShadingEnum.PHONG:
                # Phong specular term
                specular_term = (
                    self.specular
                    * (self.shininess + 2)
                    * torch.pow(torch.clamp(dot_product, min=0), self.shininess)
                    / (2 * 3.14159)
                )

                return ambient_light + diffuse_term + specular_term
        else:
            raise ValueError(f"Unknown shading_type: {shading_type}")
