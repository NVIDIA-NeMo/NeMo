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
from enum import Enum

from torch import nn


class ShadingEnum(str, Enum):
    TEXTURELESS = "textureless"
    NORMAL = "normal"
    LAMBERTIAN = "lambertian"
    PHONG = "phong"

    # TODO(ahmadki):
    # Oren–Nayar
    # Minnaert
    # Cook–Torrance
    # Ward anisotropic
    # Hanrahan–Krueger
    # Cel shading
    # Gooch shading


class MaterialsBase(nn.Module):
    """
    Base class for materials.
    """

    def __init__(self):
        super(MaterialsBase, self).__init__()
