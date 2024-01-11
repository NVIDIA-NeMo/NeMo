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
try:
    import torchvision.transforms.functional as torchvision_F

    TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False


class ImagePyramidNoCorruptions:
    r"""
        Only downsample image without any additional corruption.
    """

    def __init__(self, target_resolutions):
        self.resolutions = target_resolutions

    def obtain_image_pyramid(self, image):
        assert TORCHVISION_AVAILABLE, "Torchvision imports failed but they are required."
        # Downsampling
        data_dict = dict()
        for res in self.resolutions:
            image_downsampled = torchvision_F.resize(
                image, res, interpolation=torchvision_F.InterpolationMode.BICUBIC, antialias=True
            )
            data_dict[f'images_{res}'] = image_downsampled
        return data_dict
