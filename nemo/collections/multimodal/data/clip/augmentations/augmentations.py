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
"""
This code is adapted from public repo
https://github.com/mlfoundations/open_clip/blob/28c994406e39a5babc749c76871d92f33e9c558d/src/open_clip/transform.py
by @yaoyu-33
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import torchvision.transforms.functional as F
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )

    TORCHVISION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TORCHVISION_AVAILABLE = False

from nemo.utils import logging

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            assert TORCHVISION_AVAILABLE, "Torchvision imports failed but they are required."
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    assert TORCHVISION_AVAILABLE, "Torchvision imports failed but they are required."
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional

            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.0,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            train_transform = Compose(
                [
                    RandomResizedCrop(
                        image_size, scale=aug_cfg_dict.pop('scale'), interpolation=InterpolationMode.BICUBIC,
                    ),
                    _convert_to_rgb,
                    ToTensor(),
                    normalize,
                ]
            )
            if aug_cfg_dict:
                logging.warning(
                    f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).'
                )
        return train_transform
    else:
        if resize_longest_max:
            transforms = [ResizeMaxSize(image_size, fill=fill_color)]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend(
            [_convert_to_rgb, ToTensor(), normalize,]
        )
        return Compose(transforms)
