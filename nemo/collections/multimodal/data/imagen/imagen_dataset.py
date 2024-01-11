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
import torch

from nemo.collections.multimodal.data.common.webdataset import WebDatasetCommon
from nemo.collections.multimodal.data.imagen.augmentations.augmentations import (
    PickleTransform,
    build_resolution_filter,
)
from nemo.collections.multimodal.data.imagen.augmentations.corruption import ImagePyramidNoCorruptions
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
    identical_transform,
)
from nemo.core.classes import Dataset as NeMoDataset
from nemo.utils import logging


class ImagenSyntheticDataset(NeMoDataset):
    def __init__(
        self, res, conditioning_cfg, fake_len=100000, no_embedding=False,
    ):
        super().__init__()
        self.fake_len = fake_len
        self.res = res
        self.no_embedding = no_embedding
        if not no_embedding:
            self.out_key = conditioning_cfg.out_key if conditioning_cfg.out_key else conditioning_cfg.precached_key
            self.token_length = conditioning_cfg.token_length
            self.embed_dim = conditioning_cfg.embed_dim

    def __getitem__(self, index):
        item = {}
        if isinstance(self.res, list):
            for resolution in self.res:
                image_key = f'images_{resolution}'
                item[image_key] = torch.randn(3, resolution, resolution)
        else:
            item['images'] = torch.randn(3, self.res, self.res)

        item['raw_text'] = f'fake text {index}'
        if not self.no_embedding:
            item[f'{self.out_key}_embeddings'] = torch.randn(self.token_length, self.embed_dim)
            item[f'{self.out_key}_mask'] = torch.ones(self.token_length, dtype=torch.long)
        return item

    def __len__(self):
        return self.fake_len


def _build_functions_with_pickles(data_cfg, condition_cfg):
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict['images'] = input[0]

            # Output from pickle transform is already a dictionary
            out_dict.update(input[1])

            out_dict['raw_text'] = input[2]
            yield out_dict

    def transform_fn(sample):
        image, encodings, text = sample['jpg'], sample['pickle'], sample['txt']
        img_transform = construct_image_augmentations(data_cfg.train.get('augmentations'), normalize=True)
        pickle_transform = PickleTransform(
            encoding_keys=[condition_cfg.precached_key],
            encoding_lengths=[condition_cfg.token_length],
            out_keys=[condition_cfg.out_key],
        )
        text_transform = identical_transform
        return img_transform(image), pickle_transform(encodings), text_transform(text)

    return tuple_to_dict, transform_fn


def _build_functions_no_pickles(data_cfg):
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict['images'] = input[0]
            out_dict['raw_text'] = input[1]
            yield out_dict

    def transform_fn(sample):
        image, text = sample['jpg'], sample['txt']
        img_transform = construct_image_augmentations(data_cfg.train.get('augmentations'), normalize=True)
        text_transform = identical_transform
        return img_transform(image), text_transform(text)

    return tuple_to_dict, transform_fn


def build_train_valid_datasets(
    model_cfg, consumed_samples,
):
    data_cfg = model_cfg.data
    condition_cfg = model_cfg.conditioning

    if data_cfg.get('synthetic_data', False):
        logging.info(f'Creating Synthetic Datasaet.')
        train_data = ImagenSyntheticDataset(
            res=data_cfg.train.get('target_resolutions', 64),
            conditioning_cfg=condition_cfg,
            fake_len=data_cfg.get('synthetic_data_length', 10000),
            no_embedding=condition_cfg.get("online_encoding", False),
        )
        return train_data, None
    # This function maps data that are tuples to dictionary.
    if condition_cfg.get("online_encoding", False):
        tuple_to_dict, transform_fn = _build_functions_no_pickles(data_cfg)
    else:
        tuple_to_dict, transform_fn = _build_functions_with_pickles(data_cfg, condition_cfg)

    filter_cfg = data_cfg.train.get('filterings', None)

    # For adding corruptions and obtaining image pyramid
    if model_cfg.unet_type.startswith('sr'):
        assert data_cfg.train.get('target_resolutions'), 'SR model requires multiple resolution for training'
        logging.info(f'Resizing input images into the follow resolutions: {data_cfg.train.target_resolutions}')
        corruption_gen = ImagePyramidNoCorruptions(target_resolutions=data_cfg.train.target_resolutions)
    else:
        corruption_gen = None

    # This function is used for obtaining image pyramid
    # in SR models for Imagen, we need to use low-res image as conditioning.
    def obtain_image_pyramid(inp):
        for data_dict in inp:
            data_pyramid = corruption_gen.obtain_image_pyramid(data_dict['images'])
            data_dict.update(data_pyramid)
            yield data_dict

    compose_fn = [tuple_to_dict]
    if corruption_gen:
        compose_fn.append(obtain_image_pyramid)

    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        map_fn=transform_fn,
        compose_fn=compose_fn,
        filter_fn=build_resolution_filter(**filter_cfg.resolution, image_idx='jpg') if filter_cfg else None,
        is_train=True,
    )
    return train_data, None
