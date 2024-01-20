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
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
    identical_transform,
)
from nemo.core.classes import Dataset as NeMoDataset
from nemo.utils import logging


class SDSyntheticDataset(NeMoDataset):
    def __init__(
        self, image_H, image_W, fake_len=100000, image_key='images', txt_key='txt', seq_len=80, context_dim=768
    ):
        super().__init__()
        self.fake_len = fake_len
        self.H = image_H
        self.W = image_W
        self.image_key = image_key
        self.txt_key = txt_key
        assert image_key.endswith('encoded') == txt_key.endswith(
            'encoded'
        ), 'In precached mode, first and second stage key must both end with "encoded"'
        self.precached = self.image_key.endswith('encoded')
        self.seq_len = seq_len
        self.context_dim = context_dim

    def __getitem__(self, index):
        item = {}
        if self.precached:
            item[self.image_key] = torch.randn(8, self.H // 8, self.W // 8)
            item[self.txt_key] = torch.randn(self.seq_len, self.context_dim)
        else:
            item[self.image_key] = torch.randn(self.H, self.W, 3)
            item[self.txt_key] = f'This is meaningless fake text No.{index}'

        return item

    def __len__(self):
        return self.fake_len


def build_train_valid_datasets(
    model_cfg, consumed_samples,
):
    data_cfg = model_cfg.data

    def build_resolution_filter(value=None, method='larger'):
        assert method == 'larger' or method == 'smaller'
        if method == 'larger':
            logging.info(f'Only Selecting images with resolution >= {value}')
            return lambda x: x['jpg'].size[0] >= value and x['jpg'].size[1] >= value
        logging.info(f'Only Selecting images with resolution <= {value}')
        return lambda x: x['jpg'].size[0] <= value and x['jpg'].size[1] <= value

    # This function maps data that are tuples to dictionary.
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict[model_cfg.first_stage_key] = input[0].permute(1, 2, 0)
            out_dict[model_cfg.cond_stage_key] = input[1]
            yield out_dict

    def transform_fn(sample):
        image, text = sample["jpg"], sample["txt"]
        # TODO : If no agumentations just return the image ?
        img_transform = construct_image_augmentations(data_cfg.train.get("augmentations", None))
        text_transform = identical_transform
        return img_transform(image), text_transform(text)

    if data_cfg.get('synthetic_data', False):
        H, W = data_cfg.train.augmentations.center_crop_h_w.split(',')
        train_data = SDSyntheticDataset(
            int(H),
            int(W),
            image_key=model_cfg.first_stage_key,
            txt_key=model_cfg.cond_stage_key,
            context_dim=model_cfg.unet_config.context_dim,
            fake_len=data_cfg.synthetic_data_length,
        )

    else:
        filter_cfg = data_cfg.train.get('filterings', None)
        filter_fn = build_resolution_filter(**filter_cfg.resolution) if filter_cfg else None
        train_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=consumed_samples,
            map_fn=transform_fn,
            compose_fn=tuple_to_dict,
            filter_fn=filter_fn,
            is_train=True,
        )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("data_path"):
        if data_cfg.get('synthetic_data', False):
            val_data = SDSyntheticDataset(
                int(H),
                int(W),
                image_key=model_cfg.first_stage_key,
                txt_key=model_cfg.cond_stage_key,
                context_dim=model_cfg.unet_config.context_dim,
            )
        else:
            val_data = WebDatasetCommon(
                dataset_cfg=data_cfg,
                consumed_samples=consumed_samples,
                map_fn=transform_fn,
                compose_fn=tuple_to_dict,
                filter_fn=filter_fn,
                is_train=False,
            )

    return train_data, val_data


def build_train_valid_precached_datasets(
    model_cfg, consumed_samples,
):
    data_cfg = model_cfg.data

    # This function maps data that are tuples to dictionary.
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict[model_cfg.first_stage_key] = torch.tensor(input['autoencoderkl_image'])
            out_dict[model_cfg.cond_stage_key] = torch.tensor(input['clip-vit-large-patch14_text'])
            yield out_dict

    def transform_fn(sample):
        return sample['pickle']

    if data_cfg.get('synthetic_data', False):
        H, W = data_cfg.train.augmentations.center_crop_h_w.split(',')
        train_data = SDSyntheticDataset(
            int(H),
            int(W),
            image_key=model_cfg.first_stage_key,
            txt_key=model_cfg.cond_stage_key,
            context_dim=model_cfg.unet_config.context_dim,
        )
    else:
        train_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=consumed_samples,
            map_fn=transform_fn,
            compose_fn=tuple_to_dict,
            is_train=True,
        )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("data_path"):
        if data_cfg.get('synthetic_data', False):
            H, W = data_cfg.train.augmentations.center_crop_h_w.split(',')
            train_data = SDSyntheticDataset(
                int(H),
                int(W),
                image_key=model_cfg.first_stage_key,
                txt_key=model_cfg.cond_stage_key,
                context_dim=model_cfg.unet_config.context_dim,
            )
        else:
            val_data = WebDatasetCommon(
                dataset_cfg=data_cfg,
                consumed_samples=consumed_samples,
                map_fn=transform_fn,
                compose_fn=tuple_to_dict,
                is_train=False,
            )

    return train_data, val_data


def build_train_valid_precached_clip_datasets(model_cfg, consumed_samples):
    data_cfg = model_cfg.data

    # This function maps data that are tuples to dictionary.
    def tuple_to_dict(inp):
        for input in inp:
            out_dict = dict()
            out_dict[model_cfg.first_stage_key] = input[0]
            out_dict[model_cfg.cond_stage_key] = input[1]
            yield out_dict

    def transform_fn(sample):
        latents, text_embed = sample["pyd"]["image_embed"], sample["pyd"]['captions_embed']
        latents = torch.from_numpy(latents)
        text_embed = torch.from_numpy(text_embed)

        # latents are of shape ([4, 64, 64])
        return latents, text_embed

    train_data = WebDatasetCommon(
        dataset_cfg=data_cfg,
        consumed_samples=consumed_samples,
        map_fn=transform_fn,
        compose_fn=tuple_to_dict,
        is_train=True,
    )

    val_data = None
    if data_cfg.get("validation") is not None and data_cfg.validation.get("data_path"):
        val_data = WebDatasetCommon(
            dataset_cfg=data_cfg,
            consumed_samples=consumed_samples,
            map_fn=transform_fn,
            compose_fn=tuple_to_dict,
            is_train=False,
        )

    return train_data, val_data
