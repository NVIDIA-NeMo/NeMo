# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for the inference libraries."""

import os
import re
from glob import glob

import mediapy as media
import numpy as np
import torch

from nemo.collections.common.video_tokenizers.networks import TokenizerConfigs, TokenizerModels

_DTYPE, _DEVICE = torch.bfloat16, "cuda"
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
_SPATIAL_ALIGN = 16
_TEMPORAL_ALIGN = 8


def load_jit_model(jit_filepath: str = None, device: str = "cuda") -> torch.jit.ScriptModule:
    """Loads a torch.jit.ScriptModule from a filepath.

    Args:
        jit_filepath: The filepath to the JIT-compiled model.
        device: The device to load the model onto, default=cuda.
    Returns:
        The JIT compiled model loaded to device and on eval mode.
    """
    model = torch.jit.load(jit_filepath)
    return model.eval().to(device)


def save_jit_model(
    model: torch.jit.ScriptModule | torch.jit.RecursiveScriptModule = None,
    jit_filepath: str = None,
) -> None:
    """Saves a torch.jit.ScriptModule or torch.jit.RecursiveScriptModule to file.

    Args:
        model: JIT compiled model loaded onto `config.checkpoint.jit.device`.
        jit_filepath: The filepath to the JIT-compiled model.
    """
    torch.jit.save(model, jit_filepath)


def get_filepaths(input_pattern) -> list[str]:
    """Returns a list of filepaths from a pattern."""
    filepaths = sorted(glob(str(input_pattern)))
    return list(set(filepaths))


def get_output_filepath(filepath: str, output_dir: str = None) -> str:
    """Returns the output filepath for the given input filepath."""
    output_dir = output_dir or f"{os.path.dirname(filepath)}/reconstructions"
    output_filepath = f"{output_dir}/{os.path.basename(filepath)}"
    os.makedirs(output_dir, exist_ok=True)
    return output_filepath


def read_image(filepath: str) -> np.ndarray:
    """Reads an image from a filepath.

    Args:
        filepath: The filepath to the image.

    Returns:
        The image as a numpy array, layout HxWxC, range [0..255], uint8 dtype.
    """
    image = media.read_image(filepath)
    # convert the grey scale image to RGB
    # since our tokenizers always assume 3-channel RGB image
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    # convert RGBA to RGB
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def read_video(filepath: str) -> np.ndarray:
    """Reads a video from a filepath.

    Args:
        filepath: The filepath to the video.
    Returns:
        The video as a numpy array, layout TxHxWxC, range [0..255], uint8 dtype.
    """
    video = media.read_video(filepath)
    # convert the grey scale frame to RGB
    # since our tokenizers always assume 3-channel video
    if video.ndim == 3:
        video = np.stack([video] * 3, axis=-1)
    # convert RGBA to RGB
    if video.shape[-1] == 4:
        video = video[..., :3]
    return video


def resize_image(image: np.ndarray, short_size: int = None) -> np.ndarray:
    """Resizes an image to have the short side of `short_size`.

    Args:
        image: The image to resize, layout HxWxC, of any range.
        short_size: The size of the short side.
    Returns:
        The resized image.
    """
    if short_size is None:
        return image
    height, width = image.shape[-3:-1]
    if height <= width:
        height_new, width_new = short_size, int(width * short_size / height + 0.5)
        width_new = width_new if width_new % 2 == 0 else width_new + 1
    else:
        height_new, width_new = (
            int(height * short_size / width + 0.5),
            short_size,
        )
        height_new = height_new if height_new % 2 == 0 else height_new + 1
    return media.resize_image(image, shape=(height_new, width_new))


def resize_video(video: np.ndarray, short_size: int = None) -> np.ndarray:
    """Resizes a video to have the short side of `short_size`.

    Args:
        video: The video to resize, layout TxHxWxC, of any range.
        short_size: The size of the short side.
    Returns:
        The resized video.
    """
    if short_size is None:
        return video
    height, width = video.shape[-3:-1]
    if height <= width:
        height_new, width_new = short_size, int(width * short_size / height + 0.5)
        width_new = width_new if width_new % 2 == 0 else width_new + 1
    else:
        height_new, width_new = (
            int(height * short_size / width + 0.5),
            short_size,
        )
        height_new = height_new if height_new % 2 == 0 else height_new + 1
    return media.resize_video(video, shape=(height_new, width_new))


def write_image(filepath: str, image: np.ndarray):
    """Writes an image to a filepath."""
    return media.write_image(filepath, image)


def write_video(filepath: str, video: np.ndarray, fps: int = 24) -> None:
    """Writes a video to a filepath."""
    return media.write_video(filepath, video, fps=fps)


def numpy2tensor(
    input_image: np.ndarray,
    dtype: torch.dtype = _DTYPE,
    device: str = _DEVICE,
    range_min: int = -1,
) -> torch.Tensor:
    """Converts image(dtype=np.uint8) to `dtype` in range [0..255].

    Args:
        input_image: A batch of images in range [0..255], BxHxWx3 layout.
    Returns:
        A torch.Tensor of layout Bx3xHxW in range [-1..1], dtype.
    """
    ndim = input_image.ndim
    indices = list(range(1, ndim))[-1:] + list(range(1, ndim))[:-1]
    image = input_image.transpose((0,) + tuple(indices)) / _UINT8_MAX_F
    if range_min == -1:
        image = 2.0 * image - 1.0
    return torch.from_numpy(image).to(dtype).to(device)


def tensor2numpy(input_tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    """Converts tensor in [-1,1] to image(dtype=np.uint8) in range [0..255].

    Args:
        input_tensor: Input image tensor of Bx3xHxW layout, range [-1..1].
    Returns:
        A numpy image of layout BxHxWx3, range [0..255], uint8 dtype.
    """
    if range_min == -1:
        input_tensor = (input_tensor.float() + 1.0) / 2.0
    ndim = input_tensor.ndim
    output_image = input_tensor.clamp(0, 1).cpu().numpy()
    output_image = output_image.transpose((0,) + tuple(range(2, ndim)) + (1,))
    return (output_image * _UINT8_MAX_F + 0.5).astype(np.uint8)


def pad_image_batch(batch: np.ndarray, spatial_align: int = _SPATIAL_ALIGN) -> tuple[np.ndarray, list[int]]:
    """Pads a batch of images to be divisible by `spatial_align`.

    Args:
        batch: The batch of images to pad, layout BxHxWx3, in any range.
        align: The alignment to pad to.
    Returns:
        The padded batch and the crop region.
    """
    height, width = batch.shape[1:3]
    align = spatial_align
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [
        height_to_pad >> 1,
        width_to_pad >> 1,
        height + (height_to_pad >> 1),
        width + (width_to_pad >> 1),
    ]
    batch = np.pad(
        batch,
        (
            (0, 0),
            (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
            (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)),
            (0, 0),
        ),
        mode="constant",
    )
    return batch, crop_region


def pad_video_batch(
    batch: np.ndarray,
    temporal_align: int = _TEMPORAL_ALIGN,
    spatial_align: int = _SPATIAL_ALIGN,
) -> tuple[np.ndarray, list[int]]:
    """Pads a batch of videos to be divisible by `temporal_align` or `spatial_align`.

    Zero pad spatially. Reflection pad temporally to handle causality better.
    Args:
        batch: The batch of videos to pad., layout BxFxHxWx3, in any range.
        align: The alignment to pad to.
    Returns:
        The padded batch and the crop region.
    """
    num_frames, height, width = batch.shape[-4:-1]
    align = spatial_align
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    align = temporal_align
    frames_to_pad = (align - (num_frames - 1) % align) if (num_frames - 1) % align != 0 else 0

    crop_region = [
        frames_to_pad >> 1,
        height_to_pad >> 1,
        width_to_pad >> 1,
        num_frames + (frames_to_pad >> 1),
        height + (height_to_pad >> 1),
        width + (width_to_pad >> 1),
    ]
    batch = np.pad(
        batch,
        (
            (0, 0),
            (0, 0),
            (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
            (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)),
            (0, 0),
        ),
        mode="constant",
    )
    batch = np.pad(
        batch,
        (
            (0, 0),
            (frames_to_pad >> 1, frames_to_pad - (frames_to_pad >> 1)),
            (0, 0),
            (0, 0),
            (0, 0),
        ),
        mode="edge",
    )
    return batch, crop_region


def unpad_video_batch(batch: np.ndarray, crop_region: list[int]) -> np.ndarray:
    """Unpads video with `crop_region`.

    Args:
        batch: A batch of numpy videos, layout BxFxHxWxC.
        crop_region: [f1,y1,x1,f2,y2,x2] first, top, left, last, bot, right crop indices.

    Returns:
        np.ndarray: Cropped numpy video, layout BxFxHxWxC.
    """
    assert len(crop_region) == 6, "crop_region should be len of 6."
    f1, y1, x1, f2, y2, x2 = crop_region
    return batch[..., f1:f2, y1:y2, x1:x2, :]


def unpad_image_batch(batch: np.ndarray, crop_region: list[int]) -> np.ndarray:
    """Unpads image with `crop_region`.

    Args:
        batch: A batch of numpy images, layout BxHxWxC.
        crop_region: [y1,x1,y2,x2] top, left, bot, right crop indices.

    Returns:
        np.ndarray: Cropped numpy image, layout BxHxWxC.
    """
    assert len(crop_region) == 4, "crop_region should be len of 4."
    y1, x1, y2, x2 = crop_region
    return batch[..., y1:y2, x1:x2, :]


def get_pytorch_model(jit_filepath: str = None, tokenizer_config: str = None):
    tokenizer_name = tokenizer_config["name"]
    model = TokenizerModels[tokenizer_name].value(**tokenizer_config)
    ckpts = torch.jit.load(jit_filepath)
    return model, ckpts


def load_pytorch_model(jit_filepath: str, tokenizer_config: dict, model_type: str, device):
    """Loads a torch.nn.Module from a filepath."""
    model, ckpts = get_pytorch_model(jit_filepath, tokenizer_config)
    if model_type == "enc":
        model = model.encoder_jit()
    elif model_type == "dec":
        model = model.decoder_jit()
    model.load_state_dict(ckpts.state_dict(), strict=False)
    return model.eval().to(tokenizer_config["dtype"]).to(device)


def get_tokenizer_config(tokenizer_type) -> TokenizerConfigs:
    """return tokeinzer config from tokenizer name"""
    match = re.match("Cosmos-Tokenizer-(\D+)(\d+)x(\d+).*", tokenizer_type)
    if match:
        name, temporal, spatial = match.groups()
        tokenizer_config = TokenizerConfigs[name].value
        tokenizer_config.update(dict(spatial_compression=int(spatial)))
        tokenizer_config.update(dict(temporal_compression=int(temporal)))
        return tokenizer_config
    return None
