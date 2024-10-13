# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import warnings
import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from megatron.energon import DefaultTaskEncoder, SkipSample
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys

from nemo.lightning.io.mixin import IOMixin


def cook(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'json': The contains meta data like resolution, aspect ratio, fps, etc.
            - 'pth': contains video latent tensor
            - 'pickle': contains text embeddings
    """
    return dict(
        **basic_sample_keys(sample),
        json=sample['.json'],
        pth=sample['.pth'],
        pickle=sample['.pickle'],
    )


class BasicDiffusionTaskEncoder(DefaultTaskEncoder, IOMixin):
    """
    BasicDiffusionTaskEncoder is a class that encodes image/video samples for diffusion tasks.
    Attributes:
        cookers (list): A list of Cooker objects used for processing.
        max_frames (int, optional): The maximum number of frames to consider from the video. Defaults to None.
        text_embedding_padding_size (int): The padding size for text embeddings. Defaults to 512.
    Methods:
        __init__(*args, max_frames=None, text_embedding_padding_size=512, **kwargs):
            Initializes the BasicDiffusionTaskEncoder with optional maximum frames and text embedding padding size.
        encode_sample(sample: dict) -> dict:
            Encodes a given sample dictionary containing video and text data.
            Args:
                sample (dict): A dictionary containing 'pth' for video latent and 'json' for additional info.
            Returns:
                dict: A dictionary containing encoded video, text embeddings, text mask, and loss mask.
            Raises:
                SkipSample: If the video latent contains NaNs, Infs, or is not divisible by the tensor parallel size.
    """

    cookers = [
        Cooker(cook),
    ]

    def __init__(
        self,
        *args,
        max_frames: int = None,
        text_embedding_padding_size: int = 512,
        seq_length: int = None,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_frames = max_frames
        self.text_embedding_padding_size = text_embedding_padding_size
        self.seq_length = seq_length
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal

    def encode_sample(self, sample: dict) -> dict:
        video_latent = sample['pth']

        if torch.isnan(video_latent).any() or torch.isinf(video_latent).any():
            raise SkipSample()
        if torch.max(torch.abs(video_latent)) > 1e3:
            raise SkipSample()

        info = sample['json']
        C, T, H, W = video_latent.shape
        seq_len = (
            video_latent.shape[-1]
            * video_latent.shape[-2]
            * video_latent.shape[-3]
            // self.patch_spatial**2
            // self.patch_temporal
        )
        is_image = T == 1

        if seq_len > self.seq_length:
            raise SkipSample()

        if self.max_frames is not None:
            video_latent = video_latent[:, : self.max_frames, :, :]

        tpcp_size = parallel_state.get_tensor_model_parallel_world_size()
        if parallel_state.get_context_parallel_world_size() > 1:
            tpcp_size *= parallel_state.get_context_parallel_world_size() * 2
        if (T * H * W) % tpcp_size != 0:
            warnings.warn(f'skipping {video_latent.shape=} not divisible by {tpcp_size=}')
            raise SkipSample()

        video_latent = rearrange(
            video_latent,
            'C (T pt) (H ph) (W pw) -> (T H W) (ph pw pt C)',
            ph=self.patch_spatial,
            pw=self.patch_spatial,
            pt=self.patch_temporal,
        )

        if is_image:
            t5_text_embeddings = torch.from_numpy(sample['pickle']).to(torch.bfloat16)
        else:
            t5_text_embeddings = torch.from_numpy(sample['pickle'][0]).to(torch.bfloat16)
        t5_text_embeddings_seq_length = t5_text_embeddings.shape[0]

        if t5_text_embeddings_seq_length > self.text_embedding_padding_size:
            t5_text_embeddings = t5_text_embeddings[: self.text_embedding_padding_size]
        else:
            t5_text_embeddings = F.pad(
                t5_text_embeddings,
                (
                    0,
                    0,
                    0,
                    self.text_embedding_padding_size - t5_text_embeddings_seq_length,
                ),
            )
        t5_text_mask = torch.ones(t5_text_embeddings_seq_length, dtype=torch.bfloat16)

        if is_image:
            h, w = info['image_height'], info['image_width']
            fps = torch.tensor([30] * 1, dtype=torch.bfloat16)
            num_frames = torch.tensor([1] * 1, dtype=torch.bfloat16)
        else:
            h, w = info['height'], info['width']
            fps = torch.tensor([info['framerate']] * 1, dtype=torch.bfloat16)
            num_frames = torch.tensor([info['num_frames']] * 1, dtype=torch.bfloat16)
        image_size = torch.tensor([[h, w, h, w]] * 1, dtype=torch.bfloat16)

        pos_ids = rearrange(
            pos_id_3d.get_pos_id_3d(t=T // self.patch_temporal, h=H // self.patch_spatial, w=W // self.patch_spatial),
            'T H W d -> (T H W) d',
        )

        if self.seq_length is not None:
            pos_ids = F.pad(pos_ids, (0, 0, 0, self.seq_length - seq_len))
            loss_mask = torch.zeros(self.seq_length, dtype=torch.bfloat16)
            loss_mask[:seq_len] = 1
            video_latent = F.pad(video_latent, (0, 0, 0, self.seq_length - seq_len))
        else:
            loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)

        return dict(
            video=video_latent,
            t5_text_embeddings=t5_text_embeddings,
            t5_text_mask=t5_text_mask,
            image_size=image_size,
            fps=fps,
            num_frames=num_frames,
            loss_mask=loss_mask,
            seq_len_q=torch.tensor(seq_len, dtype=torch.int32),
            seq_len_kv=torch.tensor(t5_text_embeddings_seq_length, dtype=torch.int32),
            pos_ids=pos_ids,
            latent_shape=torch.tensor([C, T, H, W], dtype=torch.int32),
        )


class PosID3D:
    def __init__(self, *, max_t=32, max_h=128, max_w=128):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device='cpu'),
                torch.arange(self.max_h, device='cpu'),
                torch.arange(self.max_w, device='cpu'),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


pos_id_3d = PosID3D()
