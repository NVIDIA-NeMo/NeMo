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
from typing import List, Optional

import torch

from nemo.utils import logging


def build_resolution_filter(value=None, method='larger', image_idx=0):
    """
        Filter image based on its resolution.
        value: filter threshold
        method: Either larger or smaller
        image_idx: idx of the image in the tuple input
    """
    assert method == 'larger' or method == 'smaller'
    if method == 'larger':
        logging.info(f'Only Selecting images with resolution >= {value}')
        return lambda x: x[image_idx].size[0] >= value and x[image_idx].size[1] >= value

    logging.info(f'Only Selecting images with resolution <= {value}')
    return lambda x: x[image_idx].size[0] <= value and x[image_idx].size[1] <= value


class PickleTransform:
    """
        Convert encodings stored in the pickle file to encoding and mask.
        Transform the pad and resize the embedding to match the generator config.
    """

    def __init__(self, encoding_lengths: List[int], encoding_keys: List[str], out_keys: Optional[List[str]] = None):
        assert len(encoding_keys) == len(encoding_lengths)
        self.encoding_lengths = encoding_lengths
        self.encoding_keys = encoding_keys
        self.out_keys = out_keys if out_keys is not None else encoding_keys

    def _pad_and_resize(self, arr, ntokens):
        # Function for padding and resizing a numpy array

        arr = torch.tensor(arr)
        embed_dim = arr.shape[1]

        arr_padded = torch.zeros(ntokens, embed_dim, device=arr.device, dtype=torch.float32)

        # If the input text is larger than num_text_tokens, clip it.
        if arr.shape[0] > ntokens:
            arr = arr[0:ntokens]

        mask = torch.LongTensor(ntokens).zero_()
        if len(arr.shape) > 1:
            mask[0 : arr.shape[0]] = 1

        if len(arr.shape) > 1:
            arr_padded[0 : arr.shape[0]] = arr

        return arr_padded, mask

    def __call__(self, data):
        out_dict = dict()
        for token_length, encoding_key, out_key in zip(self.encoding_lengths, self.encoding_keys, self.out_keys):
            embed, mask = self._pad_and_resize(data[encoding_key]['encodings'], token_length)
            out_dict[f'{out_key}_embeddings'] = embed
            out_dict[f'{out_key}_mask'] = mask
        return out_dict
