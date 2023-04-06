# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import random

import numpy as np
import torch
import torch.nn as nn

from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType


class SpecAugment(nn.Module, Typing):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types
        """
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types
        """
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=10, rng=None, mask_value=0.0,
    ):
        super().__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = mask_value

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError("If `time_width` is a float value, must be in range [0, 1]")

            self.adaptive_temporal_width = True

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec, length):
        batch_size, num_freq_bins, _ = input_spec.shape
        # Move lengths to CPU before repeated indexing
        lengths_cpu = length.cpu().numpy()
        # Generate a numpy boolean mask. `True` elements represent where the input spec will be augmented.
        fill_mask: np.array = np.full(shape=input_spec.shape, fill_value=False)
        freq_start_upper_bound = num_freq_bins - self.freq_width
        # Choose different mask ranges for each element of the batch
        for idx in range(batch_size):
            # Set freq masking
            for _ in range(self.freq_masks):
                start = self._rng.randint(0, freq_start_upper_bound)
                width = self._rng.randint(0, self.freq_width)
                fill_mask[idx, start : start + width, :] = True

            # Derive time width, sometimes based percentage of input length.
            if self.adaptive_temporal_width:
                time_max_width = max(1, int(lengths_cpu[idx] * self.time_width))
            else:
                time_max_width = self.time_width
            time_start_upper_bound = max(1, lengths_cpu[idx] - time_max_width)

            # Set time masking
            for _ in range(self.time_masks):
                start = self._rng.randint(0, time_start_upper_bound)
                width = self._rng.randint(0, time_max_width)
                fill_mask[idx, :, start : start + width] = True
        # Bring the mask to device and fill spec
        fill_mask = torch.from_numpy(fill_mask).to(input_spec.device)
        masked_spec = input_spec.masked_fill(mask=fill_mask, value=self.mask_value)
        return masked_spec


class SpecCutout(nn.Module, Typing):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    @property
    def input_types(self):
        """Returns definitions of module input types
        """
        return {"input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    @property
    def output_types(self):
        """Returns definitions of module output types
        """
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super(SpecCutout, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec):
        sh = input_spec.shape

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = self._rng.randint(0, sh[1] - self.rect_freq)
                rect_y = self._rng.randint(0, sh[2] - self.rect_time)

                w_x = self._rng.randint(0, self.rect_freq)
                w_y = self._rng.randint(0, self.rect_time)

                input_spec[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

        return input_spec
