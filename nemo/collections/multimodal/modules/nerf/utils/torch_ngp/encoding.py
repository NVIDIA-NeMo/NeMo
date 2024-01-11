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
import torch.nn as nn


class FreqEncoder_torch(nn.Module):
    def __init__(
        self,
        input_dim,
        max_freq_log2,
        N_freqs,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
    ):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.N_freqs = N_freqs

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, max_level=None, **kwargs):

        if max_level is None:
            max_level = self.N_freqs
        else:
            max_level = int(max_level * self.N_freqs)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(max_level):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        # append 0
        if self.N_freqs - max_level > 0:
            out.append(
                torch.zeros(
                    input.shape[0],
                    (self.N_freqs - max_level) * 2 * input.shape[1],
                    device=input.device,
                    dtype=input.dtype,
                )
            )

        out = torch.cat(out, dim=-1)

        return out


def get_encoder(
    encoder_type,
    input_dim=3,
    multires=6,
    degree=4,
    num_levels=16,
    level_dim=2,
    base_resolution=16,
    log2_hashmap_size=19,
    desired_resolution=2048,
    align_corners=False,
    interpolation='linear',
    **kwargs
):

    if encoder_type is None:
        return lambda x, **kwargs: x, input_dim

    elif encoder_type == 'frequency_torch':
        encoder = FreqEncoder_torch(
            input_dim=input_dim, max_freq_log2=multires - 1, N_freqs=multires, log_sampling=True
        )

    elif encoder_type == 'frequency':  # CUDA implementation, faster than torch.
        from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.freqencoder import FreqEncoder

        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoder_type == 'sphere_harmonics':
        from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.shencoder import SHEncoder

        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoder_type == 'hashgrid':
        from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.gridencoder import GridEncoder

        encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            gridtype='hash',
            align_corners=align_corners,
            interpolation=interpolation,
        )

    elif encoder_type == 'tiledgrid':
        from nemo.collections.multimodal.modules.nerf.utils.torch_ngp.gridencoder import GridEncoder

        encoder = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution,
            gridtype='tiled',
            align_corners=align_corners,
            interpolation=interpolation,
        )

    else:
        raise NotImplementedError(
            'Unknown encoder type, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]'
        )

    return encoder, encoder.output_dim
