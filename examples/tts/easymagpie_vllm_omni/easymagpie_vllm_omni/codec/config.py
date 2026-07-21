# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Transformers configuration for the native EasyMagpie codec."""

from __future__ import annotations

from transformers import PretrainedConfig


class EasyMagpieCodecConfig(PretrainedConfig):
    """Configuration of the 25-fps causal spectral codec decoder.

    The model consumes one packed row per EasyMagpie acoustic frame. Each row
    contains ``num_codebooks * frame_stacking_factor`` scalar FSQ indices.
    """

    model_type = "easymagpie_codec"

    def __init__(
        self,
        *,
        input_dim: int = 40,
        input_filters: int = 768,
        hidden_filters: int = 1536,
        num_hidden_layers: int = 6,
        pre_upsample_rates: list[int] | None = None,
        pre_upsample_filters: list[int] | None = None,
        resblock_upsample_rates: list[int] | None = None,
        resblock_upsample_filters: list[int] | None = None,
        kernel_size: int = 3,
        resblock_kernel_size: int = 7,
        activation: str = "half_snake",
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        num_levels_per_group: list[int] | None = None,
        frame_stacking_factor: int = 2,
        output_sample_rate: int = 22050,
        **kwargs,
    ) -> None:
        kwargs.setdefault("architectures", ["EasyMagpieCodecForConditionalGeneration"])
        kwargs.setdefault("torch_dtype", "float32")
        super().__init__(**kwargs)
        self.input_dim = int(input_dim)
        self.input_filters = int(input_filters)
        self.hidden_filters = int(hidden_filters)
        self.num_hidden_layers = int(num_hidden_layers)
        self.pre_upsample_rates = list(pre_upsample_rates or [2])
        self.pre_upsample_filters = list(pre_upsample_filters or [768])
        self.resblock_upsample_rates = list(resblock_upsample_rates or [9, 7, 7])
        self.resblock_upsample_filters = list(resblock_upsample_filters or [384, 128, 32])
        self.kernel_size = int(kernel_size)
        self.resblock_kernel_size = int(resblock_kernel_size)
        self.activation = activation
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.num_levels_per_group = list(num_levels_per_group or [4, 4, 4, 4, 4])
        self.frame_stacking_factor = int(frame_stacking_factor)
        self.output_sample_rate = int(output_sample_rate)
        # Minimal language-model-shaped fields used by generic vLLM input allocation.
        self.vocab_size = max(self.codebook_size + 1, 2)
        self.hidden_size = 1

        if len(self.pre_upsample_rates) != len(self.pre_upsample_filters):
            raise ValueError("pre_upsample_rates and pre_upsample_filters must have the same length")
        if len(self.resblock_upsample_rates) != len(self.resblock_upsample_filters):
            raise ValueError("resblock_upsample_rates and resblock_upsample_filters must have the same length")
        if self.input_dim != self.num_codebooks * len(self.num_levels_per_group):
            raise ValueError(
                "input_dim must equal num_codebooks * len(num_levels_per_group), got "
                f"{self.input_dim} and {self.num_codebooks} * {len(self.num_levels_per_group)}"
            )

    @property
    def num_stacked_codebooks(self) -> int:
        return self.num_codebooks * self.frame_stacking_factor

    @property
    def samples_per_codec_frame(self) -> int:
        factor = 1
        for rate in self.pre_upsample_rates + self.resblock_upsample_rates:
            factor *= rate
        return factor

    @property
    def samples_per_frame(self) -> int:
        return self.frame_stacking_factor * self.samples_per_codec_frame
