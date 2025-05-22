# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf import DictConfig
from torch import nn

from nemo.core import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType


class AudioPerceptionModule(NeuralModule, Exportable):
    """Audio perception module that consists of audio encoder(s) and modality adapter."""

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        batch_size = torch.randint(low=1, high=max_batch, size=[1]).item()
        max_length = torch.randint(low=min_length, high=max_dim, size=[1]).item()
        signals = torch.rand(size=[batch_size, max_length]) * 2 - 1
        lengths = torch.randint(low=min_length, high=max_dim, size=[batch_size])
        lengths[0] = max_length
        return signals, lengths, None, None

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "input_signal": NeuralType(("B", "T"), AudioSignal(freq=self.preprocessor._sample_rate)),
            "input_signal_length": NeuralType(
                tuple("B"), LengthsType()
            ),  # Please note that length should be in samples not seconds.
            "processed_signal": NeuralType(("B", "D", "T"), SpectrogramType()),
            "processed_signal_length": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {
            "encoded": NeuralType(("B", "T", "D"), AcousticEncodedRepresentation()),
            "encoded_len": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Initialize components
        self.cfg = cfg
        self.preprocessor = self.from_config_dict(cfg.preprocessor)
        self.encoder = self.from_config_dict(cfg.encoder)

        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    # disable type checks to avoid type-check errors when using Conformer as modality adapter
    @typecheck.disable_checks()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)

        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))

        return encoded, encoded_len


class IdentityConnector(NeuralModule, Exportable):
    """User to pass encoder's representations as-is to the LLM."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def forward(self, audio_signal, length=None, *args, **kwargs):
        return audio_signal, length
