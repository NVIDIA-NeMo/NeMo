# ******************************************************************************
# Copyright (C) 2024 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ******************************************************************************
from enum import Enum

from nemo.collections.common.video_tokenizers.modules.distributions import GaussianDistribution, IdentityDistribution
from nemo.collections.common.video_tokenizers.modules.layers2d import Decoder, Encoder
from nemo.collections.common.video_tokenizers.modules.layers3d import (
    DecoderBase,
    DecoderFactorized,
    EncoderBase,
    EncoderFactorized,
)
from nemo.collections.common.video_tokenizers.modules.quantizers import (
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
    VectorQuantizer,
)


class EncoderType(Enum):
    Default = Encoder


class DecoderType(Enum):
    Default = Decoder


class Encoder3DType(Enum):
    BASE = EncoderBase
    FACTORIZED = EncoderFactorized


class Decoder3DType(Enum):
    BASE = DecoderBase
    FACTORIZED = DecoderFactorized


class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution


class DiscreteQuantizer(Enum):
    VQ = VectorQuantizer
    LFQ = LFQuantizer
    FSQ = FSQuantizer
    RESFSQ = ResidualFSQuantizer
