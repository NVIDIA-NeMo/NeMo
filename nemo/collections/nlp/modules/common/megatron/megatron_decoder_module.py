# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC
from typing import Dict, List, Optional

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import ChannelType, MaskType, NeuralType

__all__ = ['MegatronDecoderModule']


class MegatronDecoderModule(NeuralModule, ABC):
    """ Base class for encoder neural module to be used in NLP models. """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "dec_input": NeuralType(('B', 'T', 'D'), ChannelType()),
            "dec_attn_mask": NeuralType(('B', 'T'), MaskType()),
            "enc_output": NeuralType(('B', 'T', 'D'), ChannelType()),
            "enc_attn_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['dec_input', 'dec_attn_mask', 'enc_output', 'enc_attn_mask']

    @property
    def output_names(self) -> List[str]:
        return ['decoder_output']

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"dec_output": NeuralType(('B', 'T', 'D'), ChannelType())}
