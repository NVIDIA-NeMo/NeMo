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

__all__ = ['MegatronTokensHeadModule']


class MegatronTokensHeadModule(NeuralModule, ABC):
    """ Base class for encoder neural module to be used in NLP models. """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "dec_output": NeuralType(('B', 'T', 'D'), ChannelType()),
            "embeddings_weights": NeuralType(('T', 'D'), MaskType()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['dec_output', 'embeddings_weights']

    @property
    def output_names(self) -> List[str]:
        return ['logits']

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'T', 'D'), ChannelType())}
