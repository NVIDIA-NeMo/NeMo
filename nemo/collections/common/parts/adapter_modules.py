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

from dataclasses import dataclass

from torch import nn as nn

from nemo.collections.common.parts.utils import activation_registry


class LinearAdapter(nn.Module):
    """
    Simple Linear Feedforward Adapter module with LayerNorm and singe hidden layer with activation function.
    Note: The adapter explicitly initializes its final layer with all zeros in order to avoid affecting the
    original model when all adapters are disabled.

    Args:
        in_features: Input dimension of the module. Note that for adapters, input_dim == output_dim.
        dim: Hidden dimension of the feed forward network.
        activation: Str name for an activation function.
        norm_position: Str, can be `pre` or `post`. Defaults to `post`. Determines whether the normalization
            will occur in the first layer or the last layer. Certain architectures may prefer one over the other.
    """

    def __init__(self, in_features, dim, activation: str = 'swish', norm_position="post"):
        super().__init__()

        activation = activation_registry[activation]()
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        assert norm_position in ['pre', 'post']
        self.norm_position = norm_position

        if norm_position == 'pre':
            self.module = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
            )

        elif norm_position == 'post':
            self.module = nn.Sequential(
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
                nn.LayerNorm(in_features),
            )

        self.reset_parameters()

    def reset_parameters(self):
        # Final layer initializations must be 0
        if self.norm_position == 'pre':
            self.module[-1].weight.data *= 0

        elif self.norm_position == 'post':
            self.module[-1].weight.data *= 0
            self.module[-1].bias.data *= 0

    def forward(self, x):
        return self.module(x)


@dataclass
class LinearAdapterConfig:
    in_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'post'
    _target_: str = "{0}.{1}".format(LinearAdapter.__module__, LinearAdapter.__name__)
