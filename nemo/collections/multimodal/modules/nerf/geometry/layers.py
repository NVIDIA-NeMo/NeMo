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
from typing import Callable, List, Type, Union

import torch
import torch.nn as nn

BlockBuilder = Union[Callable[[int, int, bool], nn.Module], Type[nn.Module], None]


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) module.

    Args:
        num_input_dims (int): Number of input dimensions.
        num_output_dims (int): Number of output dimensions.
        num_hidden_dims (int): Number of hidden dimensions.
        num_layers (int): Number of layers in the MLP.
        bias (bool): If True, enables the bias in Linear layers. Default is True.
        block (BlockBuilder): A callable or class for constructing a block. Default is None.
    """

    def __init__(
        self,
        num_input_dims: int,
        num_output_dims: int,
        num_hidden_dims: int,
        num_layers: int,
        bias: bool = True,
        block: BlockBuilder = None,
    ):
        super().__init__()

        # Initialize the network as an empty list
        network = []

        # Add input layer
        network.append(nn.Linear(num_input_dims, num_hidden_dims, bias=bias))
        network.append(nn.ReLU(inplace=True))

        # Add hidden layers
        for _ in range(1, num_layers - 1):
            network.extend(self.build_layer(num_hidden_dims, num_hidden_dims, bias, block))

        # Add output layer
        network.append(nn.Linear(num_hidden_dims, num_output_dims, bias=bias))

        # Wrap layers in ModuleList for proper registration
        self.net = nn.ModuleList(network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for module in self.net:
            x = module(x)
        return x

    @staticmethod
    def build_layer(
        num_input_dims: int, num_output_dims: int, bias: bool = True, block_builder: BlockBuilder = None
    ) -> List[nn.Module]:
        """
        Build a single layer for the MLP.

        Args:
            num_input_dims (int): Number of input dimensions.
            num_output_dims (int): Number of output dimensions.
            bias (bool): If True, enables the bias in Linear layers. Default is True.
            block_builder (BlockBuilder): A callable or class for constructing a block. Default is None.

        Returns:
            List[nn.Module]: A list containing the layer's modules.
        """
        if block_builder is None:
            return [nn.Linear(num_input_dims, num_output_dims, bias=bias), nn.ReLU(inplace=True)]
        else:
            return [block_builder(num_input_dims, num_output_dims, bias=bias)]


class ResBlock(nn.Module):
    """
    A residual block module.

    Args:
        num_input_dims (int): Number of input dimensions.
        num_output_dims (int): Number of output dimensions.
        bias (bool): If True, enables the bias in Linear layers. Default is True.
    """

    def __init__(self, num_input_dims: int, num_output_dims: int, bias: bool = True):
        super().__init__()

        self.dense = nn.Linear(num_input_dims, num_output_dims, bias=bias)
        self.norm = nn.LayerNorm(num_output_dims)
        self.activation = nn.SiLU(inplace=True)

        if num_input_dims != num_output_dims:
            self.skip = nn.Linear(num_input_dims, num_output_dims, bias=False)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out
