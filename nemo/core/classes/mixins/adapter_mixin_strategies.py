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

import torch


class AbstractAdapterStrategy(ABC):
    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        """
        Forward method that defines how the output of the adapter should be merged with the input, or if it
        should be merged at all.

        Also provides the module that called this strategy - thereby allowing access to all other
        adapters in the calling module. This can be useful if one adapter is a meta adapter, that
        combines the outputs of various adapters. In such a case, the input can be forwarded across
        all other adapters, collecting their outputs, and those outputs can then be merged via some
        strategy. For example, refer to :

        - [AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://arxiv.org/abs/2005.00247)
        - [Exploiting Adapters for Cross-lingual Low-resource Speech Recognition](https://arxiv.org/abs/2105.11905)

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after all active adapters have finished their forward passes.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ResidualAddAdapterStrategy(AbstractAdapterStrategy):
    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        """
        A basic strategy, comprising of a residual connection over the input, after forward pass by
        the underlying adapter.

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after all active adapters have finished their forward passes.
        """
        out = adapter(input)
        return input + out
