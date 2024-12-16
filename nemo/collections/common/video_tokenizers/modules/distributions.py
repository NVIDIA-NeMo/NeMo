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
"""The distribution modes to use for continuous image tokenizers."""

import torch


class IdentityDistribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parameters):
        return parameters, (torch.tensor([0.0]), torch.tensor([0.0]))


class GaussianDistribution(torch.nn.Module):
    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    def forward(self, parameters):
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return self.sample(mean, logvar), (mean, logvar)
