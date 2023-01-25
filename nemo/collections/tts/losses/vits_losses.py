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

# MIT License
#
# Copyright (c) 2021 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The forward functions of the following classes are based on code from https://github.com/jaywalnut310/vits:
# KlLoss

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LossType, VoidType
from nemo.core.neural_types.neural_type import NeuralType


class KlLoss(Loss):
    @property
    def input_types(self):
        return {
            "z_p": [NeuralType(('B', 'D', 'T'), VoidType())],
            "logs_q": [NeuralType(('B', 'D', 'T'), VoidType())],
            "m_p": [NeuralType(('B', 'D', 'T'), VoidType())],
            "logs_p": [NeuralType(('B', 'D', 'T'), VoidType())],
            "z_mask": [NeuralType(('B', 'D', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p: Input distribution
        logs_q: LogVariance of target distrubution
        m_p: Mean of input distrubution
        logs_p: LogVariance of input distrubution
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l


class FeatureMatchingLoss(Loss):
    """VITS Feature Matching Loss module"""

    @property
    def input_types(self):
        return {
            "fmap_r": [[NeuralType(elements_type=VoidType())]],
            "fmap_g": [[NeuralType(elements_type=VoidType())]],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, fmap_r, fmap_g):
        """
        fmap_r, fmap_g: List[List[Tensor]]
        """
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


class DiscriminatorLoss(Loss):
    """Discriminator Loss module"""

    @property
    def input_types(self):
        return {
            "disc_real_outputs": [NeuralType(('B', 'T'), VoidType())],
            "disc_generated_outputs": [NeuralType(('B', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "real_losses": [NeuralType(elements_type=LossType())],
            "fake_losses": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, disc_real_outputs, disc_generated_outputs):
        r_losses = []
        g_losses = []
        loss = 0
        for i, (dr, dg) in enumerate(zip(disc_real_outputs, disc_generated_outputs)):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


class GeneratorLoss(Loss):
    """Generator Loss module"""

    @property
    def input_types(self):
        return {
            "disc_outputs": [NeuralType(('B', 'T'), VoidType())],
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "fake_losses": [NeuralType(elements_type=LossType())],
        }

    @typecheck()
    def forward(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
