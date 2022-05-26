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

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType, SpectrogramType

__all__ = ["MLMLoss"]


class MLMLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive.
        """
        return {
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(("B", "T", "D"), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return True

    def __init__(
        self, combine_time_steps: int = 1, mask_threshold: float = 0.8,
    ):
        super().__init__()
        self.nll_loss = nn.NLLLoss()
        self.combine_time_steps = combine_time_steps
        self.mask_threshold = mask_threshold

    @typecheck()
    def forward(self, spec_masks, decoder_outputs, targets, decoder_lengths=None, target_lengths=None):

        # outputs are log_probs
        masks = spec_masks.transpose(-2, -1)
        # BxTxC

        masks = masks.reshape(masks.shape[0], masks.shape[1] // self.combine_time_steps, -1)
        masks = masks.mean(-1) > self.mask_threshold

        out_masked_only = decoder_outputs[masks]
        targets = F.pad(targets, (0, masks.shape[-1] - targets.shape[-1]))
        targets_masked_only = targets[masks]

        loss = self.nll_loss(out_masked_only, targets_masked_only)
        loss = torch.mean(loss)

        return loss
