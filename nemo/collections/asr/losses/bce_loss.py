# ! /usr/bin/python
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

from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LossType, NeuralType, ProbsType

__all__ = ['BCELoss']


class BCELoss(Loss, Typing):
    """
    Computes Binary Cross Entropy (BCE) loss. The BCELoss class expects output from Sigmoid function.
    """

    @property
    def input_types(self):
        """Input types definitions for AnguarLoss.
        """
        return {
            "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
            "signal_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """
        Output types definitions for binary cross entropy loss. Weights for labels can be set using weight variables.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='sum', alpha=1.0, weight=torch.tensor([0.5, 0.5])):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = weight
        self.loss_f = torch.nn.BCELoss(weight=self.loss_weight, reduction=self.reduction)

    @typecheck()
    def forward(self, probs, labels, signal_lengths):
        """
        Calculate binary cross entropy loss based on probs, labels and signal_lengths variables.

        Args:
            probs (torch.tensor)
                Predicted probability value which ranges from 0 to 1. Sigmoid output is expected.
            labels (torch.tensor)
                Groundtruth label for the predicted samples.
            signal_lengths (torch.tensor):
                The actual length of the sequence without zero-padding.

        Returns:
            loss (NeuralType)
                Binary cross entropy loss value.
        """
        probs_list = [probs[k, : signal_lengths[k], :] for k in range(probs.shape[0])]
        targets_list = [labels[k, : signal_lengths[k], :] for k in range(labels.shape[0])]
        probs = torch.cat(probs_list, dim=0)
        labels = torch.cat(targets_list, dim=0)
        return self.loss_f(probs, labels)
