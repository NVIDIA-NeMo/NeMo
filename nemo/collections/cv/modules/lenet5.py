# -*- coding: utf-8 -*-
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from torch import nn

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.config import Config
from nemo.core.neural_types import AxisKind, AxisType, ImageValue, LogprobsType, NeuralType


class LeNet5(NeuralModule):
    """
    Classical LeNet-5 model for MNIST image classification.
    """

    def __init__(self, cfg: Config = Config()):
        """
        Creates the LeNet-5 model.

        Args:
            cfg: Default NeMo config containing name.
        """
        # Call the base class constructor.
        super().__init__()  # name=name)

        # Create the LeNet-5 model.
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(),
            # reshape to [-1, 120]
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1),
        )

    @property
    def input_types(self):
        """ Returns definitions of module input ports. """
        return {
            "images": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=1),
                    AxisType(kind=AxisKind.Height, size=32),
                    AxisType(kind=AxisKind.Width, size=32),
                ),
                elements_type=ImageValue(),
            )
        }

    @property
    def output_types(self):
        """ Returns definitions of module output ports. """
        return {
            "predictions": NeuralType(
                axes=(AxisType(kind=AxisKind.Batch), AxisType(kind=AxisKind.Dimension)), elements_type=LogprobsType()
            )
        }

    @typecheck()
    def forward(self, images):
        """
        Performs the forward step of the LeNet-5 model.

        Args:
            images: Batch of images to be classified.

        Returns:
            Batch of predictions.
        """

        predictions = self.model(images)
        return predictions

    def save_to(self, save_path: str):
        """Not implemented yet.
           Serialize model.

        Args:
            save_path (str): path to save serialization.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """ Not implemented yet.
            Restore module from serialization.

        Args:
            restore_path (str): path to serialization
        """
        pass
