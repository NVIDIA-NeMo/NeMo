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

from dataclasses import dataclass
from typing import Dict, Optional

from torch.utils.data import DataLoader

from nemo.collections.cv.datasets import MNISTDataset, MNISTDatasetConfig
from nemo.collections.cv.losses import NLLLoss
from nemo.collections.cv.modules import LeNet5 as LeNet5Module
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck
from nemo.core.config import Config, CosineAnnealingParams, DataLoaderConfig, NovogradParams
from nemo.core.neural_types import *


@dataclass
class NovogradScheduler(Config):
    """
    Scheduler setup for novograd
    """

    # Mandatory parameters
    name: str = "CosineAnnealing"
    args: CosineAnnealingParams = CosineAnnealingParams()  # can be a string to param, or "auto", just like in YAML

    # pytorch lightning parameters
    monitor: str = "val_loss"
    iters_per_batch: Optional[float] = None  # computed at runtime
    max_steps: Optional[int] = None  # computed at runtime or explicitly set here
    reduce_on_plateau: bool = False


@dataclass
class MNISTOptimizer(Config):
    """
    Optimizer setup for novograd
    """

    # Mandatory arguments
    name: str = "novograd"
    lr: float = 0.01

    args: NovogradParams = NovogradParams(betas=(0.8, 0.5))
    # sched: NovogradScheduler = NovogradScheduler()


@dataclass
class MNISTLeNet5Config(Config):
    """
    Structured config for LeNet-5 model class - that also contains parameters of dataset and dataloader.
    """

    dataset: MNISTDatasetConfig = MNISTDatasetConfig(width=32, height=32)
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, shuffle=True)
    module: Config = Config()
    optim: MNISTOptimizer = MNISTOptimizer()


class MNISTLeNet5(ModelPT):
    """
    The LeNet-5 convolutional model.
    """

    def __init__(self, cfg: MNISTLeNet5Config = MNISTLeNet5Config()):
        super().__init__(cfg=cfg)

        # Initialize modules.
        self.module = LeNet5Module(cfg.module)
        self.loss = NLLLoss()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` input types.
        """
        return self.module.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """
        Returns:
            :class:`LeNet5Module` output types.
        """
        return self.module.output_types

    @typecheck()
    def forward(self, images):
        """ Propagates data by calling the module :class:`LeNet5Module` forward. """
        return self.module.forward(images=images)

    def setup_training_data(self, train_data_layer_config: Optional[Dict] = None):
        """ Creates dataset, wrap it with dataloader and return the latter """
        # Instantiate dataset.
        mnist_ds = MNISTDataset(self._cfg.dataset)
        # Configure data loader.
        train_dataloader = DataLoader(dataset=mnist_ds, **(self._cfg.dataloader))
        self._train_dl = train_dataloader

    def setup_validation_data(self, val_data_layer_config: Optional[Dict] = None):
        """ Not implemented. """
        self._val_dl = None

    def setup_test_data(self, test_data_layer_params: Optional[Dict] = None):
        """ Not implemented. """
        self._test_dl = None

    def training_step(self, batch, what_is_this_input):
        """ Training step, calculate loss. """
        # "Unpack" the batch.
        _, images, targets, _ = batch

        # Get predictions.
        predictions = self(images=images)

        # Calculate loss.
        loss = self.loss(predictions=predictions, targets=targets)

        # Return it.
        return {"loss": loss}
        # of course "return loss" doesn't work :]

    def train_dataloader(self):
        """ Not implemented. """
        return self._train_dl

    def save_to(self, save_path: str):
        """ Not implemented. """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """ Not implemented. """
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """ Not implemented. """
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        """ Not implemented. """
        pass
