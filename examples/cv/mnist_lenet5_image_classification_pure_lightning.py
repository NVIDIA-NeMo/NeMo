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

# from hydra.utils import instantiate
import pytorch_lightning as ptl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.cv.models import MNISTLeNet5, MNISTLeNet5Config
from nemo.core.config import Config, TrainerConfig, hydra_runner
from nemo.utils import logging


@dataclass
class AppConfig(Config):
    """
    This is structured config for this application.

    Args:
        name: Description of the application.
        trainer: configuration of the trainer.
        model: configuation of the model.
    """

    name: str = "Training of a LeNet-5 Model using a pure PyTorchLightning approach - using DDP on 2 GPUs."
    trainer: TrainerConfig = TrainerConfig(devices=2, accelerator="gpu", strategy="ddp")
    model: MNISTLeNet5Config = MNISTLeNet5Config()


@hydra_runner(schema=AppConfig, config_name="AppConfig")
def main(cfg: DictConfig):
    # Show configuration - user can modify every parameter from command line!
    logging.info("Application config\n" + OmegaConf.to_yaml(cfg))

    # The "model" - with dataloader/dataset inside of it.
    lenet5 = MNISTLeNet5(cfg.model)

    # Setup train data loader and optimizer
    lenet5.setup_training_data()

    # Setup optimizer and scheduler
    lenet5.setup_optimization()

    # Create trainer.
    trainer = ptl.Trainer(**(cfg.trainer))
    # trainer = instantiate(cfg.trainer)

    # Train.
    trainer.fit(model=lenet5)


if __name__ == "__main__":
    main()  # TODO: No cfg in function call, and no hydra runner
