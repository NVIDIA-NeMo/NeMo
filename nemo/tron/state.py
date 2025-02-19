# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import yaml
from megatron.core.timers import Timers
from torch.distributed.checkpoint.stateful import Stateful

from nemo.tron.config import ConfigContainer
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils import dump_dataclass_to_yaml, get_rank_safe, get_world_size_safe


def _timers_write_to_wandb(
        self,
        names: list[str],
        writer,
        iteration: int,
        normalizer: float = 1.0,
        reset: bool = True,
        barrier: bool = False,
    ):
        """Write timers to a tensorboard writer. Note that we only report maximum time across ranks
           to tensorboard.

        Args:
            names (List[str]): Names of the timers to log.
            writer (SummaryWriter): Tensorboard SummaryWriter object
            iteration (int): Current iteration.
            normalizer (float, optional): Normalizes the timer values by the factor.
                                          Defaults to 1.0.
            reset (bool, optional): Whether to reset timer values after logging. Defaults to True.
            barrier (bool, optional): Whether to do a global barrier before time measurments.
                                      Defaults to False.
        """
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
        if writer is not None:
            for name in name_to_min_max_time:
                _, max_time = name_to_min_max_time[name]
                writer.log({name + '-time': max_time}, iteration)

@dataclass
class TrainState(Stateful):
    step: int = 0
    consumed_train_samples: int = 0
    skipped_train_samples: int = 0
    consumed_valid_samples: int = 0
    variable_seq_lengths: bool = False
    do_train: bool = False
    do_valid: bool = False
    do_test: bool = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "consumed_train_samples": torch.tensor(self.consumed_train_samples, dtype=torch.int32),
            "skipped_train_samples": torch.tensor(self.skipped_train_samples, dtype=torch.int32),
            "consumed_valid_samples": torch.tensor(self.consumed_valid_samples, dtype=torch.int32),
            "variable_seq_lengths": torch.tensor(self.variable_seq_lengths, dtype=torch.bool),
            "do_train": torch.tensor(self.do_train, dtype=torch.bool),
            "do_valid": torch.tensor(self.do_valid, dtype=torch.bool),
            "do_test": torch.tensor(self.do_test, dtype=torch.bool),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = state_dict["step"].item()
        self.consumed_train_samples = state_dict["consumed_train_samples"].item()
        self.skipped_train_samples = state_dict["skipped_train_samples"].item()
        self.consumed_valid_samples = state_dict["consumed_valid_samples"].item()
        self.variable_seq_lengths = state_dict["variable_seq_lengths"].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()


# replacement for Megatron's global variables, except mbs calc and parallel state
class GlobalState:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent reinitialization in subsequent instantiations.
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.cfg = None
        self._tokenizer = None
        self._tensorboard_logger = None
        self._wandb_logger = None
        self._timers = None
        self._train_state = None
        self.start_time = time.time()

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, value: ConfigContainer):
        self._cfg = value

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = build_tokenizer(self.cfg.tokenizer_config)
        return self._tokenizer

    @property
    def tensorboard_logger(self):
        if self._tensorboard_logger is None:
            if self.cfg.logger_config.tensorboard_dir and get_rank_safe() == (get_world_size_safe() - 1):
                from torch.utils.tensorboard.writer import SummaryWriter

                print("> setting tensorboard ...")
                self._tensorboard_logger = SummaryWriter(
                    log_dir=self.cfg.logger_config.tensorboard_dir,
                    max_queue=self.cfg.logger_config.tensorboard_queue_size,
                )
            else:
                self._tensorboard_logger = None
        return self._tensorboard_logger

    @property
    def wandb_logger(self):
        if self._wandb_logger is None:
            if self.cfg.logger_config.wandb_project and get_rank_safe() == (get_world_size_safe() - 1):
                if self.cfg.logger_config.wandb_exp_name == "":
                    raise ValueError("Please specify the wandb experiment name!")

                import wandb

                save_dir = self.cfg.logger_config.wandb_save_dir or os.path.join(self.cfg.save, "wandb")
                wandb_kwargs = {
                    "dir": save_dir,
                    "name": self.cfg.logger_config.wandb_exp_name,
                    "project": self.cfg.logger_config.wandb_project,
                    "config": yaml.safe_load(dump_dataclass_to_yaml(self.cfg)),
                }
                os.makedirs(wandb_kwargs["dir"], exist_ok=True)
                wandb.init(**wandb_kwargs)

                self._wandb_logger = wandb
            else:
                self._wandb_logger = None
        return self._wandb_logger

    @property
    def timers(self):
        if self._timers is None:
            self._timers = Timers(self.cfg.logger_config.timing_log_level, self.cfg.logger_config.timing_log_option)
            self._timers.write_to_wandb = _timers_write_to_wandb
        return self._timers

    @property
    def train_state(self):
        if self._train_state is None:
            self._train_state = TrainState()
        return self._train_state

    @train_state.setter
    def train_state(self, value: TrainState):
        self._train_state = value
