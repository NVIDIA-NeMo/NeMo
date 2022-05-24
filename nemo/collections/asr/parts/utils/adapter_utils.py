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
from omegaconf import DictConfig, OmegaConf

from nemo.utils import logging


def update_adapter_cfg_input_dim(module: torch.nn.Module, cfg: DictConfig, *, module_dim: int):
    if 'in_features' in cfg:
        in_planes = cfg['in_features']

        if in_planes != module_dim:
            logging.info(f"Updating {module.__class__.__name__} Adapter input dim from {in_planes} to {module_dim}")
            in_planes = module_dim

        cfg['in_features'] = in_planes
        return cfg
    else:
        raise ValueError(
            f"Failed to infer the input dimension of the Adapter cfg. Provided config : \n" f"{OmegaConf.to_yaml(cfg)}"
        )
