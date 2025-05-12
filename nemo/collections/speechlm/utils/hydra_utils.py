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


from typing import Any, Dict, List, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.core.classes.common import Serialization


def get_object_list_from_config(cfg: Union[DictConfig, ListConfig, Dict]) -> List[Any]:
    """
    Instantiates a list of objects from a config object.
    Args:
        cfg: Config object (DictConfig or ListConfig).
    Returns:
        List of instantiated objects.
    """
    if isinstance(cfg, (DictConfig, Dict)) and "_target_" not in cfg:
        candidates = list(cfg.values())
    elif isinstance(cfg, (ListConfig, List)):
        candidates = cfg
    elif isinstance(cfg, (DictConfig, Dict)) and "_target_" in cfg:
        candidates = [cfg]
    else:
        raise ValueError(f"Unsupported config type: {type(cfg)}, with content: {cfg}")

    return [Serialization.from_config_dict(c) for c in candidates]


def to_dict_config(cfg):
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    return cfg
