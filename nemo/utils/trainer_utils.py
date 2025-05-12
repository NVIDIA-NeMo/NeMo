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

from typing import Mapping

_HAS_HYDRA = True

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:
    DictConfig = Mapping
    OmegaConf = None
    _HAS_HYDRA = False


def resolve_trainer_cfg(trainer_cfg: DictConfig) -> DictConfig:
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not _HAS_HYDRA:
        return trainer_cfg
    if (strategy := trainer_cfg.get("strategy", None)) is not None and isinstance(strategy, Mapping):
        trainer_cfg["strategy"] = hydra.utils.instantiate(strategy)
    return trainer_cfg
