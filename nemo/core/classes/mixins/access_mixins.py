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

from abc import ABC
from typing import Optional

import torch
from omegaconf import DictConfig

_ACCESS_CFG = DictConfig({"detach": False, "convert_to_cpu": False})
_ACCESS_ENABLED = False


def set_access_cfg(cfg: 'DictConfig'):
    if cfg is None or not isinstance(cfg, DictConfig):
        raise TypeError(f"cfg must be a DictConfig")
    global _ACCESS_CFG
    _ACCESS_CFG = cfg


class AccessMixin(ABC):
    """
    Allows access to output of intermediate layers of a model
    """

    def __init__(self):
        super().__init__()
        self._registry = {}  # dictionary of lists

    def register_accessible_tensor(self, name, tensor):
        """
        Register tensor for later use.
        """
        if self.access_cfg.get('convert_to_cpu', False):
            tensor = tensor.cpu()

        if self.access_cfg.get('detach', False):
            tensor = tensor.detach()

        if not hasattr(self, '_registry'):
            self._registry = {}

        if name not in self._registry:
            self._registry[name] = []

        self._registry[name].append(tensor)

    @classmethod
    def get_module_registry(cls, module: torch.nn.Module):
        """
        Extract all registries from named submodules, return dictionary where
        the keys are the flattened module names, the values are the internal registry
        of each such module.
        """
        module_registry = {}
        for name, m in module.named_modules():
            if hasattr(m, '_registry') and len(m._registry) > 0:
                module_registry[name] = m._registry
        return module_registry

    def reset_registry(self: torch.nn.Module, registry_key: Optional[str] = None):
        """
        Reset the registries of all named sub-modules
        """
        if hasattr(self, "_registry"):
            if registry_key is None:
                self._registry.clear()
            else:
                if registry_key in self._registry:
                    self._registry.pop(registry_key)
                else:
                    raise KeyError(
                        f"Registry key `{registry_key}` provided, but registry does not have this key.\n"
                        f"Available keys in registry : {list(self._registry.keys())}"
                    )

        for _, m in self.named_modules():
            if hasattr(m, "_registry"):
                if registry_key is None:
                    m._registry.clear()
                else:
                    if registry_key in self._registry:
                        self._registry.pop(registry_key)
                    else:
                        raise KeyError(
                            f"Registry key `{registry_key}` provided, but registry does not have this key.\n"
                            f"Available keys in registry : {list(self._registry.keys())}"
                        )

        # Explicitly disable registry cache after reset
        AccessMixin.set_access_enabled(access_enabled=False)

    @property
    def access_cfg(self):
        """
        Returns:
            The global access config shared across all access mixin modules.
        """
        global _ACCESS_CFG
        return _ACCESS_CFG

    @classmethod
    def update_access_cfg(cls, cfg: dict):
        global _ACCESS_CFG
        _ACCESS_CFG.update(cfg)

    @classmethod
    def is_access_enabled(cls):
        global _ACCESS_ENABLED
        return _ACCESS_ENABLED

    @classmethod
    def set_access_enabled(cls, access_enabled: bool):
        global _ACCESS_ENABLED
        _ACCESS_ENABLED = access_enabled
