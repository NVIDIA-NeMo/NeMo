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

from abc import ABC
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.utils import logging, model_utils

# Global registry of all adapters
ADAPTER_REGISTRY = {}


@dataclass
class AdapterRegistryInfo:
    base_class: type
    adapter_class: type

    # generated automatically
    base_class_path: str = ""
    adapter_class_path: str = ""

    def __post_init__(self):
        self.base_class_path = f'{self.base_class.__module__}.{self.base_class.__name__}'
        self.adapter_class_path = f'{self.adapter_class.__module__}.{self.adapter_class.__name__}'


def register_adapter(base_class: type, adapter_class: type):
    """
    Registers a pair (Base class, Adapter class) into the adapter registry, used for de-referencing.

    Args:
        base_class: A Class, which is the base class of the object.
        adapter_class: A Class, which is the subclass of the base class, and implements the Adapter mixin methods.
    """
    global ADAPTER_REGISTRY
    base_class_path = f'{base_class.__module__}.{base_class.__name__}'
    adapter_class_path = f'{adapter_class.__module__}.{adapter_class.__name__}'

    # test if base class already in registry
    if base_class_path in ADAPTER_REGISTRY:
        raise ValueError(f"`{base_class_path}` has already been added to the adapter registry !")

    # test if adapter is a subclass of the base class
    if not issubclass(adapter_class, base_class):
        raise ValueError(f"`{adapter_class_path}` is not a sub-class of {base_class_path} !")

    # register the base class : adapter class pair
    ADAPTER_REGISTRY[base_class_path] = AdapterRegistryInfo(base_class=base_class, adapter_class=adapter_class)

    # attach adapter class to base class
    base_class._meta_adapter_class = adapter_class

    # attach base class to adapter class
    adapter_class._meta_base_class = base_class


def get_registered_adapter(cls: Union[str, type]) -> Optional[AdapterRegistryInfo]:
    """
    Resolves a provided `cls` (whether str path to class, a registered base or an adapter class)
    to obtain the metadata for the adapter.

    Args:
        cls: Can be a str (absolute path to a class), a base class or an adapter class (which have already
            been registered).

    Returns:
        A AdapterRegistryInfo object if it could resolve successfully, otherwise None.
    """
    global ADAPTER_REGISTRY
    if isinstance(cls, str):
        cls = model_utils.import_class_by_path(cls)

    # If an adapter class was provided, de-reference its base class
    if hasattr(cls, '_meta_base_class'):
        cls = cls._meta_base_class

    class_path = f'{cls.__module__}.{cls.__name__}'

    # If base class, check registry
    if class_path in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[class_path]

    return None


class AdapterModuleMixin(ABC):
    """ Generic Adapter Mixin that can augment any torch.nn.Module with Adapter module support.

    This mixin class adds a hierarchical way to add any type of Adapter modules to a pre-existing module.
    Since Models are inherently also nn.Module, this mixin can be attached to any Model or Module.
    This mixin class adds several utility methods which are utilized or overridden as necessary.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

        -   It's input and output dimension are the same, while the hidden dimension need not be the same.
        -   The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
                yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
        -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.

    **Note**: This module is **not** responsible for maintaining its config. Subclasses must ensure config is updated
        or preserved as needed. It is the responsibility of the subclasses to propagate the most up to date config to
        lower layers.
    """

    adapter_global_cfg_key = "global_cfg"

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this module.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig or Dataclass that contains at the bare minimum `__target__` to instantiate a
                new Adapter module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Add adapter_layer ModuleDict() if not present.
        if not hasattr(self, 'adapter_layer'):
            self.adapter_layer = nn.ModuleDict()

        # Add adapter_cfg if it doesnt exist or hasnt been assigned yet.
        if not hasattr(self, 'adapter_cfg'):
            self.adapter_cfg = OmegaConf.create({})

        # Assert that name is globally unique to all adapters.
        if name in self.adapter_layer:
            raise ValueError(f"Adapter with name `{name}` already exists !")

        # Assert that name is not `adapter_global_cfg_key`
        if name == self.adapter_global_cfg_key:
            raise ValueError(f"Adapters cannot have the reserved name : `{self.adapter_global_cfg_key}`")

        # Update internal config and instantiate the Adapter module
        with open_dict(cfg), open_dict(self.adapter_cfg):
            adapter_enabled = cfg.pop('enabled', True)
            self.adapter_layer[name] = instantiate(cfg)

            cfg['enabled'] = adapter_enabled
            self.adapter_cfg[name] = cfg

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None and len(self.adapter_layer) > 0
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        .. code::

            module.set_enabled_adapters(enabled=False)
            module.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to enable/disable")

        # If name is None, enable/disable all adapters.
        if name is None:
            for key, config in self.adapter_cfg.items():
                # Skip the global adapter config
                if key == self.adapter_global_cfg_key:
                    continue

                # Enable/Disable the current adapter
                self.adapter_cfg[key]['enabled'] = enabled
        else:
            # Cannot set the state of the global config for adapters
            if name == self.adapter_global_cfg_key:
                raise ValueError(
                    f'Cannot set the state of the global config of adapters, given name = `{self.adapter_global_cfg_key}`'
                )

            # Enable/Disable just named adapter
            self.adapter_cfg[name]['enabled'] = enabled

    def get_enabled_adapters(self) -> List[str]:
        """
        Returns a list of all enabled adapters.

        Returns:
            A list of str names of each enabled adapter(s).
        """
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to get enabled/disabled state")

        enabled_adapters = []
        for name, config in self.adapter_cfg.items():
            # Skip the global adapter config
            if name == self.adapter_global_cfg_key:
                continue

            if self.adapter_cfg[name]['enabled']:
                enabled_adapters.append(name)

        return enabled_adapters

    def unfreeze_enabled_adapters(self, freeze_batchnorm: bool = True) -> None:
        """
        Utility method to unfreeze only the enabled Adapter module(s).

        A common user pattern is to freeze all the modules (including all the adapters), and then
        unfreeze just the required adapters.

        .. code::

            module.freeze()  # only available to nemo.core.NeuralModule !
            module.unfreeze_enabled_adapters()

        Args:
            freeze_batchnorm: An optional (and recommended) practice of freezing the updates to the moving average
                buffers of any and all BatchNorm*D layers. This is necessary to ensure that disabling all adapters
                will precisely yield the original (base) model's outputs.
        """
        if freeze_batchnorm:
            for mname, module in self.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
                    module.track_running_stats = False  # prevent running stats from updated during finetuning

                    logging.info(f"Froze module {mname}: {module}")

        adapter_names = set([])
        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'adapter_layer') and module.is_adapter_available():
                for name, config in self.adapter_cfg.items():
                    # Skip global adapter config
                    if name == self.adapter_global_cfg_key:
                        continue

                    # Check if adapter is enabled or not
                    if self.adapter_cfg[name]['enabled']:
                        # Recursively set training mode of submodules
                        module.adapter_layer[name].train()

                        # Recursively set grad required for submodules
                        for pname, param in module.adapter_layer[name].named_parameters():
                            param.requires_grad_(True)

                        # unfreeze batch norm if any in the adapter submodules
                        for mname, module in module.adapter_layer[name].named_modules():
                            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                                module.track_running_stats = (
                                    True  # prevent running stats from updated during finetuning
                                )
                                logging.info(f"Unfroze adapter module {mname}: {module}")

                        adapter_names.add(name)

        for name in adapter_names:
            logging.info(f"Unfrozen adapter : {name}")


class AdapterModelPTMixin(AdapterModuleMixin):
    """ Adapter Mixin that can augment a ModelPT subclass with Adapter support.

        This mixin class should be used only with a top level ModelPT subclass.
        This mixin class adds several utility methods which should be subclassed and overriden to
        propagated to the submodules as necessary.

        An Adapter module is any Pytorch nn.Module that possess a few properties :

        - It's input and output dimension are the same, while the hidden dimension need not be the same.
        - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
            yields the original output.

        This mixin adds the following instance variables to the class this inherits it:

            -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                    and values are the Adapter nn.Module().
            -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
            -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.

        **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
            write Adapter config information to `self.cfg.adapters`.
        """

    def setup_adapters(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        adapters that were previously added.

        Should be overriden by the subclass for additional setup steps as required.

        This method should be called just once at constructor time.
        """
        # Test if `adapters` is part of the config (injected from previous Adapter additions)
        if 'adapters' in self.cfg:
            # Set the global config of adapters
            self.update_adapter_cfg(self.cfg.adapters)

            # Dispatch the call to the encoder, for every adapter contained in the config.
            for adapter_name, adapter_cfg in self.cfg.adapters.items():
                # reserve special key `model.adapters.cfg`
                if adapter_name == self.adapter_global_cfg_key:
                    continue

                self.add_adapter(name=adapter_name, cfg=adapter_cfg)
                logging.info(
                    f"Finished setup of adapter : '{adapter_name}'. Enabled: {adapter_cfg.get('enabled', True)}."
                )

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this model.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Adapter module.
        """
        self._check_valid_model_with_adapter_support()

        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Update the model.cfg with information about the new adapter from cfg
        with open_dict(cfg), open_dict(self.cfg):
            if 'adapters' not in self.cfg:
                self.cfg.adapters = OmegaConf.create({})

            if 'enabled' not in cfg:
                cfg['enabled'] = True

            self.cfg.adapters[name] = OmegaConf.create(cfg)

            # Set the global config of adapters
            self.update_adapter_cfg(self.cfg.adapters)

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Should be overridden by the subclass.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        self._check_valid_model_with_adapter_support()

        if 'adapters' in self.cfg:
            self.update_adapter_cfg(self.cfg.adapters)

        return 'adapters' in self.cfg

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        .. code::

            model.set_enabled_adapters(enabled=False)
            model.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
        self._check_valid_model_with_adapter_support()

        # Update the adapter config with information about whether it is enabled/disabled.
        with open_dict(self.cfg.adapters):
            # If no name is provided, update all adapters.
            if name is None:
                for key in self.cfg.adapters.keys():
                    # Skip the global adapter config
                    if key == self.adapter_global_cfg_key:
                        continue

                    self.cfg.adapters[key]['enabled'] = enabled
                    logging.info(f"Setting adapter '{key}' status : Enabled = {enabled}")

            else:
                # Cannot set the state of the global config for adapters
                if name == self.adapter_global_cfg_key:
                    raise ValueError(
                        f'Cannot set the state of the global config of adapters, given name = `{self.adapter_global_cfg_key}`'
                    )

                # Otherwise, update just the specified adapter.
                self.cfg.adapters[name]['enabled'] = enabled
                logging.info(f"Setting adapter '{name}' status : Enabled = {enabled}")

            self.update_adapter_cfg(self.cfg.adapters)

    def get_enabled_adapters(self) -> List[str]:
        """
        Returns a list of all enabled adapters.

        Should be implemented by the subclass.

        Returns:
            A list of str names of each enabled adapter(s).
        """
        self._check_valid_model_with_adapter_support()

        if 'adapters' in self.cfg:
            self.update_adapter_cfg(self.cfg.adapters)
        return []

    def _check_valid_model_with_adapter_support(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.

        Should be implemented by the subclass.
        """
        pass

    def update_adapter_cfg(self, cfg: DictConfig):
        """
        Utility method to recursively update all of the Adapter module configs with the provided config.
        **Note**: It is not a (deep)copy, but a reference copy. Changes made to the config will be reflected to
            adapter submodules, but it is still encouraged to explicitly update the adapter_cfg using this method.

        Args:
            cfg: DictConfig containing the value of `model.cfg.adapters`.
        """
        for module in self.modules():  # access PT subclass method via inheritance
            if isinstance(module, AdapterModuleMixin):
                module.adapter_cfg = cfg
