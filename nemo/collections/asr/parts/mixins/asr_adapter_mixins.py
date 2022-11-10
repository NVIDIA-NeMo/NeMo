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

from typing import List, Optional, Tuple

from omegaconf import DictConfig, open_dict

from nemo.core.classes.mixins.adapter_mixins import AdapterModelPTMixin, AdapterModuleMixin
from nemo.utils import logging, logging_mode


class ASRAdapterModelMixin(AdapterModelPTMixin):
    """ ASR Adapter Mixin that can augment any Encoder module with Adapter module support.

    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.

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

        This method should be called just once at constructor time.
        """
        supports_adapters = False

        # At least the encoder must extend AdapterModuleMixin
        if hasattr(self, 'encoder') and isinstance(self.encoder, AdapterModuleMixin):
            supports_adapters |= True

        if hasattr(self, 'decoder') and isinstance(self.decoder, AdapterModuleMixin):
            supports_adapters |= True

        if hasattr(self, 'joint') and isinstance(self.joint, AdapterModuleMixin):
            supports_adapters |= True

        # If adapters are supported, setup the adapter config + any modules (pre-existing adapter modules)
        if supports_adapters:
            super().setup_adapters()

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this model.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Adapter module.
        """
        # setup the config for adapters
        super().add_adapter(name=name, cfg=cfg)

        # Resolve module name and adapter name
        module_name, _ = self.resolve_adapter_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        # Update the model.cfg with information about the new adapter from cfg
        with open_dict(self.cfg):
            for module_name in module_names:
                # Check if encoder adapters should be added
                if module_name in ('', 'encoder'):
                    # Dispatch the call to the encoder.
                    self.encoder.add_adapter(name=name, cfg=cfg)

                # Check if decoder adapters should be added
                if module_name == 'decoder':
                    # Dispatch call to the decoder.
                    self.decoder.add_adapter(name=name, cfg=cfg)

                # Check if joint adapters should be added;
                # Note: We need additional check if joint even exists in model (for CTC models)
                if hasattr(self, 'joint') and module_name == 'joint':
                    # Dispatch call to the joint.
                    self.joint.add_adapter(name=name, cfg=cfg)

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        config_contains_adapter = super().is_adapter_available()

        # Forward the method call to the individual modules
        if hasattr(self, 'encoder') and isinstance(self.encoder, AdapterModuleMixin):
            config_contains_adapter |= self.encoder.is_adapter_available()

        if hasattr(self, 'decoder') and isinstance(self.decoder, AdapterModuleMixin):
            config_contains_adapter |= self.decoder.is_adapter_available()

        if hasattr(self, 'joint') and isinstance(self.joint, AdapterModuleMixin):
            config_contains_adapter |= self.joint.is_adapter_available()

        return config_contains_adapter

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        .. code::

            model.set_enabled_adapters(enabled=False)
            model.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
        super().set_enabled_adapters(name=name, enabled=enabled)

        # Resolve the module name and adapter name
        if name is not None:
            module_name, _ = self.resolve_adapter_module_name_(name)
        else:
            module_name = None

        # Use + as a splitter, in order to share one name across multiple modules
        if module_name is not None and '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        for module_name in module_names:
            # Check if encoder adapters should be used
            # Dispatch the call to the encoder.
            if name is None or module_name in ('', 'encoder'):
                if self.encoder.is_adapter_available():
                    self.encoder.set_enabled_adapters(name=name, enabled=enabled)

            # Dispatch the call to the decoder.
            if name is None or module_name == 'decoder':
                if self.decoder.is_adapter_available():
                    self.decoder.set_enabled_adapters(name=name, enabled=enabled)

            # Dispatch the call to the joint.
            # Note: We need additional check for joint, since it may not exist (CTC models).
            if name is None or module_name == 'joint':
                if hasattr(self, 'joint') and self.joint.is_adapter_available():
                    self.joint.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        """
        Returns a list of all enabled adapters.

        Returns:
            A list of str names of each enabled adapter(s).
        """
        enabled_adapters = super().get_enabled_adapters()

        # Check if encoder adapters should be used or are enabled
        if hasattr(self, 'encoder') and isinstance(self.encoder, AdapterModuleMixin):
            enabled_adapters.extend(self.encoder.get_enabled_adapters())

        if hasattr(self, 'decoder') and isinstance(self.decoder, AdapterModuleMixin):
            enabled_adapters.extend(self.decoder.get_enabled_adapters())

        if hasattr(self, 'joint') and isinstance(self.joint, AdapterModuleMixin):
            enabled_adapters.extend(self.joint.get_enabled_adapters())

        enabled_adapters = list(sorted(list(set(enabled_adapters))))

        return enabled_adapters

    def check_valid_model_with_adapter_support_(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.
        """
        # Obtain the global adapter config if possible, otherwise use sensible defaults.
        global_cfg = self._get_global_cfg()

        # Test whether the encoder supports adapters
        use_encoder_adapter = global_cfg.get('check_encoder_adapter', True)
        if use_encoder_adapter:
            if not hasattr(self, 'encoder'):
                logging.warning(
                    "Cannot add adapter to this object as it does not have an `encoder` sub-module!",
                    mode=logging_mode.ONCE,
                )

            if hasattr(self, 'encoder') and not isinstance(self.encoder, AdapterModuleMixin):
                logging.warning(
                    f'{self.encoder.__class__.__name__} does not implement `AdapterModuleMixin`',
                    mode=logging_mode.ONCE,
                )

        # Test whether the decoder supports adapters
        use_decoder_adapter = global_cfg.get('check_decoder_adapter', True)
        if use_decoder_adapter:
            if not hasattr(self, 'decoder'):
                logging.warning(
                    "Cannot add adapter to this object as it does not have an `decoder` sub-module!",
                    mode=logging_mode.ONCE,
                )

            if hasattr(self, 'decoder') and not isinstance(self.decoder, AdapterModuleMixin):
                logging.warning(
                    f'{self.decoder.__class__.__name__} does not implement `AdapterModuleMixin`',
                    mode=logging_mode.ONCE,
                )

        # Test whether the joint supports adapters
        use_joint_adapter = global_cfg.get('check_joint_adapter', True)
        if use_joint_adapter:
            # Joint is only for RNNT models, skip assertion that it must always exist.
            if hasattr(self, 'joint') and not isinstance(self.joint, AdapterModuleMixin):
                logging.warning(
                    f'{self.joint.__class__.__name__} does not implement `AdapterModuleMixin`', mode=logging_mode.ONCE
                )

    def resolve_adapter_module_name_(self, name: str) -> Tuple[str, str]:
        """
        Utility method to resolve a given global/module adapter name to its components.
        Always returns a tuple representing (module_name, adapter_name). ":" is used as the
        delimiter for denoting the module name vs the adapter name.

        Will attempt to also resolve a given adapter_name alone back to (module_name, adapter_name)
        if the metadata config exists for access.

        Args:
            name: A global adapter, or a module adapter name (with structure module_name:adapter_name).

        Returns:
            A tuple representing (module_name, adapter_name). If a global adapter is provided,
            module_name is set to ''.
        """
        module_name, adapter_name = super().resolve_adapter_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        # resolve name and module only for valid modules
        valid_module_names = self.adapter_module_names

        for mod_name in module_names:
            if mod_name not in valid_module_names:
                raise ValueError(f"Provided module name `{mod_name}` is not in valid list : {valid_module_names}")

        return (module_name, adapter_name)

    def _get_global_cfg(self):
        """
        Utility method, to either extract or construct the global config inside adapters config.
        """
        global_config = DictConfig({})
        if 'adapters' in self.cfg and self.adapter_global_cfg_key in self.cfg.adapters:
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]
        return global_config

    @property
    def adapter_module_names(self) -> List[str]:
        valid_module_names = ['', 'encoder', 'decoder', 'joint']
        return valid_module_names
