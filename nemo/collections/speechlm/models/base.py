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


import uuid
from typing import Dict, List, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities import model_summary
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.utils import logging
from nemo.utils.app_state import AppState

__all__ = ['SpeechLanguageModel']


class SpeechLanguageModel(pl.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(self):
        super().__init__()
        app_state = AppState()

        self._set_model_guid()
        self._cfg = OmegaConf.create({})
        # Set device_id in AppState
        if torch.cuda.is_available() and torch.cuda.current_device() is not None:
            app_state.device_id = torch.cuda.current_device()

        # Create list of lists for val and test outputs to support multiple dataloaders
        # Initialize an empty list as sometimes self._validation_dl can be None at this stage
        self._validation_step_outputs = None
        self._validation_names = None
        self._num_validation_dl = None

        # Initialize an empty list as sometimes self._test_dl can be None at this stage
        self._test_step_outputs = None
        self._test_names = None
        self._num_test_dl = None

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    @cfg.setter
    def cfg(self, cfg: DictConfig):
        self._cfg = cfg

    def summarize(self, max_depth: int = 1) -> model_summary.ModelSummary:
        """Summarize this LightningModule.
        Args:
            max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
                layer summary off. Default: 1.
        Return:
            The model summary object
        """
        return model_summary.summarize(self, max_depth=max_depth)

    def freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = True

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit, validate, test, or predict.
        This is called on every process when using DDP.

        Args:
            stage: fit, validate, test or predict
        """

        self.propagate_model_guid()

        self.setup_multi_validation_data()
        self.setup_multi_test_data()

        with open_dict(self._cfg):
            if self.trainer is not None:
                self._cfg.data = self.trainer.datamodule.cfg

        logging.info(self.summarize())

    def _set_model_guid(self):
        if not hasattr(self, 'model_guid'):
            appstate = AppState()

            # Generate a unique uuid for the instance
            # also determine if the model is being restored or not, and preserve the path
            self.model_guid = str(uuid.uuid4())
            appstate.register_model_guid(self.model_guid)

    def propagate_model_guid(self):
        """
        Propagates the model GUID to all submodules, recursively.
        """

        def recursively_propagate_guid(module: pl.LightningModule):
            module.model_guid = self.model_guid
            for child in module.children():
                recursively_propagate_guid(child)

        for _, module in self.named_modules():
            module.model_guid = self.model_guid
            recursively_propagate_guid(module)

    def setup_multi_validation_data(self):
        if self.trainer is None:
            logging.warning("Trainer is not set. Cannot setup multi validation/test data.")
            return

        data_module = self.trainer.datamodule

        if getattr(data_module, "_validation_ds", None) is None:
            logging.info("No validation dataset found. Skipping setup of multi validation data.")
            return

        self._num_validation_dl = getattr(data_module, "_num_validation_dl", None)
        self._validation_names = getattr(data_module, "_validation_names", None)
        if self._num_validation_dl is None:
            # special case for lhotse dataloader
            self._num_validation_dl = len(self._validation_names)

        if self._validation_names is None:
            if not self._num_validation_dl:
                raise ValueError(
                    f"`_num_validation_dl` not found/valid in data module: { getattr(data_module, '_num_validation_dl', None) }"
                )
            self._validation_names = [f'val_{idx}' for idx in range(self._num_validation_dl)]
            logging.info(
                f"`_validation_names` not found in data module. Setting default names: {self._validation_names}"
            )
        elif len(self._validation_names) != self._num_validation_dl:
            raise ValueError(
                f"Number of validation names provided ({len(self._validation_names)}) does not match number of validation datasets ({self._num_validation_dl})."
            )

    def setup_multi_test_data(self):
        if self.trainer is None:
            logging.warning("Trainer is not set. Cannot setup multi validation/test data.")
            return

        data_module = self.trainer.datamodule

        if getattr(data_module, "_test_ds", None) is None:
            logging.info("No test dataset found. Skipping setup of multi test data.")
            return

        self._num_test_dl = getattr(data_module, "_num_test_dl", None)
        self._test_names = getattr(data_module, "_test_names", None)
        if self._num_test_dl is None:
            # special case for lhotse dataloader
            self._num_test_dl = len(self._test_names)

        if self._test_names is None:
            if not self._num_test_dl:
                raise ValueError(
                    f"`_num_test_dl` not found/valid in data module: {getattr(data_module, '_num_test_dl', None)}"
                )
            self._test_names = [f'test_{idx}' for idx in range(self._num_test_dl)]
            logging.info(f"`_test_names` not found in data module. Setting default names: {self._test_names}")
        elif len(self._test_names) != self._num_test_dl:
            raise ValueError(
                f"Number of test names provided ({len(self._test_names)}) does not match number of test datasets ({self._num_test_dl})."
            )

    @property
    def validation_step_outputs(self):
        """
        Cached outputs of validation_step. It can be a list of items (for single data loader) or a list of lists
        (for multiple data loaders).

        Returns:
            List of outputs of validation_step.
        """
        if self._validation_step_outputs is not None:
            return self._validation_step_outputs

        # Initialize new output list
        self._validation_step_outputs = []
        # Check len(data_module._validation_ds) > 1 as sometimes single dataloader can be in a list: [<Dataloader obj>] when ds_item in
        # config has 1 item passed in a list
        if self._num_validation_dl > 1:
            for _ in range(self._num_validation_dl):
                self._validation_step_outputs.append([])

        return self._validation_step_outputs

    @validation_step_outputs.setter
    def validation_step_outputs(self, value):
        self._validation_step_outputs = value

    @property
    def test_step_outputs(self):
        """
        Cached outputs of test_step. It can be a list of items (for single data loader) or a list of lists (for multiple data loaders).

        Returns:
            List of outputs of test_step.
        """
        if self._test_step_outputs is not None:
            return self._test_step_outputs

        # Initialize new output list
        self._test_step_outputs = []
        # Check len(data_module._test_ds) > 1 as sometimes single dataloader can be in a list: [<Dataloader obj>] when ds_item in
        # config has 1 item passed in a list
        if self._num_test_dl > 1:
            for _ in range(self._num_test_dl):
                self._test_step_outputs.append([])

        return self._test_step_outputs

    @test_step_outputs.setter
    def test_step_outputs(self, value):
        self._test_step_outputs = value

    def on_validation_epoch_end(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Validation set which automatically supports multiple data loaders
        via `multi_validation_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_validation_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `val_loss`,
            only the `val_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `val_dl_idx: int`
            inside the `validation_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        # Case where we dont provide data loaders
        if self.validation_step_outputs is not None and len(self.validation_step_outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if isinstance(self.validation_step_outputs[0], dict):
            output_dict = self.multi_validation_epoch_end(self.validation_step_outputs, dataloader_idx=0)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            self.validation_step_outputs.clear()  # free memory
            return output_dict

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, val_outputs in enumerate(self.validation_step_outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_validation_epoch_end(val_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `val_loss` resolution first (if provided outside logs)
                if 'val_loss' in dataloader_logs:
                    if 'val_loss' not in output_dict and dataloader_idx == self._val_dl_idx:
                        output_dict['val_loss'] = dataloader_logs['val_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}

                        for k_log, v_log in v.items():
                            # If we are logging the metric, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if k_log not in output_dict['log'] and dataloader_idx == self._val_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[f'{dataloader_prefix}_{k_log}'] = v_log

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = f'{dataloader_prefix}_{k_log}'

                            # Store log value
                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict['log']
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = f'{dataloader_prefix}_{k}'
                        output_dict[new_k] = v

                self.validation_step_outputs[dataloader_idx].clear()  # free memory

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)
            # return everything else
            return output_dict

    def on_test_epoch_end(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Default DataLoader for Test set which automatically supports multiple data loaders
        via `multi_test_epoch_end`.

        If multi dataset support is not required, override this method entirely in base class.
        In such a case, there is no need to implement `multi_test_epoch_end` either.

        .. note::
            If more than one data loader exists, and they all provide `test_loss`,
            only the `test_loss` of the first data loader will be used by default.
            This default can be changed by passing the special key `test_dl_idx: int`
            inside the `test_ds` config.

        Args:
            outputs: Single or nested list of tensor outputs from one or more data loaders.

        Returns:
            A dictionary containing the union of all items from individual data_loaders,
            along with merged logs from all data loaders.
        """
        # Case where we dont provide data loaders
        if self.test_step_outputs is not None and len(self.test_step_outputs) == 0:
            return {}

        # Case where we provide exactly 1 data loader
        if isinstance(self.test_step_outputs[0], dict):
            output_dict = self.multi_test_epoch_end(self.test_step_outputs, dataloader_idx=0)

            if output_dict is not None and 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            self.test_step_outputs.clear()  # free memory
            return output_dict

        else:  # Case where we provide more than 1 data loader
            output_dict = {'log': {}}

            # The output is a list of list of dicts, outer list corresponds to dataloader idx
            for dataloader_idx, test_outputs in enumerate(self.test_step_outputs):
                # Get prefix and dispatch call to multi epoch end
                dataloader_prefix = self.get_test_dataloader_prefix(dataloader_idx)
                dataloader_logs = self.multi_test_epoch_end(test_outputs, dataloader_idx=dataloader_idx)

                # If result was not provided, generate empty dict
                dataloader_logs = dataloader_logs or {}

                # Perform `test_loss` resolution first (if provided outside logs)
                if 'test_loss' in dataloader_logs:
                    if 'test_loss' not in output_dict and dataloader_idx == self._test_dl_idx:
                        output_dict['test_loss'] = dataloader_logs['test_loss']

                # For every item in the result dictionary
                for k, v in dataloader_logs.items():
                    # If the key is `log`
                    if k == 'log':
                        # Parse every element of the log, and attach the prefix name of the data loader
                        log_dict = {}
                        for k_log, v_log in v.items():
                            # If we are logging the loss, but dont provide it at result level,
                            # store it twice - once in log and once in result level.
                            # Also mark log with prefix name to avoid log level clash with other data loaders
                            if k_log not in output_dict['log'] and dataloader_idx == self._test_dl_idx:
                                new_k_log = k_log

                                # Also insert duplicate key with prefix for ease of comparison / avoid name clash
                                log_dict[f'{dataloader_prefix}_{k_log}'] = v_log

                            else:
                                # Simply prepend prefix to key and save
                                new_k_log = f'{dataloader_prefix}_{k_log}'

                            log_dict[new_k_log] = v_log

                        # Update log storage of individual data loader
                        output_logs = output_dict.get('log', {})
                        output_logs.update(log_dict)

                        # Update global log storage
                        output_dict['log'] = output_logs

                    else:
                        # If any values are stored outside 'log', simply prefix name and store
                        new_k = f'{dataloader_prefix}_{k}'
                        output_dict[new_k] = v
                self.test_step_outputs[dataloader_idx].clear()  # free memory

            if 'log' in output_dict:
                self.log_dict(output_dict.pop('log'), on_epoch=True)

            # return everything else
            return output_dict

    def multi_validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Adds support for multiple validation datasets. Should be overriden by subclass,
        so as to obtain appropriate logs for each of the dataloaders.

        Args:
            outputs: Same as that provided by LightningModule.on_validation_epoch_end()
                for a single dataloader.
            dataloader_idx: int representing the index of the dataloader.

        Returns:
            A dictionary of values, optionally containing a sub-dict `log`,
            such that the values in the log will be pre-pended by the dataloader prefix.
        """
        logging.warning(
            "Multi data loader support has been enabled, but "
            "`multi_validation_epoch_end(outputs, dataloader_idx) has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override "
            "`on_validation_epoch_end(outputs)."
        )

    def multi_test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Adds support for multiple test datasets. Should be overriden by subclass,
        so as to obtain appropriate logs for each of the dataloaders.

        Args:
            outputs: Same as that provided by LightningModule.on_validation_epoch_end()
                for a single dataloader.
            dataloader_idx: int representing the index of the dataloader.

        Returns:
            A dictionary of values, optionally containing a sub-dict `log`,
            such that the values in the log will be pre-pended by the dataloader prefix.
        """
        logging.warning(
            "Multi data loader support has been enabled, but "
            "`multi_test_epoch_end(outputs, dataloader_idx) has not been implemented.\n"
            "If you require multi data loader support for validation sets, please override this method.\n"
            "If you do not require multi data loader support, please instead override "
            "`on_test_epoch_end(outputs)."
        )

    def get_validation_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """
        Get the name of one or more data loaders, which will be prepended to all logs.

        Args:
            dataloader_idx: Index of the data loader.

        Returns:
            str name of the data loader at index provided.
        """
        return self._validation_names[dataloader_idx]

    def get_test_dataloader_prefix(self, dataloader_idx: int = 0) -> str:
        """
        Get the name of one or more data loaders, which will be prepended to all logs.

        Args:
            dataloader_idx: Index of the data loader.

        Returns:
            str name of the data loader at index provided.
        """
        return self._test_names[dataloader_idx]

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReductionWithLossMask()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReductionWithLossMask(validation_step=True)

        return self._validation_loss_reduction
