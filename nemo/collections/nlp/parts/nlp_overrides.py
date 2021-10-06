# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.nlp.modules.common.megatron.megatron_bert import (
    get_megatron_checkpoint_version,
    set_megatron_checkpoint_version,
)
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.trainer.trainer import Trainer
import shutil
import tarfile
from pytorch_lightning.utilities.enums import GradClipAlgorithmType
from torch.nn.modules.module import Module

from torch.optim.optimizer import Optimizer
from nemo.utils.get_rank import is_global_rank_zero
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
import os
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from apex.transformer import tensor_parallel
from apex.transformer import parallel_state
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from torch.nn.parallel import DistributedDataParallel

from nemo.utils import AppState, logging


class NLPDDPPlugin(DDPPlugin):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.
    """

    distributed_backend = "ddp"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices, num_nodes, cluster_environment, sync_batchnorm, **kwargs)

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        # call PTL init ddp
        super().setup_distributed()

        # init model parallel if needed
        app_state = AppState()

        if app_state.model_parallel_size is not None:
            self.init_model_parallel(app_state.global_rank, app_state.world_size)
            # if self.lightning_module.has_megatron_encoder and not self.lightning_module.is_model_parallel_initialized:
            #     self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def start_training(self, trainer: 'Trainer') -> None:
        """ PTL Hook that is called after DPP is initialized. """

        if self.lightning_module.has_megatron_encoder:
            app_state = AppState()
            if app_state.model_parallel_size is not None:
                # mpu grad clipping needs parameters to have the attribute model_parallel
                parameters = self.lightning_module.parameters()
                for p in parameters:
                    if not hasattr(p, 'model_parallel'):
                        p.model_parallel = False

                if get_megatron_checkpoint_version() is not None:
                    # megatron checkpoint already restored
                    pass
                elif trainer.checkpoint_connector.resume_checkpoint_path is not None:
                    # PTL auto-resuming, need to update checkpoint name
                    # update path based on model parallel rank
                    filepath = trainer.checkpoint_connector.resume_checkpoint_path
                    dirname = os.path.dirname(os.path.dirname(filepath))
                    basename = os.path.basename(filepath)
                    filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'
                    trainer.checkpoint_connector.resume_checkpoint_path = filepath
                    logging.info(
                        f'Resuming training from checkpoint {trainer.checkpoint_connector.resume_checkpoint_path}'
                    )
                    # need to set checkpoint version for megatron-lm
                    checkpoint_version = torch.load(trainer.checkpoint_connector.resume_checkpoint_path).get(
                        'checkpoint_version', None
                    )
                    if checkpoint_version is not None:
                        set_megatron_checkpoint_version(checkpoint_version)
                    else:
                        logging.warning('Megatron-lm checkpoint version not found. Setting checkpoint_version to 0.')
                        set_megatron_checkpoint_version(0)
                else:
                    self.lightning_module.restore_megatron_encoder_weights()
            else:
                if get_megatron_checkpoint_version() is not None:
                    # megatron checkpoint already restored
                    pass
                else:
                    self.lightning_module.restore_megatron_encoder_weights()

            self.lightning_module.register_megatron_checkpoint_version()

        return super().start_training(trainer)

    def start_testing(self, trainer: 'Trainer') -> None:
        """ PTL Hook that is called after DPP is initialized. """
        app_state = AppState()

        if app_state.model_parallel_size is not None:

            if self.has_megatron_encoder:
                # check megatron checkpoint version
                checkpoint_version = get_megatron_checkpoint_version()
                if checkpoint_version is None:
                    raise ValueError("Unable to find megatron checkpoint version.")

        return super().start_testing(trainer)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info(f"Configuring DDP for model parallelism.")

            # With model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=device_ids,
                output_device=device_ids[0],
                process_group=app_state.data_parallel_group,
                find_unused_parameters=False,
                **self._ddp_kwargs,
            )

        else:
            super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            if torch.distributed.is_initialized():
                parallel_state.initialize_model_parallel(app_state.model_parallel_size)
                app_state.model_parallel_group = parallel_state.get_tensor_model_parallel_group()
                app_state.data_parallel_group = parallel_state.get_data_parallel_group()
                app_state.model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
                app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
                app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
                logging.info(f'mp_rank: {app_state.model_parallel_rank}')
                logging.info(f'dp_rank: {app_state.data_parallel_rank}')

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        app_state = AppState()
        # dump states as a checkpoint dictionary object
        # TrainingTypePlugin.on_save() just seems to return the same thing.
        # checkpoint = self.on_save(checkpoint)
        if self.is_global_zero or app_state.data_parallel_rank == 0:
            try:
                # write the checkpoint dictionary on the file
                atomic_save(checkpoint, filepath)
            except AttributeError as err:
                key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
                checkpoint.pop(key, None)
                rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
                atomic_save(checkpoint, filepath)

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPPlugin, self).distributed_sampler_kwargs


class NLPCheckpointConnector(CheckpointConnector):
    """ Override PTL CheckpointConnector to support model parallel checkpoints from Megatron-LM.
    """

    def __init__(self, trainer, resume_from_checkpoint):
        super().__init__(trainer, resume_from_checkpoint)

    def save_checkpoint(self, filepath, weights_only: bool = False) -> None:
        """Slightly modified version of PyTorch Lightning's save_checkpoint.
           Accounts for model parallel training.
           Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            filepath: write-target file's path
            weights_only: saving model weights only
        """
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # filepath needs to be updated to include mp_rank
            dirname = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'
            _checkpoint = self.dump_checkpoint(weights_only)
            # each model parallel rank needs to save a copy of its model
            if app_state.data_parallel_rank == 0:
                self.trainer.accelerator.save_checkpoint(_checkpoint, filepath)
        else:
            super().save_checkpoint(filepath, weights_only)


class NLPSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self) -> None:
        super().__init__()

    def save_to(self, model, save_path: str):
        # TODO: make save_to work with always_save_nemo for model parallel
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            # each model parallel rank creates a .nemo file
            # after all .nemo files are created, each rank
            # will add their checkpoint to global rank 0

            base_dir = os.path.dirname(save_path)  # use the directory to merge mp_rank .nemo files into one

            # update save_path based on model parallel_rank
            base_path = os.path.splitext(save_path)[0]  # everything except the extension

            mp_save_path = f'{base_path}_mp_rank_{app_state.model_parallel_rank:02d}.nemo'

            # first we save .nemo file for each model parallel rank
            if app_state.data_parallel_rank == 0:
                super().save_to(model, mp_save_path)

            # we need a barrier for data_parallel_rank 0 when using model parallel so that all processes have finished writing their weights before creating .nemo file
            # torch.distributed.barrier(group=parallel_state.get_tensor_model_parallel_group())
            # torch.distributed.barrier()

            if is_global_rank_zero():
                # extract all tar files
                for mp_rank in range(app_state.model_parallel_size):
                    mp_tar_path = f'{base_path}_mp_rank_{mp_rank:02d}.nemo'
                    mp_tar = tarfile.open(mp_tar_path, 'r:gz')
                    mp_tar.extractall(path=os.path.join(base_dir, f'mp_rank_{mp_rank:02d}'))
                    mp_tar.close()
                    os.remove(mp_tar_path)

                # move rank 0 .nemo extract to base_path
                shutil.move(os.path.join(base_dir, 'mp_rank_00'), base_path)

                # move mp_rank_00 checkpoint to mp_rank_00 directory inside base_path
                os.mkdir(os.path.join(base_path, 'mp_rank_00'))
                shutil.move(os.path.join(base_path, 'model_weights.ckpt'), os.path.join(base_path, 'mp_rank_00'))

                # move other mp_rank checkpoints from base_dir to base_path
                for mp_rank in range(1, app_state.model_parallel_size):
                    os.mkdir(os.path.join(base_path, f'mp_rank_{mp_rank:02d}'))
                    shutil.move(
                        os.path.join(base_dir, f'mp_rank_{mp_rank:02d}', 'model_weights.ckpt'),
                        os.path.join(base_path, f'mp_rank_{mp_rank:02d}'),
                    )
                    # clean up leftover directory
                    shutil.rmtree(os.path.join(base_dir, f'mp_rank_{mp_rank:02d}'))

                # create tar file from base_path
                self._make_nemo_file_from_folder(save_path, base_path)

                # clean up base_path
                shutil.rmtree(base_path)

        else:
            return super().save_to(model, save_path)


class NLPNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    def __init__(self, init_scale: float = 2 ** 32, growth_interval: int = 1000) -> None:
        super().__init__(precision=16)

        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale, growth_interval=growth_interval)

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType,
        model: Optional[Module],
    ) -> None:
        """Override PTL gradient clipping.
           Do nothing because we've already clipped gradients in `on_before_optimizer_step` hook.
        """
        pass


class NLPNativeBfloat16PrecisionPlugin(NativeMixedPrecisionPlugin):
    def __init__(self) -> None:
        super().__init__(precision='bf16')

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType,
        model: Optional[Module],
    ) -> None:
        """Override PTL gradient clipping.
           Model parallel models require gradient clipping from megatron-lm.
        """

        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        app_state = AppState()
        if app_state.model_parallel_size is not None:
            parameters = model.parameters()
            clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)
        else:
            return super().clip_gradients(
                optimizer, clip_val, gradient_clip_algorithm=gradient_clip_algorithm, model=model
            )


class NLPPrecisionPlugin(PrecisionPlugin):
    def __init__(self) -> None:
        super().__init__()

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType,
        model: Optional[Module],
    ) -> None:
        """Override PTL gradient clipping.
           Model parallel models require gradient clipping from megatron-lm.
        """

        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        app_state = AppState()
        if app_state.model_parallel_size is not None:
            parameters = model.parameters()
            clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)
        else:
            return super().clip_gradients(
                optimizer, clip_val, gradient_clip_algorithm=gradient_clip_algorithm, model=model
            )
