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


import os

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

try:
    import apex

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ["MegatronBaseModel"]


class MegatronBaseModel(NLPModel):
    """
    Megatron base class
    It does the following things:
    1. Initialize the model parallel for nemo given the model parallel parameters.
    2. Turn on all the nvidia optimizations.
    3. If `cfg.tokenizer` is available, it loads the tokenizer and pad the vocab to the correct size for tensor model parallelism.
    4. It help to run `configure_gradient_clipping`, if `grad_clip_pl_default` is set True,  it uses the pytorch lightning default
       gradient clipping. Or if `megatron_amp_o2` is set True, it uses the parameters from optimizer to clip the gradients.
       Otherwise, it uses the parameters calculated in the `setup_optimizer_param_groups` method.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, no_lm_init=True):
        # FIXME: switch to self._cfg
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None

        super().__init__(cfg, trainer=trainer, no_lm_init=no_lm_init)

        # used in NVIDIA NGC PyTorch containers
        self._enable_nvidia_optimizations()

        if self._cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            pipeline_model_parallel_size=cfg.get('pipeline_model_parallel_size', 1),
            pipeline_model_parallel_split_rank=cfg.get('pipeline_model_parallel_split_rank', 0),
            micro_batch_size=cfg.get('micro_batch_size'),
            global_batch_size=cfg.get('global_batch_size'),
            seed=self.cfg.get('seed', 1234),
            apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
        )

        self.grad_clip_pl_default = False  # use pytorch default for gradient clipping. Default False

        if hasattr(self._cfg, "tokenizer"):
            # build tokenizer (defaults to nemo supported tokenizers)
            self._build_tokenizer()

            # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
            self._build_vocab()

    def _enable_nvidia_optimizations(self):
        "These optimizations are present in NVIDIA NGC PyTorch Containers"

        # Version check
        nvidia_torch_version = os.getenv('NVIDIA_PYTORCH_VERSION', None)
        if nvidia_torch_version is not None:
            NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split('.')[0])
            try:
                NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split('.')[1])
            except Exception:
                NVIDIA_TORCH_MINOR = 0

            # Apex Persistent layer norm is supported from Nvidia PyTorch container v21.11
            if NVIDIA_TORCH_MAJOR < 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR < 11):
                self.cfg.persist_layer_norm = False

            if NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11):
                # NVFUSER
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                torch._C._jit_override_can_fuse_on_cpu(False)
                torch._C._jit_override_can_fuse_on_gpu(False)
                torch._C._jit_set_texpr_fuser_enabled(False)
                torch._C._jit_set_nvfuser_enabled(True)
                torch._C._debug_set_autodiff_subgraph_inlining(False)
        else:
            # Not a Nvidia container. Dependency check is on users
            pass

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.merge_file),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            legacy=True if self._cfg.tokenizer.library == 'sentencepiece' else False,
        )

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def _get_parameters(self):
        """
        private method to load all the trainable parameters from optimizer param groups
        """
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if self.grad_clip_pl_default:
            # use the default behavior
            return super().configure_gradient_clipping(*args, **kwargs)
        elif self.megatron_amp_o2:
            # grep fp32 master parameters for gradient clipping
            parameters = self._optimizer.get_parameters()
        else:
            parameters = self._get_parameters()

        grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)

        self.log('grad_norm', grad_norm, rank_zero_only=True)
