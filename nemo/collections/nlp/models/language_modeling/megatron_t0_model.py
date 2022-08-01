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
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.data import ConcatMapDataset
from nemo.collections.nlp.data.language_modeling.t0_dataset import T0JSONLMemMapDataset
from nemo.collections.nlp.models.language_modeling.megatron_finetune_model import MegatronT5FinetuneModel
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['MegatronT0Model']


class MegatronT0Model(MegatronT5FinetuneModel):
    """T0 (https://arxiv.org/abs/2110.08207) Model that Inherits from MegatronT5FinetuneModel and overrides the dataset building."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def _build_dataset(self, data_cfg, check_implict_grad_acc=False, is_train=True):
        if (
            check_implict_grad_acc
            and data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size()
        ):
            raise ValueError(
                f'You are trying to use "implicit gradient accumulation" of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} in your validation/test datasets. This is not supported. Please set global_batch_size equal to micro_batch_size * data_parallel_world_size.'
            )
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"T0 train/validation datasets must be provided as a list of individual JSONL files.")
        for file_name in data_cfg.file_names:
            dataset = T0JSONLMemMapDataset(
                dataset_paths=[file_name],
                tokenizer=self.tokenizer,
                max_src_seq_length=data_cfg.max_src_seq_length,
                max_tgt_seq_length=data_cfg.max_tgt_seq_length,
            )
            datasets.append(dataset)

        if is_train:
            return ConcatMapDataset(
                datasets=datasets,
                sampling_technique=data_cfg.get('concat_sampling_technique', 'temperature'),
                sampling_temperature=data_cfg.get('concat_sampling_temperature', 5),
                sampling_probabilities=data_cfg.get(
                    'concat_sampling_probabilities', [1 / len(datasets)] * len(datasets)
                ),
                consumed_samples=self.compute_consumed_samples(0),
            )
        else:
            return datasets

    def build_train_valid_test_datasets(self, stage):
        if stage != 'test':
            logging.info('Building T0 validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(
                self.cfg.data.validation_ds, check_implict_grad_acc=True, is_train=False
            )
            logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                logging.info('Building T0 test datasets.')
                # Wrap this in a list since the general finetuning parent class supports multi-validation.
                self._test_ds = self._build_dataset(self.cfg.data.test_ds, check_implict_grad_acc=True, is_train=False)
                logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

        if stage == 'validate' or stage == 'test':
            return
        logging.info('Building T0 traing datasets.')
        self._train_ds = self._build_dataset(self.cfg.data.train_ds, check_implict_grad_acc=False)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
