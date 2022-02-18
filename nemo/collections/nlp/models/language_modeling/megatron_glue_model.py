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

from operator import itemgetter
from typing import Any, Optional

import torch
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import TextToTextGLUEDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.core.optim.lr_scheduler import prepare_lr_scheduler
from nemo.core.optim.optimizer_with_main_params import MainParamsOptimizerWrapper
from nemo.utils import logging

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5FineTuneModel']


class MegatronT5FineTuneModel(NLPModel):
    """
    Base class for finetuning pre-trained T5 models.
    Inherit from this class and implement the dataset building and train/validation steps.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg, trainer)
        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)
        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(
            self.register_artifact('t5_base_model', cfg.restore_from_path), trainer=trainer, return_config=True
        )
        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_o2

        self.model = MegatronT5Model.restore_from(
            self.register_artifact('t5_base_model', cfg.restore_from_path),
            trainer=trainer,
            override_config_path=t5_cfg,
        )
        self.setup_optimizer_param_groups()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def process_batch(self, batch):
        return self.model.process_batch(batch)

    def build_train_valid_test_datasets(self):
        pass

    def build_pretraining_data_loader(self, dataset):
        pass

    def setup(self, stage=None):
        pass

    def setup_training_data(self, train_data_config=None):
        pass

    def setup_validation_data(self, validation_data_config=None):
        pass

    def setup_test_data(self, test_data_config=None):
        pass

    def configure_optimizers(self):
        self.setup_optimization()

        # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_o2 and self._optimizer is not None:
            if self.cfg.precision == 'bf16':
                fp32_grad_accum = True
                contiguous_grad_bucket = True
                async_grad_allreduce = True

            elif self.cfg.precision == 16:
                fp32_grad_accum = False
                # TODO: contiguous grad bucket for fp16 is also planned to be supported
                contiguous_grad_bucket = False
                async_grad_allreduce = False

            self._optimizer = MainParamsOptimizerWrapper(
                self._optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
            )
            assert self._trainer.max_steps is not None, "'max_steps' is missing in trainer config."
            if hasattr(self._cfg.optim, 'sched'):
                sched_config = self._cfg.optim.sched
                sched_config['max_steps'] = self._trainer.max_steps
                self._scheduler = prepare_lr_scheduler(
                    optimizer=self._optimizer, scheduler_config=sched_config, train_dataloader=self._train_dl
                )

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = _get_params_for_weight_decay_optimization([self.model.enc_dec_model])

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self.model.prediction_step(batch, batch_idx, dataloader_idx)


class MegatronT5GLUEModel(MegatronT5FineTuneModel):
    """
    Megatron T5 finetuning for GLUE datasets.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg
        self.acc_metric = ExactStringPerCategoryMatchMetric()

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)

        tokens_loss = itemgetter("tokens_loss")(
            self.model(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        )

        loss = self.model.loss_func(loss_mask, tokens_loss)
        self.log('train_loss', loss)
        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self.model._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self.model._reduced_loss_buffer) / len(self.model._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self.model._reduced_loss_buffer = []

        return loss

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, _, _, labels, enc_mask, _ = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10
        )

        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for i, (pred, label) in enumerate(zip(preds, labels)):
            if self.model.tokenizer.eos_id in pred:
                idx = pred.index(self.model.tokenizer.eos_id)
                pred = pred[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.model.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.model.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.model.tokenizer.special_token_to_id.values()]
            pred = self.model.tokenizer.ids_to_text(pred)
            label = self.model.tokenizer.ids_to_text(label)
            _ = self.acc_metric(pred, label)

        return {'loss': loss}

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        val_acc = self.acc_metric.compute()
        self.log('validation_loss', averaged_loss)
        self.log('validation_acc', val_acc['acc'])
        self.acc_metric.reset()
        return averaged_loss[0], val_acc

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        val_loss, val_acc = self.inference_epoch_end(outputs)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'Validation accuracy: {val_acc}')

    def test_step(self, batch, batch_idx):
        raise NotImplementedError(
            "Testing is not supported for GLUE because the test data does not have labels. To evaluate on the validation dataset, call trainer.validate(model)"
        )

    def test_epoch_end(self, outputs):
        raise NotImplementedError(
            "Testing is not supported for GLUE because the test data does not have labels. To evaluate on the validation dataset, call trainer.validate(model)"
        )

    def build_train_valid_test_datasets(self, validation_only=False):
        logging.info('Building GLUE datasets.')
        self._validation_ds = TextToTextGLUEDataset(
            self.cfg.data.validation_ds.file_path,
            task_name=self.cfg.data.validation_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.validation_ds.max_seq_length,
        )
        if validation_only:
            return None, self._validation_ds
        self._train_ds = TextToTextGLUEDataset(
            self.cfg.data.train_ds.file_path,
            task_name=self.cfg.data.train_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.train_ds.max_seq_length,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Finished building T5 datasets.')
        return self._train_ds, self._validation_ds

    def build_pretraining_data_loader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

        # Data loader. Note that batch size is the per GPU batch size.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def setup(self, stage=None):
        if stage == 'predict':
            return

        # NOTE: PTL uses the same stage string "test" for both testing and validation.
        self.build_train_valid_test_datasets(validation_only=stage == 'validate')
        self.setup_validation_data()
        if stage == 'validate':
            return
        self.setup_training_data()

    def setup_training_data(self, train_data_config=None):
        self._train_dl = self.build_pretraining_data_loader(
            self._train_ds,
            self.cfg.data.train_ds.batch_size,
            shuffle=self.cfg.data.train_ds.shuffle,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=self.cfg.data.train_ds.pin_memory,
        )

    def setup_validation_data(self, validation_data_config=None):
        self._validation_dl = self.build_pretraining_data_loader(
            self._validation_ds,
            self.cfg.data.validation_ds.batch_size,
            shuffle=self.cfg.data.validation_ds.shuffle,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=self.cfg.data.validation_ds.pin_memory,
        )

    @classmethod
    def list_available_models(cls):
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result
