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
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import (
    TextToTextGLUEDataset,
    TextToTextXNlIDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging

__all__ = ['MegatronXNlIModel']


class MegatronXNlIModel(MegatronT5GLUEModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg
        self.acc_metrics = ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)

    def process_batch(self, batch):
        """Build the batch."""
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = super().process_batch(batch)
        if 'lang' in batch:
            lang = batch['lang']
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, lang
        else:
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, langs = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10
        )

        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for i, (pred, label, lang) in enumerate(zip(preds, labels, langs)):
            if self.model.tokenizer.eos_id in pred:
                idx = pred.index(self.model.tokenizer.eos_id)
                pred = pred[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.model.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.model.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.model.tokenizer.special_token_to_id.values()]
            pred = self.model.tokenizer.ids_to_text(pred)
            label = self.model.tokenizer.ids_to_text(label)
            _ = self.acc_metrics(pred, label, lang)

        return {'loss': loss}

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        acc_result = self.acc_metrics.compute()
        self.log('validation_loss', averaged_loss)
        self.log('validation_acc', acc_result['acc'])
        for lang in self.cfg.eval_languages:
            self.log(f'{lang}_acc', acc_result[lang])
        self.acc_metrics.reset()
        return averaged_loss[0], acc_result

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def setup(self, stage=None):
        # NOTE: PTL uses the same stage string "test" for both testing and validation.
        self.build_train_valid_test_datasets(test_only=stage == 'test')
        self.setup_test_data()
        if stage == 'test':
            return
        self.setup_validation_data()
        self.setup_training_data()

    def setup_test_data(self, test_data_config=None):
        self._test_dl = self.build_pretraining_data_loader(
            self._test_ds,
            self.cfg.data.validation_ds.batch_size,
            shuffle=self.cfg.data.validation_ds.shuffle,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=self.cfg.data.validation_ds.pin_memory,
        )

    def validation_epoch_end(self, outputs):
        val_loss, val_acc = self.inference_epoch_end(outputs)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc['acc'], prog_bar=True)
        for lang in self.cfg.eval_languages:
            logging.info(f"Validation {lang} accuracy: {val_acc[lang]} total: {val_acc[lang+'_total']}")
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f"Validation accuracy: {val_acc['acc']}")

    def test_epoch_end(self, outputs):
        test_loss, test_acc = self.inference_epoch_end(outputs)
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', test_acc['acc'], prog_bar=True)
        for lang in self.cfg.eval_languages:
            logging.info(f"Test {lang} accuracy: {test_acc[lang]} total: {test_acc[lang+'_total']}")
        logging.info(f'Test loss: {test_loss}')
        logging.info(f"Test accuracy: {test_acc['acc']}")

    def build_train_valid_test_datasets(self, test_only=False):
        logging.info('Building GLUE datasets.')
        self._test_ds = TextToTextXNlIDataset(
            self.cfg.data.test_ds.file_path,
            task_name=self.cfg.data.test_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.test_ds.max_seq_length,
            lang_list=self.cfg.eval_languages,
        )
        if test_only:
            return None, None, self._test_ds
        self._train_ds = TextToTextGLUEDataset(
            self.cfg.data.train_ds.file_path,
            task_name=self.cfg.data.train_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.train_ds.max_seq_length,
        )
        self._validation_ds = TextToTextXNlIDataset(
            self.cfg.data.validation_ds.file_path,
            task_name=self.cfg.data.validation_ds.task_name,
            tokenizer=self.model.tokenizer,
            max_seq_length=self.cfg.data.validation_ds.max_seq_length,
            lang_list=self.cfg.eval_languages,
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T5 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds
