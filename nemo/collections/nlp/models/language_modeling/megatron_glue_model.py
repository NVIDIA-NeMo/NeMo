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
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import (
    TextToTextGLUEDataset,
    TextToTextXNLIDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.parts.nlp_overrides import GlobalBatchDataFetcher
from nemo.utils import AppState, app_state, logging

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5GLUEModel']


class MegatronT5GLUEModel(MegatronT5Model):
    """GLUE Model that Inherits from MegatronT5Model instead."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        if hasattr(self.cfg, 'eval_languages'):
            self.acc_metric = ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)
        else:
            self.acc_metric = ExactStringPerCategoryMatchMetric()

    def setup(self, stage=None):
        # This is just to keep the parent class happy since we override its setup() method.
        self.init_consumed_samples = 0
        self.init_global_step = 0
        if stage == 'predict':
            return

        # NOTE: PTL uses the same stage string "test" for both testing and validation.
        self.build_train_valid_test_datasets(stage=stage)
        if hasattr(self, '_validation_ds'):
            self.setup_validation_data()
        if hasattr(self, '_test_ds'):
            self.setup_test_data()
        if hasattr(self, '_train_ds'):
            self.setup_training_data()

    def _process_global_batch(self, global_batch):
        """ Prepares the global batch for apex fwd/bwd functions.
            Global batch is a list of micro batches.
        """
        text_enc_list = []
        text_dec_list = []
        labels_list = []
        loss_mask_list = []
        enc_mask_list = []
        dec_mask_list = []

        # Determine the maximum encoder and decoder sequence lengths amongst microbatches and pad each microbatch to the max seq length.
        # NOTE: This should only happen for model finetuning where we pad dynamically. Training uses fixed training shapes.

        max_enc_seq_lenth = max([micro_batch['text_enc'].shape[1] for micro_batch in global_batch])
        max_dec_seq_lenth = max([micro_batch['text_dec'].shape[1] for micro_batch in global_batch])

        for micro_batch in global_batch:
            text_enc, text_dec, loss_mask, labels, enc_mask, dec_mask = self.process_micro_batch(micro_batch)
            # Check if encoder sequence length < max encoder sequence length of the global batch and pad.
            if text_enc.shape[1] < max_enc_seq_lenth:
                text_enc = torch.nn.functional.pad(
                    text_enc, (0, max_enc_seq_lenth - text_enc.shape[1], 0, 0), 'constant', self.tokenizer.pad_id
                )
                enc_mask = torch.nn.functional.pad(
                    enc_mask, (0, max_enc_seq_lenth - enc_mask.shape[1], 0, 0), 'constant', 0
                )
            if text_dec.shape[1] < max_dec_seq_lenth:
                text_dec = torch.nn.functional.pad(
                    text_dec, (0, max_dec_seq_lenth - text_dec.shape[1], 0, 0), 'constant', self.tokenizer.pad_id
                )
                dec_mask = torch.nn.functional.pad(
                    dec_mask, (0, max_dec_seq_lenth - dec_mask.shape[1], 0, 0), 'constant', 0
                )
                labels = torch.nn.functional.pad(
                    labels, (0, max_dec_seq_lenth - labels.shape[1], 0, 0), 'constant', self.tokenizer.pad_id
                )
                loss_mask = torch.nn.functional.pad(
                    loss_mask, (0, max_dec_seq_lenth - loss_mask.shape[1], 0, 0), 'constant', 0
                )
            text_enc_list.append(text_enc)
            text_dec_list.append(text_dec)
            labels_list.append(labels)
            loss_mask_list.append(loss_mask)
            enc_mask_list.append(enc_mask)
            dec_mask_list.append(dec_mask)

        # Concatenate to (num_microbatches x micro_batch_size x seq_len)
        tokens_enc_tensor = torch.concat(text_enc_list, dim=0)
        tokens_dec_tensor = torch.concat(text_dec_list, dim=0)
        labels_tensor = torch.concat(labels_list, dim=0)
        loss_mask_tensor = torch.concat(loss_mask_list, dim=0)
        enc_mask_tensor = torch.concat(enc_mask_list, dim=0)
        dec_mask_tensor = torch.concat(dec_mask_list, dim=0)

        return tokens_enc_tensor, tokens_dec_tensor, loss_mask_tensor, labels_tensor, enc_mask_tensor, dec_mask_tensor

    def process_global_batch(self, global_batch):
        """Process a list of microbatches into a global batch."""
        # If there is no language information in the global batch (ex: English MNLI), we can use the parent global batch processor as is.
        if len(global_batch[0]) == 6:
            return self._process_global_batch(global_batch)

        # For validation data (XNLI), we need to process the global batch and and then deal with language info separately.
        else:
            assert len(global_batch[0]) == 7
            langs_list = []
            (
                tokens_enc_tensor,
                tokens_dec_tensor,
                loss_mask_tensor,
                labels_tensor,
                enc_mask_tensor,
                dec_mask_tensor,
            ) = self._process_global_batch(
                [{k: v for k, v in micro_batch.items() if k != 'lang'} for micro_batch in global_batch]
            )
            for micro_batch in global_batch:
                langs_list.extend(micro_batch['lang'])
            return (
                tokens_enc_tensor,
                tokens_dec_tensor,
                loss_mask_tensor,
                labels_tensor,
                enc_mask_tensor,
                dec_mask_tensor,
                langs_list,
            )

    def on_validation_epoch_start(self):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self):
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        return super().on_test_epoch_start()

    def on_validation_epoch_end(self):
        app_state = AppState()
        if hasattr(self, "_train_ds"):
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.validation_ds.global_batch_size,
                micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return super().on_validation_epoch_end()

    def training_step(self, batch, batch_idx):
        micro_batch_size = batch[0]['text_enc'].size(0)
        # This should happen only on the last batch of the dataset.
        if micro_batch_size != self.cfg.data.train_ds.micro_batch_size:
            app_state = AppState()
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=micro_batch_size
                * parallel_state.get_data_parallel_world_size()
                * get_num_microbatches(),
                micro_batch_size=micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        return super().training_step(batch, batch_idx)

    def inference_step(self, batch, batch_idx):
        batch_has_lang_information = len(batch[0]) == 7
        # XNLI Batches have language information that need to be removed before calling the parent validation step.
        if batch_has_lang_information:
            processed_batch = []
            for micro_batch in batch:
                micro_batch = {k: v for k, v in micro_batch.items() if k != 'lang'}
                processed_batch.append(micro_batch)
        else:
            processed_batch = batch

        micro_batch_size = processed_batch[0]['text_enc'].size(0)

        # This should happen only on the last batch of the dataset.
        if micro_batch_size != self.cfg.data.validation_ds.micro_batch_size:
            app_state = AppState()
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=micro_batch_size
                * parallel_state.get_data_parallel_world_size()
                * get_num_microbatches(),
                micro_batch_size=micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # Call parent validation step to get the loss.
        loss = super().validation_step(processed_batch, batch_idx)

        # Remainder of the code is to run the decoding loop, and compute accuracies.
        if batch_has_lang_information:
            tokens_enc, _, _, labels, enc_mask, _, langs = self.process_global_batch(batch)
        else:
            tokens_enc, _, _, labels, enc_mask, _ = self.process_global_batch(batch)

        predicted_token_ids, _ = self.decode(tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=10)

        preds_text, labels_text = self.preds_and_labels_to_text(predicted_token_ids, labels)

        if not batch_has_lang_information:
            langs = [None] * len(preds_text)

        assert len(langs) == len(preds_text) == len(labels_text)
        for _, (pred, label, lang) in enumerate(zip(preds_text, labels_text, langs)):
            _ = self.acc_metric(pred, label, lang)

        return loss

    def preds_and_labels_to_text(self, preds, labels):
        preds = preds.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        preds_text, labels_text = [], []
        for _, (pred, label) in enumerate(zip(preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.tokenizer.special_token_to_id.values()]
            pred = self.tokenizer.ids_to_text(pred)
            label = self.tokenizer.ids_to_text(label)
            preds_text.append(pred)
            labels_text.append(label)

        return preds_text, labels_text

    def inference_epoch_end(self, outputs, mode):
        # Parent class will handle logging of the loss.
        if mode == 'validation':
            averaged_loss = super().validation_epoch_end(outputs)
        else:
            averaged_loss = super().test_epoch_end(outputs)
        accuracy = self.acc_metric.compute()
        # Loss is logged in the parent epoch end class.
        self.log(f'{mode}_acc', accuracy['acc'])
        if hasattr(self.cfg, 'eval_languages'):
            for lang in self.cfg.eval_languages:
                self.log(f'{lang}_acc', accuracy[lang])
                logging.info(f"{mode} {lang} accuracy: {accuracy[lang]} total: {accuracy[lang+'_total']}")
        logging.info(f"{mode} accuracy: {accuracy['acc']}")
        self.acc_metric.reset()
        return averaged_loss, accuracy['acc']

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'validation')

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'test')

    def build_data_loader(
        self,
        dataset,
        micro_batch_size,
        global_batch_size,
        shuffle,
        num_workers,
        pin_memory,
        drop_last,
        check_validation_interval,
    ):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        # This check makes sure the val_check_interval is less than the number of global batches.
        # Normally, PTL would do this check and properly account for gradient accumulation.
        # But now, it is implicit in the apex fwd/bwd functions and so we need to check for this somewhere.
        # The consequence of not doing this is that training loop will never run validation.
        # NOTE: Prog bar is also broken as a result of this.
        global_batch_size_per_data_parallel_rank = global_batch_size // parallel_state.get_data_parallel_world_size()
        if (
            self.trainer.val_check_interval > (sampler.num_samples // global_batch_size_per_data_parallel_rank)
            and check_validation_interval
        ):
            raise ValueError(
                f"trainer.val_check_interval {self.trainer.val_check_interval} is > number of global batches {sampler.num_samples // global_batch_size}"
            )

        # Data loader. Note that batch size is the per GPU batch size.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=micro_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def setup_training_data(self):
        self._train_dl = self.build_data_loader(
            self._train_ds,
            micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
            global_batch_size=self.cfg.data.train_ds.global_batch_size,
            shuffle=self.cfg.data.train_ds.shuffle,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=self.cfg.data.train_ds.pin_memory,
            drop_last=self.cfg.data.train_ds.drop_last,
            check_validation_interval=True,
        )

    def setup_validation_data(self):
        self._validation_dl = self.build_data_loader(
            self._validation_ds,
            micro_batch_size=self.cfg.data.validation_ds.micro_batch_size,
            global_batch_size=self.cfg.data.validation_ds.global_batch_size,
            shuffle=self.cfg.data.validation_ds.shuffle,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=self.cfg.data.validation_ds.pin_memory,
            drop_last=self.cfg.data.validation_ds.drop_last,
            check_validation_interval=False,
        )

    def setup_test_data(self):
        self._test_dl = self.build_data_loader(
            self._test_ds,
            micro_batch_size=self.cfg.data.test_ds.micro_batch_size,
            global_batch_size=self.cfg.data.test_ds.global_batch_size,
            shuffle=self.cfg.data.test_ds.shuffle,
            num_workers=self.cfg.data.test_ds.num_workers,
            pin_memory=self.cfg.data.test_ds.pin_memory,
            drop_last=self.cfg.data.test_ds.drop_last,
            check_validation_interval=False,
        )

    def _build_dataset(self, data_cfg):
        if data_cfg.task_name == 'xnli':
            dataset = TextToTextXNLIDataset(
                data_cfg.file_path,
                task_name=data_cfg.task_name,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                lang_list=self.cfg.eval_languages,
            )
        else:
            dataset = TextToTextGLUEDataset(
                data_cfg.file_path,
                task_name=data_cfg.task_name,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
            )
        return dataset

    def build_train_valid_test_datasets(self, stage):
        logging.info('Building GLUE/XNLI datasets.')
        if stage != 'test':
            self._validation_ds = self._build_dataset(self.cfg.data.validation_ds)
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                self._test_ds = self._build_dataset(self.cfg.data.test_ds)
                logging.info(f'Length of test dataset: {len(self._test_ds)}')

        if stage == 'validate' or stage == 'test':
            return
        self._train_ds = self._build_dataset(self.cfg.data.train_ds)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Finished building GLUE/XNLI datasets.')

    def on_train_start(self) -> None:
        """PTL hook used to override DataFetcher with GlobalBatchDataFetcher """
        self.trainer.fit_loop._data_fetcher = GlobalBatchDataFetcher()

    def on_validation_start(self) -> None:
        """PTL hook used to override DataFetcher with GlobalBatchDataFetcher """
        self.trainer.fit_loop.epoch_loop.val_loop._data_fetcher = GlobalBatchDataFetcher()
        self.trainer.validate_loop._data_fetcher = GlobalBatchDataFetcher()

    def on_test_start(self) -> None:
        self.trainer.test_loop._data_fetcher = GlobalBatchDataFetcher()
