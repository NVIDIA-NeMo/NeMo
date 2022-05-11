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
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.data.common.sequence_to_sequence_dataset import SequenceToSequenceDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.parts.nlp_overrides import GlobalBatchDataFetcher
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5FinetuneModel']


class MegatronT5FinetuneModel(MegatronT5Model):
    """Finetune Model that Inherits from MegatronT5Model instead."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.val_metric = self.setup_metric(self.cfg.data.validation_ds)
        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric = self.setup_metric(self.cfg.data.test_ds)

    def setup_metric(self, data_cfg):
        # XNLI
        if hasattr(self.cfg, "eval_languages"):
            metric = ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)
        # GLUE
        elif hasattr(data_cfg, "task_name"):
            metric = ExactStringPerCategoryMatchMetric()
        # General Seq2seq finetuning
        else:
            if isinstance(data_cfg.src_file_name, ListConfig):
                if hasattr(data_cfg, "names") and isinstance(data_cfg.names, ListConfig):
                    metric = ExactStringPerCategoryMatchMetric(self.cfg.data.validation_ds.names)
                else:
                    metric = ExactStringPerCategoryMatchMetric(
                        [str(i) for i in range(len(self.cfg.data.test_ds.src_file_name))]
                    )
            else:
                metric = ExactStringPerCategoryMatchMetric()

        return metric

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

    def inference_step(self, batch, batch_idx, mode, dataloader_idx=0):
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

        predicted_token_ids, _ = self.decode(tokens_enc=tokens_enc, enc_mask=enc_mask, num_tokens_to_generate=30)

        preds_text, labels_text = self.preds_and_labels_to_text(predicted_token_ids, labels)

        if not batch_has_lang_information:
            if (
                mode == 'validation'
                and hasattr(self.cfg.data.validation_ds, "names")
                and isinstance(self.cfg.data.validation_ds.names, ListConfig)
            ):
                categories = [self.cfg.data.validation_ds.names[dataloader_idx]] * len(preds_text)
            elif (
                mode == 'test'
                and hasattr(self.cfg.data.test_ds, "names")
                and isinstance(self.cfg.data.test_ds.names, ListConfig)
            ):
                categories = [self.cfg.data.test_ds.names[dataloader_idx]] * len(preds_text)
            else:
                categories = [None] * len(preds_text)
        else:
            categories = langs

        metric = self.val_metric if mode == 'validation' else self.test_metric
        assert len(categories) == len(preds_text) == len(labels_text)
        for _, (pred, label, category) in enumerate(zip(preds_text, labels_text, categories)):
            _ = metric(pred, label, category)

        return {'loss': loss, 'preds': preds_text, 'labels': labels_text, 'categories': categories}

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
        if not outputs:
            return
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for _, output in enumerate(outputs):
            if mode == 'validation':
                averaged_loss = super().validation_epoch_end([x['loss'] for x in output])
            else:
                averaged_loss = super().test_epoch_end([x['loss'] for x in output])

        if mode == 'validation':
            accuracy = self.val_metric.compute()
        else:
            accuracy = self.test_metric.compute()
        # Loss is logged in the parent epoch end class.
        self.log(f'{mode}_acc', accuracy['acc'])
        logging.info(f"{mode} accuracy: {accuracy['acc']}")

        for k, v in accuracy.items():
            if k != 'acc' and 'total' not in k:
                logging.info(f"{mode} {k} accuracy: {v} total: {accuracy[k+'_total']}")
                self.log(f'{mode}_{k}_acc', v)

        if mode == 'validation':
            self.val_metric.reset()
        else:
            self.test_metric.reset()

        return averaged_loss, accuracy['acc']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.inference_step(batch, batch_idx, 'validation', dataloader_idx)

    def validation_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'validation')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.inference_step(batch, batch_idx, 'test', dataloader_idx)

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
        global_batch_size_per_gpu = micro_batch_size * get_num_microbatches()
        if (
            self.trainer.val_check_interval > (sampler.num_samples // global_batch_size_per_gpu)
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

    def setup_eval_data(self, datasets, data_cfg):
        dataloaders = []
        for dataset in datasets:
            eval_dl = self.build_data_loader(
                dataset,
                micro_batch_size=data_cfg.micro_batch_size,
                global_batch_size=data_cfg.global_batch_size,
                shuffle=data_cfg.shuffle,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                drop_last=data_cfg.drop_last,
                check_validation_interval=False,
            )
            dataloaders.append(eval_dl)
        return dataloaders

    def setup_validation_data(self):
        self._validation_dl = self.setup_eval_data(self._validation_ds, self.cfg.data.validation_ds)

    def setup_test_data(self):
        self._test_dl = self.setup_eval_data(self._test_ds, self.cfg.data.test_ds)

    def _build_train_dataset(self, data_cfg):
        """Build the training dataset."""
        if (
            data_cfg.drop_last is False
            and data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size()
        ):
            raise ValueError(
                f"Cannot use drop_last=False in your training data with gradient accumulation found grad acc of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} with global_batch_size {data_cfg.global_batch_size}, micro_batch_size {data_cfg.micro_batch_size}, data parallel size {parallel_state.get_data_parallel_world_size()}"
            )
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_src_list_config = isinstance(data_cfg.src_file_name, ListConfig)
        is_tgt_list_config = isinstance(data_cfg.tgt_file_name, ListConfig)

        if (is_src_list_config and not is_tgt_list_config) or (is_tgt_list_config and not is_src_list_config):
            raise ValueError("src_list and tgt_list must both be either a ListConfig or a string. ")
        if is_src_list_config:
            if len(data_cfg.src_file_name) != len(data_cfg.tgt_file_name):
                raise ValueError("src_file_name and tgt_file_name must have the same number of elements. ")
        else:
            data_cfg.src_file_name = [data_cfg.src_file_name]
            data_cfg.tgt_file_name = [data_cfg.tgt_file_name]

        for src, tgt in zip(data_cfg.src_file_name, data_cfg.tgt_file_name):
            dataset = SequenceToSequenceDataset(
                src_file_name=src,
                tgt_file_name=tgt,
                src_tokenizer=self.tokenizer,
                tgt_tokenizer=self.tokenizer,
                max_src_seq_length=data_cfg.max_src_seq_length,
                max_tgt_seq_length=data_cfg.max_tgt_seq_length,
            )
            datasets.append(dataset)

        if len(datasets) > 1:
            dataset = ConcatDataset(
                datasets=datasets,
                sampling_technique=data_cfg.get('concat_sampling_technique', 'temperature'),
                sampling_temperature=data_cfg.get('concat_sampling_temperature', 5),
                sampling_probabilities=data_cfg.get(
                    'concat_sampling_probabilities', [1 / len(datasets)] * len(datasets)
                ),
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )
            return dataset
        else:
            return datasets[0]

    def _build_eval_dataset(self, data_cfg, mode='train'):
        """Build the evaluation dataset."""
        if data_cfg.global_batch_size > data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size():
            raise ValueError(
                f'You are trying to use "implicit gradient accumulation" of {data_cfg.global_batch_size // (data_cfg.micro_batch_size * parallel_state.get_data_parallel_world_size())} in your validation/test datasets. This is not supported. Please set global_batch_size equal to micro_batch_size * data_parallel_world_size.'
            )
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_src_list_config = isinstance(data_cfg.src_file_name, ListConfig)
        is_tgt_list_config = isinstance(data_cfg.tgt_file_name, ListConfig)
        is_names_list_config = False
        if hasattr(data_cfg, "names"):
            if isinstance(data_cfg.names, ListConfig):
                is_names_list_config = True

        if (is_src_list_config and not is_tgt_list_config) or (is_tgt_list_config and not is_src_list_config):
            raise ValueError("src_list and tgt_list must both be either a ListConfig or a string. ")
        if is_src_list_config:
            if len(data_cfg.src_file_name) != len(data_cfg.tgt_file_name):
                raise ValueError("src_file_name and tgt_file_name must have the same number of elements. ")
            if is_names_list_config and len(data_cfg.names) != len(data_cfg.src_file_name):
                raise ValueError(
                    "If you are providing names for each src/tgt file, they must have the same number of elements."
                )
        else:
            data_cfg.src_file_name = [data_cfg.src_file_name]
            data_cfg.tgt_file_name = [data_cfg.tgt_file_name]

        for src, tgt in zip(data_cfg.src_file_name, data_cfg.tgt_file_name):
            dataset = SequenceToSequenceDataset(
                src_file_name=src,
                tgt_file_name=tgt,
                src_tokenizer=self.tokenizer,
                tgt_tokenizer=self.tokenizer,
                max_src_seq_length=data_cfg.max_src_seq_length,
                max_tgt_seq_length=data_cfg.max_tgt_seq_length,
            )
            datasets.append(dataset)

        if mode == 'train' and len(datasets) > 1:
            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    sampling_technique=data_cfg.get('concat_sampling_technique', 'temperature'),
                    sampling_temperature=data_cfg.get('concat_sampling_temperature', 5),
                    sampling_probabilities=data_cfg.get(
                        'concat_sampling_probabilities', [1 / len(datasets)] * len(datasets)
                    ),
                    global_rank=parallel_state.get_data_parallel_rank(),
                    world_size=parallel_state.get_data_parallel_world_size(),
                )
            return dataset

        return datasets

    def build_train_valid_test_datasets(self, stage):
        logging.info('Building datasets ...')
        if stage != 'test':
            self._validation_ds = self._build_eval_dataset(self.cfg.data.validation_ds, mode='validation')

        if stage != 'validation':
            if hasattr(self.cfg.data, 'test_ds'):
                self._test_ds = self._build_eval_dataset(self.cfg.data.test_ds, mode='test')

        if stage == 'validation' or stage == 'test':
            return
        self._train_ds = self._build_train_dataset(self.cfg.data.train_ds)
        logging.info(f'Finished building datasets ...')

    def on_train_start(self) -> None:
        """PTL hook used to override DataFetcher with GlobalBatchDataFetcher """
        self.trainer.fit_loop._data_fetcher = GlobalBatchDataFetcher()

    def on_validation_start(self) -> None:
        """PTL hook used to override DataFetcher with GlobalBatchDataFetcher """
        self.trainer.fit_loop.epoch_loop.val_loop._data_fetcher = GlobalBatchDataFetcher()
        self.trainer.validate_loop._data_fetcher = GlobalBatchDataFetcher()

    def on_test_start(self) -> None:
        self.trainer.test_loop._data_fetcher = GlobalBatchDataFetcher()
