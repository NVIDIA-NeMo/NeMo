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
import json

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.metrics import MetricStringToTorchMetric
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
        self.val_metric, self.val_metric_name = self.setup_metric(self.cfg.data.validation_ds)
        self.val_metric = torch.nn.ModuleList(self.val_metric)
        if hasattr(self.cfg.data, "test_ds"):
            self.test_metric, self.test_metric_name = self.setup_metric(self.cfg.data.test_ds)
            self.test_metric = torch.nn.ModuleList(self.test_metric)

    def setup_metric(self, data_cfg):
        # XNLI is a special case.
        metric_name = "exact_string_match"
        if hasattr(self.cfg, "eval_languages"):
            metric = [ExactStringPerCategoryMatchMetric(self.cfg.eval_languages)]
        else:
            if not hasattr(data_cfg, "metric"):
                metric = MetricStringToTorchMetric["exact_string_match"]
            else:
                if not hasattr(data_cfg.metric, "name"):
                    raise ValueError("Metric name is not provided in the metric config.")
                if data_cfg.metric.name not in MetricStringToTorchMetric:
                    raise KeyError(
                        f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                    )
            metric_name = data_cfg.metric.name
            metric = MetricStringToTorchMetric[metric_name]
            # GLUE will not have a "src_file_name" attribute and will always have only a single metric.
            if hasattr(data_cfg, "src_file_name"):
                if isinstance(data_cfg.src_file_name, ListConfig):
                    # We pass average and num_classes to the metric constructor via kwargs even if they don't exist for each metric.
                    metric = [
                        metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)
                        for _ in range(len(self.cfg.data.test_ds.src_file_name))
                    ]
                else:
                    metric = [metric(average=data_cfg.metric.average, num_classes=data_cfg.metric.num_classes)]
            else:
                metric = [metric()]  # GLUE does need to specify average or num_classes.

        return metric, metric_name

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
        """Process a list of microbatches into a global batch."""
        # If there is no language information in the global batch (ex: English MNLI), we can use the parent global batch processor as is.
        if 'lang' not in global_batch[0]:
            return self._process_global_batch_without_megatron_batch_sampler(global_batch)

        # For validation data (XNLI), we need to process the global batch and and then deal with language info separately.
        else:
            assert all(['lang' in micro_batch for micro_batch in global_batch])
            langs_list = []
            processed_global_batch = self._process_global_batch_without_megatron_batch_sampler(
                [{k: v for k, v in micro_batch.items() if k != 'lang'} for micro_batch in global_batch]
            )
            for micro_batch in global_batch:
                langs_list.extend(micro_batch['lang'])
            processed_global_batch['lang'] = langs_list
            return processed_global_batch

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
        # At this point batch is a list of dictionaries where eatch dict is a microbatch.
        # After the process_global_batch call, batch will be a single dictionary containing the global batch.
        # This is required since the parent class expects a single global batch dictioanry.
        batch = self._process_global_batch(batch)
        return super().training_step(batch, batch_idx)

    def cast_for_metric(self, pred, label, metric_name):
        if metric_name == 'exact_string_match':
            return pred, label
        pred = pred.replace(' ', '')
        label = label.replace(' ', '')

        # Correlation metrics require casting to float.
        if metric_name in ['pearson_corr_coef', 'spearman_corr_coef']:
            # Text-to-text model predictions may not always be valid floating point numbers.
            try:
                pred = float(pred)
            except ValueError:
                pred = 0.0

            try:
                label = float(label)
            except ValueError:
                raise ValueError(f'Could not convert {label} to float.')

            pred = torch.FloatTensor([pred]).to(self.device)
            label = torch.FloatTensor([label]).to(self.device)

        # Other metrics require casting to integers.
        elif metric_name in ['accuracy', 'auc', 'auroc', 'average_precision', 'f1']:
            # Text-to-text model predictions may not always be valid integers.
            try:
                pred = int(pred)
            except ValueError:
                pred = 0

            try:
                label = int(label)
            except ValueError:
                raise ValueError(f'Could not convert {label} to int.')

            pred = torch.LongTensor([pred]).to(self.device)
            label = torch.LongTensor([label]).to(self.device)

        else:
            raise ValueError(f'Metric {metric_name} not supported.')

        return pred, label

    def inference_step(self, batch, batch_idx, mode, dataloader_idx=0):
        batch_has_lang_information = len(batch[0]) == 7

        micro_batch_size = batch[0]['text_enc'].size(0)
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

        # At this point processed_batch is a list of dictionaries where eatch dict is a microbatch.
        # After the process_global_batch call, processed_batch will be a single dictionary containing the global batch.
        # This is required since the parent class expects a single global batch dictioanry.
        processed_batch = self._process_global_batch(batch)

        # Call parent validation step to get the loss.
        # NOTE: There could be extra keys in the processed_batch dictionary such as "langs" for XNLI, this will be ignored in the parent class.
        loss = super().validation_step(processed_batch, batch_idx)

        predicted_token_ids, _ = self.decode(
            tokens_enc=processed_batch['text_enc'], enc_mask=processed_batch['enc_mask'], num_tokens_to_generate=30
        )

        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = self.ids_to_text(predicted_token_ids)
        labels_text = self.ids_to_text(processed_batch['labels'])
        input_text = self.ids_to_text(processed_batch['text_enc'])

        if not batch_has_lang_information:
            categories = [None] * len(preds_text)
        else:
            categories = processed_batch['lang']

        metric = self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
        assert len(categories) == len(preds_text) == len(labels_text)
        for _, (pred, label, category) in enumerate(zip(preds_text, labels_text, categories)):
            # To compute metrics like pearson or spearman correlation, we need to cast the predicted string and labels to floats.
            pred, label = self.cast_for_metric(
                pred, label, self.val_metric_name if mode == 'validation' else self.test_metric_name
            )
            if batch_has_lang_information:
                _ = metric(pred, label, category)
            else:
                _ = metric(pred, label)

        return {
            'loss': loss,
            'preds': preds_text,
            'labels': labels_text,
            'categories': categories,
            'inputs': input_text,
        }

    def ids_to_text(self, batch_ids):
        batch_ids = batch_ids.cpu().numpy().tolist()
        texts = []
        for ids in batch_ids:
            if self.tokenizer.eos_id in ids:
                idx = ids.index(self.tokenizer.eos_id)
                ids = ids[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.tokenizer, 'special_token_to_id'):
                ids = [id for id in ids if id not in self.tokenizer.special_token_to_id.values()]
            text = self.tokenizer.ids_to_text(ids)
            texts.append(text)

        return texts

    def _determine_log_key(self, data_config, dataloader_idx, metric_name, mode):
        # Function that determines whether to log based on the user provided name of the dataset or the dataloader index.
        base_key = f"{mode}_{metric_name}_" if metric_name is not None else f"{mode}_"
        # If the user provided names for each validation/test dataset, use those.
        if hasattr(data_config, "names") and data_config.names is not None:
            # With only a single validation/test dataset, the name is not a list.
            if not isinstance(data_config.names, ListConfig):
                name = data_config.names
            else:
                name = data_config.names[dataloader_idx]
            return base_key + name
        else:
            return base_key + f"dataloader{dataloader_idx}"

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs:
            return
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            loss = super().validation_epoch_end([x['loss'] for x in output])
            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            # Determine the key used to log the eval metric based on the user provided name of the dataset or the dataloader index.
            metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
            self.log(loss_log_key, loss)
            metric_object = (
                self.val_metric[dataloader_idx] if mode == 'validation' else self.test_metric[dataloader_idx]
            )
            metric = metric_object.compute()
            # Handle logging of GLUE/XNLI separately here. XNLI has a separate metric per language.
            if isinstance(metric, dict):
                # GLUE case:
                if len(metric) == 1 and 'acc' in metric:
                    metric = metric['acc']
                    self.log(metric_log_key, metric)
                    logging.info(f"{mode} {metric_name}: {metric}")
                # XNLI case where the metric dictionary contains the language and the computed metric as values.
                else:
                    for k, v in metric.items():
                        if k != 'acc' and 'total' not in k:
                            self.log(metric_log_key + f'_{k}', v)
                            logging.info(f"{mode} {metric_name} lang {k} : {v}")
                    metric = metric['acc']
            else:
                self.log(metric_log_key, metric)
                logging.info(f"{mode} {metric_name}: {metric}")
            metric_object.reset()

            averaged_loss.append(loss)
            averaged_metric.append(metric)

            # Write predictions, labels, and inputs to a file for each validation/test dataset.
            if data_cfg.get("write_predictions_to_file", False):

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )

                # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
                gathered_outputs = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(
                    gathered_outputs,
                    [
                        {
                            'preds': x['preds'],
                            'labels': x['labels'],
                            'categories': x['categories'],
                            'inputs': x['inputs'],
                        }
                        for x in output
                    ],
                )

                # Figure out what the suffix of the file should be.
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)

                # Keep a set of ground truths and inputs to write deduplicated predictions. Distributed Sampler may duplicate examples.
                gt_inp_set = set()
                deduplicated_outputs = {
                    'preds': [],
                    'labels': [],
                    'categories': [],
                    'inputs': [],
                }

                # PTL models have a self.global_rank attribute and we want to write to disk only on global rank 0.
                if self.global_rank == 0:
                    for rank in range(0, self.world_size):
                        for batch in gathered_outputs[rank]:
                            for pred, label, input, category in zip(
                                batch['preds'], batch['labels'], batch['inputs'], batch['categories']
                            ):
                                if input + label not in gt_inp_set:
                                    gt_inp_set.add(input + label)
                                    deduplicated_outputs['preds'].append(pred)
                                    deduplicated_outputs['labels'].append(label)
                                    deduplicated_outputs['categories'].append(category)
                                    deduplicated_outputs['inputs'].append(input)
                self.write_predictions_to_file(
                    deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}"
                )
                torch.distributed.barrier()

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric)

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if torch.isinf(averaged_metric) or torch.isnan(averaged_metric):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("validation_loss", averaged_loss)
            self.log(f"validation_{self.val_metric_name}", averaged_metric)
        elif mode == 'test':
            self.log("test_loss", averaged_loss)
            self.log(f"test_{self.test_metric_name}", averaged_metric)

        return averaged_loss, averaged_metric

    def write_predictions_to_file(self, outputs, output_file_path_prefix):
        with open(output_file_path_prefix + "_inputs_preds_labels.json", "w") as f_json:
            json_output = {
                "inputs": outputs["inputs"],
                "preds": outputs["preds"],
                "labels": outputs["labels"],
            }
            json.dump(json_output, f_json)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.inference_step(batch, batch_idx, 'validation', dataloader_idx)

    def validation_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'validation', self.cfg.data.validation_ds)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.inference_step(batch, batch_idx, 'test', dataloader_idx)

    def test_epoch_end(self, outputs):
        _ = self.inference_epoch_end(outputs, 'test', self.cfg.data.test_ds)

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

    def _build_eval_dataset(self, data_cfg):
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

        return datasets

    def build_train_valid_test_datasets(self, stage):
        logging.info('Building datasets ...')
        if stage != 'test':
            self._validation_ds = self._build_eval_dataset(self.cfg.data.validation_ds)

        if stage != 'validation':
            if hasattr(self.cfg.data, 'test_ds'):
                self._test_ds = self._build_eval_dataset(self.cfg.data.test_ds)

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
