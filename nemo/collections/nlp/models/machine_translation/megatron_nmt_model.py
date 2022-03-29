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
import itertools
import random
from typing import Optional

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.trainer.trainer import Trainer
from sacrebleu import corpus_bleu

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import (
    MegatronLMEncoderDecoderModel,
)
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ["MegatronNMTModel"]


class MegatronNMTModel(MegatronLMEncoderDecoderModel):
    """
    Megatron NMT training
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        # All of the lines below need to be set when the parent class calls self._build_tokenizer()
        self.encoder_tokenizer_library = cfg.encoder_tokenizer.get('library', 'yttm')
        self.decoder_tokenizer_library = cfg.decoder_tokenizer.get('library', 'yttm')
        self.special_tokens = {}
        self.src_language = cfg.get("src_language", None)
        self.tgt_language = cfg.get("tgt_language", None)

        self.multilingual = cfg.get("multilingual", False)
        self.multilingual_ids = []

        self.validate_input_ids = cfg.get("validate_input_ids", True)
        if self.multilingual:
            if isinstance(self.src_language, ListConfig) and isinstance(self.tgt_language, ListConfig):
                raise ValueError(
                    "cfg.src_language and cfg.tgt_language cannot both be lists. We only support many-to-one or one-to-many multilingual models."
                )
            elif isinstance(self.src_language, ListConfig):
                pass
            elif isinstance(self.tgt_language, ListConfig):
                for lng in self.tgt_language:
                    self.special_tokens["<" + lng + ">"] = "<" + lng + ">"
            else:
                raise ValueError(
                    "Expect either cfg.src_language or cfg.tgt_language to be a list when multilingual=True."
                )

        super().__init__(cfg, trainer=trainer)

    def setup(self, stage=None):
        # NOTE: super().__init__ will try and setup train/val/test datasets, but we sidestep this using a if self._train_ds is not None condition
        # We then set things up for real only once setup() of this class is called.
        self.init_consumed_samples = 0  # This is just to keep the parent class happy.
        if stage == 'predict':
            return

        # If the user wants to manually override train and validation dataloaders before calling `.fit()`
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self._cfg.train_ds)
        self.setup_validation_data(self._cfg.validation_ds)
        self.setup_test_data(self._cfg.test_ds)

    def _build_tokenizer(self):
        # Instantiates tokenizers and register to be saved with NeMo Model archive
        # After this call, there will be self.encoder_tokenizer and self.decoder_tokenizer
        # Which can convert between tokens and token_ids for SRC and TGT languages correspondingly.
        encoder_tokenizer_model = self.register_artifact(
            "encoder_tokenizer.tokenizer_model", self._cfg.encoder_tokenizer.get('tokenizer_model')
        )
        decoder_tokenizer_model = self.register_artifact(
            "decoder_tokenizer.tokenizer_model", self._cfg.decoder_tokenizer.get('tokenizer_model')
        )

        self.encoder_tokenizer, self.decoder_tokenizer = MTEncDecModel.setup_enc_dec_tokenizers(
            encoder_tokenizer_library=self.encoder_tokenizer_library,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0)
            if self._cfg.encoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            encoder_model_name=None,
            encoder_r2l=self._cfg.encoder_tokenizer.get('r2l', False),
            decoder_tokenizer_library=self.decoder_tokenizer_library,
            encoder_tokenizer_vocab_file=self._cfg.encoder_tokenizer.get('vocab_file', None),
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0)
            if self._cfg.decoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            decoder_model_name=None,
            decoder_r2l=self._cfg.decoder_tokenizer.get('r2l', False),
            special_tokens=self.special_tokens,
        )

        # Set up pre and post processors as well.
        if self.multilingual:
            (
                self.source_processor_list,
                self.target_processor_list,
                self.multilingual_ids,
            ) = MTEncDecModel.setup_multilingual_ids_and_processors(
                src_language=self.src_language,
                tgt_language=self.tgt_language,
                tokenizer=self.encoder_tokenizer,  # Multilingual training requires shared tokenizers.
                tokenizer_library=self.encoder_tokenizer_library,
            )
        else:
            # After this call, the model will have  self.source_processor and self.target_processor objects
            MTEncDecModel.setup_pre_and_post_processing_utils(
                self.src_language, self.tgt_language, self.encoder_tokenizer_library, self.decoder_tokenizer_library,
            )
            self.multilingual_ids = [None]

    def _build_vocab(self):
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.encoder_tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def training_step(self, batch, batch_idx):
        # Need to squeze dim 0 for tarred datasets since things are pre-batched and we ask the dataloader for batch size 1.
        batch = [x.squeeze(dim=0) if x.ndim == 3 else x for x in batch]
        return super().training_step(batch, batch_idx)

    def eval_step(self, batch, batch_idx, dataloader_idx):
        # Need to squeze dim 0 for tarred datasets since things are pre-batched and we ask the dataloader for batch size 1.
        batch = [x.squeeze(dim=0) if x.ndim == 3 else x for x in batch]

        # This returns the averaged loss across data-parallel groups.
        reduced_loss = super().validation_step(batch, batch_idx)
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)
        predicted_tokens_ids, _ = self.decode(
            tokens_enc,
            enc_mask,
            tokens_enc.size(1)
            + self._cfg.max_generation_delta,  # Generate up to src-length + max generation delta. TODO: Implement better stopping when everything hits <EOS>.
            tokenizer=self.decoder_tokenizer,
        )

        # Post-process the translations and inputs to log.

        # Convert ids to lists.
        preds = predicted_tokens_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        encoder_inputs = tokens_enc.cpu().numpy().tolist()

        # Filter out the special tokens and de-tokenize.
        inputs = []
        translations = []
        ground_truths = []
        for _, (pred, label, input) in enumerate(zip(preds, labels, encoder_inputs)):
            if self.decoder_tokenizer.eos_id in pred:
                idx = pred.index(self.decoder_tokenizer.eos_id)
                pred = pred[:idx]

            # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
            if hasattr(self.decoder_tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.decoder_tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.decoder_tokenizer.special_token_to_id.values()]

            pred = self.decoder_tokenizer.ids_to_text(pred)
            label = self.decoder_tokenizer.ids_to_text(label)
            input = self.encoder_tokenizer.ids_to_text(input)
            translations.append(pred)
            ground_truths.append(label)
            inputs.append(input)

        if self.multilingual:
            self.source_processor = self.source_processor_list[dataloader_idx]
            self.target_processor = self.target_processor_list[dataloader_idx]

        # De-tokenize inputs, translations and ground truths.
        if self.target_processor is not None:
            ground_truths = [self.target_processor.detokenize(tgt.split(' ')) for tgt in ground_truths]
            translations = [self.target_processor.detokenize(translation.split(' ')) for translation in translations]

        if self.source_processor is not None:
            inputs = [self.source_processor.detokenize(src.split(' ')) for src in inputs]

        return {
            'inputs': inputs,
            'translations': translations,
            'ground_truths': ground_truths,
            'loss': reduced_loss,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self.eval_step(batch, batch_idx, dataloader_idx)

    def _setup_eval_dataloader_from_config(self, cfg: DictConfig, dataset):

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        dataloaders = []
        for _dataset in dataset:
            sampler = torch.utils.data.distributed.DistributedSampler(
                _dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset=_dataset,
                    batch_size=1,
                    sampler=sampler,
                    num_workers=cfg.get("num_workers", 0),
                    pin_memory=cfg.get("pin_memory", False),
                    drop_last=cfg.get("drop_last", False),
                    shuffle=False,
                )
            )

        return dataloaders

    def validation_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')

    def eval_epoch_end(self, outputs, mode):
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        loss_list = []
        bleu_score_list = []
        for dataloader_idx, output in enumerate(outputs):
            averaged_loss = average_losses_across_data_parallel_group([x['loss'] for x in output])
            inputs = list(itertools.chain(*[x['inputs'] for x in output]))
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))
            assert len(translations) == len(inputs)
            assert len(translations) == len(ground_truths)

            # Gather translations and ground truths from all workers
            tr_gt_inp = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            # we also need to drop pairs where ground truth is an empty string
            torch.distributed.all_gather_object(
                tr_gt_inp,
                [(t, g, i) for (t, g, i) in zip(translations, ground_truths, inputs)],
                group=parallel_state.get_data_parallel_group(),
            )
            if parallel_state.get_data_parallel_rank() == 0:
                _translations = []
                _ground_truths = []
                _inputs = []

                # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
                gt_inp_set = set()
                for rank in range(0, parallel_state.get_data_parallel_world_size()):
                    for t, g, i in tr_gt_inp[rank]:
                        if g + i not in gt_inp_set:
                            gt_inp_set.add(g + i)
                            _translations.append(t)
                            _ground_truths.append(g)
                            _inputs.append(i)

                if self.tgt_language in ['ja']:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="ja-mecab")
                elif self.tgt_language in ['zh']:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="zh")
                else:
                    sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")

                bleu_score = sacre_bleu.score * parallel_state.get_data_parallel_world_size()

                dataset_name = "Validation" if mode == 'val' else "Test"
                logging.info(f"{dataset_name}, Dataloader index: {dataloader_idx}, Set size: {len(_translations)}")
                logging.info(
                    f"{dataset_name}, Dataloader index: {dataloader_idx}, SacreBLEU = {bleu_score / parallel_state.get_data_parallel_world_size()}"
                )
                logging.info(f"{dataset_name}, Dataloader index: {dataloader_idx}, Translation Examples:")
                logging.info('============================================================')
                for example_idx in range(0, 3):
                    random_index = random.randint(0, len(_translations) - 1)
                    logging.info("    " + '\u0332'.join(f"Example {example_idx}:"))
                    logging.info(f"    Input:        {_inputs[random_index]}")
                    logging.info(f"    Prediction:   {_translations[random_index]}")
                    logging.info(f"    Ground Truth: {_ground_truths[random_index]}")
                    logging.info('============================================================')

            else:
                bleu_score = 0.0

            loss_list.append(averaged_loss[0].cpu().numpy())
            bleu_score_list.append(bleu_score)
            if dataloader_idx == 0:
                self.log(f'{mode}_sacreBLEU', bleu_score, sync_dist=True)
                self.log(f'{mode}_loss', averaged_loss[0], prog_bar=True)
                if self.multilingual:
                    self._log_multilingual_bleu_and_loss(dataloader_idx, bleu_score, averaged_loss[0], mode)
            else:
                if self.multilingual:
                    self._log_multilingual_bleu_and_loss(dataloader_idx, bleu_score, averaged_loss[0], mode)
                else:
                    self.log(f'{mode}_sacreBLEU_dl_index_{dataloader_idx}', bleu_score, sync_dist=True)
                    self.log(f'{mode}_loss_dl_index_{dataloader_idx}', averaged_loss[0], prog_bar=False)

        if len(loss_list) > 1:
            self.log(f"{mode}_loss_avg", np.mean(loss_list), sync_dist=True)
            self.log(f"{mode}_sacreBLEU_avg", np.mean(bleu_score_list), sync_dist=True)

    def _log_multilingual_bleu_and_loss(self, dataloader_idx, bleu_score, loss, mode):
        """
        Function to log multilingual BLEU scores with the right source-target language string instead of just the dataloader idx.
        """
        # Check if one-many or many-one and log with lang ids instead of dataloader_idx
        reverse_lang_direction = self._cfg.train_ds.reverse_lang_direction
        if isinstance(self.src_language, ListConfig):
            translation_lang_string = (
                f'{self.src_language[dataloader_idx]}-{self.tgt_language}'
                if not reverse_lang_direction
                else f'{self.tgt_language}-{self.src_language[dataloader_idx]}'
            )
            self.log(f'{mode}_sacreBLEU_{translation_lang_string}', bleu_score, sync_dist=True)
            self.log(f'{mode}_loss_{translation_lang_string}', loss, sync_dist=True)
        else:
            translation_lang_string = (
                f'{self.src_language}-{self.tgt_language[dataloader_idx]}'
                if not reverse_lang_direction
                else f'{self.tgt_language[dataloader_idx]}-{self.src_language}'
            )
            self.log(f'{mode}_sacreBLEU_{translation_lang_string}', bleu_score, sync_dist=True)
            self.log(f'{mode}_loss_{translation_lang_string}', loss, sync_dist=True)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if hasattr(self, '_validation_ds'):
            self._validation_dl = self._setup_eval_dataloader_from_config(
                cfg=val_data_config, dataset=self._validation_ds
            )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if hasattr(self, '_test_ds'):
            self._test_dl = self._setup_eval_dataloader_from_config(cfg=test_data_config, dataset=self._test_ds)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        # TODO: Figure out how to set global rank and world size for model parallel.
        if hasattr(self, '_train_ds'):
            self._train_dl = MTEncDecModel._setup_dataloader_from_config(cfg=train_data_config, dataset=self._train_ds)

    def process_batch(self, batch):
        """Override parent process_batch since TranslationDataset does not return dictionaries."""
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        batch = {
            'text_enc': src_ids,
            'text_dec': tgt_ids,
            'labels': labels,
            'enc_mask': src_mask.long(),  # super().process_batch() expects torch.int64
            'dec_mask': tgt_mask.long(),  # super().process_batch() expects torch.int64
            'loss_mask': tgt_mask.long(),  # super().process_batch() expects torch.int64
        }
        return super().process_batch(batch)

    def build_train_valid_test_datasets(self):
        self._train_ds = MTEncDecModel._setup_dataset_from_config(
            cfg=self._cfg.train_ds,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            global_rank=parallel_state.get_data_parallel_rank(),
            world_size=parallel_state.get_data_parallel_world_size(),
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
        )
        self._validation_ds = MTEncDecModel._setup_eval_dataset_from_config(
            cfg=self._cfg.validation_ds,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
        )
        # Test data config is optional.
        if hasattr(self._cfg, 'test_ds'):
            self._test_ds = MTEncDecModel._setup_eval_dataset_from_config(
                cfg=self._cfg.validation_ds,
                multilingual=self.multilingual,
                multilingual_ids=self.multilingual_ids,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
            )

    def list_available_models(self):
        pass
