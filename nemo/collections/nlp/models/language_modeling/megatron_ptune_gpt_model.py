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
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.glue_benchmark.gpt_ptune_dataset import GPTPTuneDataset, GPTPTuneInferenceDataset
from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import (
    make_attention_mask_3d,
    make_history_mask_3d,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder
from nemo.utils import logging

try:
    from apex.transformer import tensor_parallel

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


__all__ = ['MegatronGPTPTuneModel']


class MegatronGPTPTuneModel(NLPModel):
    """
    Megatron GPT P-Tune
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=cfg.get('seed', 1234),
        )

        self.model = MegatronGPTModel.restore_from(
            self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
            trainer=trainer,
        )

        self.tokenizer = self.model.tokenizer
        self.float_type = self.model.model.language_model.encoder.layers[0].dtype

        if not cfg.use_lm_finetune:
            self.model.freeze()

        hidden_size = self.model.cfg.hidden_size

        self.embeddings = self.model.model.language_model.embedding.word_embeddings

        self.template = cfg.prompt_encoder.template

        self.prompt_encoder = PromptEncoder(
            template=cfg.prompt_encoder.template,
            hidden_size=hidden_size,
            lstm_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
        )

        self._reduced_loss_buffer = []

        # load prompt encoder
        self.hidden_size = hidden_size
        self.tokenizer.add_special_tokens({'additional_special_tokens': [cfg.pseudo_token]})

        self.pseudo_token_id = self.tokenizer.token_to_id(cfg.pseudo_token)
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id
        self.spell_length = sum(self.template)
        self.special_tokens = set(
            [
                self.tokenizer.eos_id,
                self.tokenizer.pad_id,
                self.tokenizer.sep_id,
                self.tokenizer.unk_id,
                self.tokenizer.bos_id,
                self.tokenizer.cls_id,
            ]
        )

    def embed_input(self, queries, enc_taskname):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()

        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pad_token_id

        raw_embeds = self.embeddings(queries_for_embedding).clone()
        if self.cfg.prompt_encoder.task_dependent:
            enc_taskname = self.embeddings(enc_taskname)
        else:
            enc_taskname = None

        if self.float_type == torch.float32:
            replace_embeds = self.prompt_encoder(enc_taskname=enc_taskname)
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                replace_embeds = self.prompt_encoder(enc_taskname=enc_taskname)

        blocked_indices = queries == self.pseudo_token_id
        raw_embeds = raw_embeds.clone().type(self.float_type)
        # find the index to the psedo-tokens
        index = blocked_indices.nonzero().reshape((bz, -1, 2))[:, :, 1][:, :, None]

        _, seq, _ = index.shape
        _, _, emb = raw_embeds.shape
        index = index.expand(bz, seq, emb)

        if enc_taskname is None:
            # taskname none, encoder returens batch 1
            # need to expand
            _, replace_seq, _ = replace_embeds.shape
            replace_embeds = replace_embeds.expand(bz, replace_seq, emb)

        # scatter the psedo-token embeddings to the raw embeddings
        raw_embeds.scatter_(1, index, replace_embeds)
        # slow version of above scatter logics
        # for bidx in range(bz):
        #     position = blocked_indices[bidx].nonzero()[:, 0]
        #     for i in range(len(position)):
        #         raw_embeds[bidx, position[i], :] = replace_embeds[bidx, i, :]

        return raw_embeds

    def get_loss(self, batch):
        enc_input = batch['enc_input']
        enc_taskname = batch['enc_taskname']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        enc_query = batch['enc_query']
        input_attn_mask = batch['input_attn_mask']

        input_attn_mask = input_attn_mask.unsqueeze(1) < 0.5

        input_embeds = self.embed_input(enc_input, enc_taskname)

        encoder_position_ids = build_position_ids(enc_input)

        position_embeddings = self.model.model.language_model.embedding.position_embeddings(encoder_position_ids)

        encoder_input = input_embeds + position_embeddings

        if self.float_type == torch.float32:
            output = self.model.model(
                None, None, encoder_input=encoder_input, attention_mask=input_attn_mask, labels=labels,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                output = self.model.model(
                    None, None, encoder_input=encoder_input, attention_mask=input_attn_mask, labels=labels,
                )
        output_tensor, encoder_hidden_states = output
        loss = self.loss_func(loss_mask, output_tensor)
        return loss

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss)
        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def inference_step(self, batch, batch_ix):
        loss = self.get_loss(batch)
        enc_query = batch['enc_query']
        enc_taskname = batch['enc_taskname']
        labels = batch['labels']
        label_position = batch['label_position']
        # loss, tokens_enc, labels, enc_mask, encoder_input = self.get_loss(batch)
        predicted_token_ids, log_probs = self.decode(
            enc_query=enc_query,
            enc_taskname=enc_taskname,
            label_position=label_position,
            num_tokens_to_generate=self.num_tokens_to_gen,
        )

        return {
            'loss': loss,
            'predicted_token_ids': predicted_token_ids,
            'labels': labels,
            'label_position': label_position,
        }

    def decode(self, enc_query, enc_taskname, label_position, num_tokens_to_generate):
        with torch.no_grad():
            predicted_tokens_dec = enc_query

            label_start = label_position[:, 0].clone()

            for _ in range(num_tokens_to_generate):
                attn_mask = make_attention_mask_3d(predicted_tokens_dec, predicted_tokens_dec, self.pad_token_id)
                attn_mask = attn_mask * make_history_mask_3d(predicted_tokens_dec)

                attn_mask = attn_mask < 0.5

                attn_mask = attn_mask.unsqueeze(1)

                input_embeds = self.embed_input(predicted_tokens_dec, enc_taskname)

                encoder_position_ids = build_position_ids(predicted_tokens_dec)
                position_embeddings = self.model.model.language_model.embedding.position_embeddings(
                    encoder_position_ids
                )

                encoder_input = input_embeds + position_embeddings

                if self.float_type == torch.float32:
                    output = self.model.model(None, None, encoder_input=encoder_input, attention_mask=attn_mask,)
                else:
                    with torch.autocast(device_type="cuda", dtype=self.float_type):
                        output = self.model.model(None, None, encoder_input=encoder_input, attention_mask=attn_mask,)
                output_tensor = output

                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)

                # TODO, add logic to use the allowed labels if it is defined
                log_probs, token_ids = torch.max(nn.functional.log_softmax(output_tensor, dim=-1), dim=-1)

                new_pred = torch.full_like(token_ids[:, 0:1], self.pad_token_id)
                predicted_tokens_dec = torch.cat([predicted_tokens_dec, new_pred], 1)

                predicted = torch.gather(token_ids, 1, label_start.view(-1, 1))

                # need to scatter the token id at the right position
                label_start += 1
                predicted_tokens_dec.scatter_(1, label_start.view(-1, 1), predicted)

        return predicted_tokens_dec, log_probs

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()
            label_positions = item['label_position'].cpu().numpy().tolist()
            for i, (pred, label, label_position) in enumerate(zip(preds, labels, label_positions)):
                start_position = label_position[0] + 1
                pred = pred[start_position:]
                if self.tokenizer.eos_id in pred:
                    idx = pred.index(self.tokenizer.eos_id)
                    pred = pred[:idx]
                pred = [id for id in pred if id not in self.special_tokens]
                label = [id for id in label[label_position[0] : label_position[1]] if id not in self.special_tokens]
                pred = self.tokenizer.ids_to_text(pred)
                label = self.tokenizer.ids_to_text(label)
                all_preds.append(pred)
                all_labels.append(label)

        correct = 0
        for pred, label in zip(all_preds, all_labels):
            if pred == label:
                correct += 1
        acc = correct / len(all_preds)
        return averaged_loss[0], acc

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        val_loss, val_acc = self.inference_epoch_end(outputs)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'Validation accuracy: {val_acc}')

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        test_loss, test_acc = self.inference_epoch_end(outputs)
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', test_acc, prog_bar=True)
        logging.info(f'Test loss: {test_loss}')
        logging.info(f'Test accuracy: {test_acc}')

    def build_train_valid_test_datasets(self, test_only=False):
        logging.info('Building P-Tune datasets.')
        self._test_ds = GPTPTuneDataset(
            self.cfg.data.test_ds.file_path,
            data_type="test",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
            max_seq_length_decoder=self.cfg.get('max_decode_length', None),
        )
        # update the num_tokens_to_gen from dataset
        length_for_test = self._test_ds.max_seq_length_decoder
        if test_only:
            self.num_tokens_to_gen = length_for_test
            return None, None, self._test_ds
        self._train_ds = GPTPTuneDataset(
            self.cfg.data.train_ds.file_path,
            data_type="train",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
            max_seq_length_decoder=length_for_test,
        )
        # update the num_tokens_to_gen from dataset
        length_for_train = self._train_ds.max_seq_length_decoder
        self._validation_ds = GPTPTuneDataset(
            self.cfg.data.validation_ds.file_path,
            data_type="validation",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
            max_seq_length_decoder=length_for_train,
        )
        length_for_validation = self._validation_ds.max_seq_length_decoder
        self.num_tokens_to_gen = min(length_for_validation, length_for_train)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T5 datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def setup(self, stage=None):
        if stage == 'predict':
            return
        self.build_train_valid_test_datasets(test_only=stage == 'test')
        self.setup_test_data()
        if stage == 'test':
            return
        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self):
        self._train_dl = self.build_pretraining_data_loader(
            self._train_ds,
            self.cfg.data.train_ds.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=True,
        )

    def setup_validation_data(self):
        self._validation_dl = self.build_pretraining_data_loader(
            self._validation_ds,
            self.cfg.data.validation_ds.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=True,
        )

    def setup_test_data(self):
        self._test_dl = self.build_pretraining_data_loader(
            self._test_ds,
            self.cfg.data.test_ds.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.test_ds.num_workers,
            pin_memory=True,
        )

    def list_available_models():
        pass

    @torch.no_grad()
    def ptune_inference(self, queries: List[Dict], batch_size: int = 1, decode_token_len: int = 5) -> List[str]:
        """
        Get prediction for the queries
        Args:
            queries: List of data samples without labels
            batch_size: batch size to use during inference
            decode_token_len: max number of tokens to generate during inference
        Returns:
            all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            # Switch model to evaluation mode
            self.eval()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            dataloader_cfg = {"batch_size": batch_size, "num_workers": 3, "pin_memory": False}
            infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, queries, decode_token_len)
            for i, batch in enumerate(infer_datalayer):
                enc_query = batch['enc_query'].to(self.device)
                label_position = batch['label_position'].to(self.device)
                enc_taskname = batch['enc_taskname'].to(self.device)
                # loss, tokens_enc, labels, enc_mask, encoder_input = self.get_loss(batch)
                predicted_token_ids, _ = self.decode(
                    enc_query=enc_query,
                    enc_taskname=enc_taskname,
                    label_position=label_position,
                    num_tokens_to_generate=self.num_tokens_to_gen,
                )
                preds = predicted_token_ids.cpu().numpy().tolist()
                label_positions = label_position.cpu().numpy().tolist()
                for i, (pred, label_position) in enumerate(zip(preds, label_positions)):
                    start_position = label_position[0] + 1
                    pred = pred[start_position:]
                    if self.tokenizer.eos_id in pred:
                        idx = pred.index(self.tokenizer.eos_id)
                        pred = pred[:idx]
                    pred = [id for id in pred if id not in self.special_tokens]
                    pred = self.tokenizer.ids_to_text(pred)
                    all_preds.append(pred)
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)
        return all_preds

    def _setup_infer_dataloader(
        self, cfg: Dict, queries: List[str], decode_token_len: int
    ) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            queries: queries object
        Returns:
            A pytorch DataLoader.
        """
        # dataset = PTuneTextClassificationDataset(None, queries, prompt)
        dataset = GPTPTuneInferenceDataset(
            queries=queries,
            data_type="test",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.model.cfg.encoder_seq_length,
            max_seq_length_decoder=decode_token_len,
        )

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=False,
        )
