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

from typing import Dict, List

import torch
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.nlp.data.glue_benchmark.t5_ptune_dataset import T5PTuneDataset, T5PTuneInferenceDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.collections.nlp.modules.common.t5_prompt_encoder import PromptEncoder
from nemo.utils import logging

try:
    from apex.transformer import tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronT5PTuneModel']


class MegatronT5PTuneModel(MegatronBaseModel):
    """
    Megatron T5 P-Tune
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        raise Exception("Please use the NeMo r1.8.0 branch for T5 PTuning.")

        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)
        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(
            self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
            trainer=trainer,
            return_config=True,
        )
        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_o2
            # TODO, need to fix this later
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = 4
            t5_cfg.global_batch_size = 4

        self.model = MegatronT5Model.restore_from(
            self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
            trainer=trainer,
            override_config_path=t5_cfg,
        )

        # self.model = MegatronT5Model.restore_from(
        #     self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
        #     trainer=trainer)

        self.tokenizer = self.model.tokenizer

        self.float_type = self.model.enc_dec_model.enc_dec_model.encoder.model.layers[0].dtype

        if not cfg.use_lm_finetune:
            self.model.freeze()

        hidden_size = self.model.cfg.hidden_size

        # register the file containing the labels into the artifacts to get stored in the '.nemo' file later
        self.word_embeddings = self.model.enc_dec_model.encoder_embedding.word_embeddings
        self.position_embeddings = self.model.enc_dec_model.encoder_embedding.position_embeddings

        # self.vocab = self.tokenizer.tokenizer.get_vocab()

        self.template = cfg.prompt_encoder.template

        self.prompt_encoder = PromptEncoder(
            template=cfg.prompt_encoder.template,
            hidden_size=hidden_size,
            lstm_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
        )

        # load prompt encoder
        self.hidden_size = hidden_size
        self.tokenizer.add_special_tokens([cfg.pseudo_token])

        self.pseudo_token_id = self.tokenizer.special_token_to_id[cfg.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id
        self.spell_length = sum(self.template)
        self._reduced_loss_buffer = []
        self.decoder_seq_length = cfg.get('decoder_seq_length', 10)

    def embed_input(self, enc_input_id: Tensor, enc_taskname_id: Tensor):
        """
        This method will replace the virtual tokens in the enc_input_id with
        embeddings calculated from `prompt_encoder`. If the `enc_taskname_id` is
        not None, the computed virtual token embeddings are depenedent on it.
        The virtual token placeholders has the token_id `self.pseudo_token_id`.
        params:
            enc_input_id: the input token ids
            enc_taskname_id: the NLP task tag token ids
        returns:
            the token embedding for the LM model.
        """
        bz = enc_input_id.shape[0]
        queries_for_embedding = enc_input_id.clone()

        queries_for_embedding[(enc_input_id == self.pseudo_token_id)] = self.pad_token_id
        raw_embeds = self.word_embeddings(queries_for_embedding).clone()

        if self.cfg.prompt_encoder.task_dependent:
            enc_taskname = self.word_embeddings(enc_taskname_id)
        else:
            enc_taskname = None

        if self.float_type == torch.float32:
            replace_embeds = self.prompt_encoder(enc_taskname=enc_taskname)
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                replace_embeds = self.prompt_encoder(enc_taskname=enc_taskname)

        blocked_indices = enc_input_id == self.pseudo_token_id
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
        return raw_embeds

    def process_batch(self, batch):
        """Build the batch."""

        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_taskname']
        datatype = torch.int64
        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']
        enc_taskname = data_b['enc_taskname']

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_taskname

    def get_loss(self, batch):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_taskname = self.process_batch(batch)
        input_embeds = self.embed_input(tokens_enc, enc_taskname)

        encoder_position_ids = build_position_ids(tokens_enc)

        position_embeddings = self.position_embeddings(encoder_position_ids)

        encoder_input = input_embeds + position_embeddings

        if self.float_type == torch.float32:
            output = self.model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=tokens_dec,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                enc_hidden_states=None,
                output_enc_hidden_only=False,
                enc_input=encoder_input,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                output = self.model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=tokens_dec,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    enc_hidden_states=None,
                    output_enc_hidden_only=False,
                    enc_input=encoder_input,
                )

        tokens_loss = output

        loss = self.model.loss_func(loss_mask, tokens_loss)
        self.log('train_loss', loss)

        return loss, tokens_enc, labels, enc_mask, encoder_input

    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self.get_loss(batch)
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
        loss, tokens_enc, labels, enc_mask, encoder_input = self.get_loss(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
        )

        return {'loss': loss, 'predicted_token_ids': predicted_token_ids, 'labels': labels}

    def inference_epoch_end(self, outputs):
        losses = [x['loss'] for x in outputs]
        averaged_loss = average_losses_across_data_parallel_group(losses)
        all_preds = []
        all_labels = []
        for item in outputs:
            preds = item['predicted_token_ids'].cpu().numpy().tolist()
            labels = item['labels'].cpu().numpy().tolist()
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if self.tokenizer.eos_id in pred:
                    idx = pred.index(self.tokenizer.eos_id)
                    pred = pred[:idx]
                pred = [id for id in pred if id not in self.tokenizer.special_token_to_id.values()]
                label = [id for id in label if id not in self.tokenizer.special_token_to_id.values()]
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
        logging.info('Building GLUE datasets.')
        self._test_ds = T5PTuneDataset(
            self.cfg.data.test_ds.file_path,
            data_type="test",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.cfg.data.test_ds.max_seq_length,
            max_seq_length_decoder=None,
        )
        if test_only:
            return None, None, self._test_ds
        self._train_ds = T5PTuneDataset(
            self.cfg.data.train_ds.file_path,
            data_type="train",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.cfg.data.train_ds.max_seq_length,
            max_seq_length_decoder=None,
        )
        self._validation_ds = T5PTuneDataset(
            self.cfg.data.validation_ds.file_path,
            data_type="validation",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.cfg.data.validation_ds.max_seq_length,
            max_seq_length_decoder=None,
        )
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

    def setup_training_data(self, training_data_config=None):
        self._train_dl = self.build_pretraining_data_loader(
            self._train_ds,
            self.cfg.data.train_ds.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.train_ds.num_workers,
            pin_memory=True,
        )

    def setup_validation_data(self, validation_data_config=None):
        self._validation_dl = self.build_pretraining_data_loader(
            self._validation_ds,
            self.cfg.data.validation_ds.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.validation_ds.num_workers,
            pin_memory=True,
        )

    def setup_test_data(self, test_data_config=None):
        self._test_dl = self.build_pretraining_data_loader(
            self._test_ds,
            self.cfg.data.test_ds.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.test_ds.num_workers,
            pin_memory=True,
        )

    @classmethod
    def list_available_models(cls):
        pass

    @torch.no_grad()
    def ptune_inference(self, queries: List[Dict], batch_size: int = 1, decode_token_len: int = None) -> List[str]:
        """
        Get prediction for the queries
        Args:
            queries: List of data samples without labels
            batch_size: batch size to use during inference
            decode_token_len: max number of tokens to generate during inference
        Returns:
            all_preds: model predictions
        """
        if decode_token_len is None:
            decode_token_len = self.decoder_seq_length
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
                tokens_enc = batch['text_enc'].to(self.device)
                enc_taskname = batch['enc_taskname'].to(self.device)
                enc_mask = batch['enc_mask'].to(self.device)

                input_embeds = self.embed_input(tokens_enc, enc_taskname)

                encoder_position_ids = build_position_ids(tokens_enc)

                position_embeddings = self.position_embeddings(encoder_position_ids)

                encoder_input = input_embeds + position_embeddings

                # loss, tokens_enc, labels, enc_mask, encoder_input = self.get_loss(batch)
                if self.float_type == torch.float32:
                    predicted_token_ids, _ = self.model.decode(
                        tokens_enc=tokens_enc,
                        enc_mask=enc_mask,
                        num_tokens_to_generate=decode_token_len,
                        enc_input=encoder_input,
                    )
                else:
                    with torch.autocast(device_type="cuda", dtype=self.float_type):
                        predicted_token_ids, _ = self.model.decode(
                            tokens_enc=tokens_enc,
                            enc_mask=enc_mask,
                            num_tokens_to_generate=decode_token_len,
                            enc_input=encoder_input,
                        )

                preds = predicted_token_ids.cpu().numpy().tolist()
                for i, pred in enumerate(preds):
                    if self.tokenizer.eos_id in pred:
                        idx = pred.index(self.tokenizer.eos_id)
                        pred = pred[:idx]
                    pred = [id for id in pred if id not in self.tokenizer.special_token_to_id.values()]
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
        dataset = T5PTuneInferenceDataset(
            queries=queries,
            data_type="test",
            tokenizer=self.tokenizer,
            templates=self.template,
            pseudo_token_id=self.pseudo_token_id,
            pad_id=self.pad_token_id,
            max_seq_length=self.cfg.data.test_ds.max_seq_length,
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
