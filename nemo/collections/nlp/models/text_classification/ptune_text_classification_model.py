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
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.nlp.data.text_classification.ptune_text_classification_dataset import (
    PTuneTextClassificationDataset,
    token_wrapper,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LossType, NeuralType, PredictionsType, StringLabel, StringType
from nemo.utils import logging
from nemo.utils.app_state import AppState

__all__ = ['PTuneTextClassificationModel']


SMALL_LOGITS = -100


class PTuneTextClassificationModel(NLPModel, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"sentences": [NeuralType(('T'), StringType())], "labels": [NeuralType(('T'), StringLabel())]}

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "floss": NeuralType((), LossType()),
            "returned_pred": NeuralType(('B'), PredictionsType()),
            "returned_label": NeuralType(('B'), PredictionsType()),
        }

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the PTune TextClassifier model."""
        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=True)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=cfg.get('seed', 1234),
        )

        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset
        # tokenizer needs to get initialized before the super.__init__()
        # as dataloaders and datasets need it to process the data
        self.tokenizer = get_nmt_tokenizer(
            library=cfg.tokenizer.library,
            model_name=cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merges_file", cfg.tokenizer.merge_file),
        )

        self.class_weights = None

        self.model = MegatronGPTModel.restore_from(
            self.register_artifact('language_model.nemo_file', cfg.language_model.get('nemo_file', None)),
            trainer=trainer,
        )

        if not cfg.use_lm_finetune:
            self.model.freeze()

        hidden_size = self.model.cfg.hidden_size

        # register the file containing the labels into the artifacts to get stored in the '.nemo' file later
        self.classes = cfg.dataset.classes

        self.embeddings = self.model.model.language_model.embedding.word_embeddings

        # set allowed vocab set
        self.vocab = self.tokenizer.tokenizer.get_vocab()

        # make sure classes are part of the vocab
        for k in cfg.dataset.classes:
            if token_wrapper(k) not in self.vocab:
                logging.error(f'class {k} is not part of the vocabulary. Please add it to your vocab')
        self.allowed_vocab_ids = set(self.vocab[token_wrapper(k)] for k in cfg.dataset.classes)

        # map from id to label
        self.allowed_vocab = {}
        self.label_ids = {}
        self.id_to_label = {}
        for i, k in enumerate(cfg.dataset.classes):
            self.allowed_vocab[self.vocab[token_wrapper(k)]] = i
            self.label_ids[k] = i
            self.id_to_label[i] = k

        self.template = cfg.prompt_encoder.template

        self.prompt_encoder = PromptEncoder(
            template=cfg.prompt_encoder.template,
            hidden_size=hidden_size,
            lstm_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
        )

        # load prompt encoder
        self.hidden_size = hidden_size
        self.tokenizer.add_special_tokens({'additional_special_tokens': [cfg.pseudo_token]})

        self.pseudo_token_id = self.tokenizer.tokenizer.get_vocab()[cfg.pseudo_token]
        self.pad_token_id = (
            self.tokenizer.tokenizer.pad_token_id
            if self.tokenizer.tokenizer.pad_token_id is not None
            else self.tokenizer.tokenizer.unk_token_id
        )
        self.spell_length = sum(self.template)

    def setup(self, stage):
        # setup to track metrics, need to put here
        # as data_parallel_group is initialized when calling `fit, or test function`
        app = AppState()
        self.classification_report = ClassificationReport(
            num_classes=len(self.classes),
            label_ids=self.label_ids,
            mode='micro',
            dist_sync_on_step=True,
            process_group=app.data_parallel_group,
        )

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()

        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pad_token_id
        raw_embeds = self.embeddings(queries_for_embedding)
        dtype = self.model.model.language_model.encoder.layers[0].dtype
        if dtype == torch.float32:
            replace_embeds = self.prompt_encoder(enc_taskname=None)
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                replace_embeds = self.prompt_encoder(enc_taskname=None)

        blocked_indices = queries == self.pseudo_token_id
        raw_embeds = raw_embeds.clone().type(dtype)
        # find the index to the psedo-tokens
        index = blocked_indices.nonzero().reshape((bz, -1, 2))[:, :, 1][:, :, None]

        _, seq, _ = index.shape
        _, _, emb = raw_embeds.shape
        index = index.expand(bz, seq, emb)

        _, replace_seq, replace_emb = replace_embeds.shape
        replace_embeds = replace_embeds.expand(bz, replace_seq, replace_emb)
        # scatter the psedo-token embeddings to the raw embeddings
        raw_embeds.scatter_(1, index, replace_embeds)
        # slow version of above scatter logics
        # for bidx in range(bz):
        #     position = blocked_indices[bidx].nonzero()[:, 0]
        #     for i in range(len(position)):
        #         raw_embeds[bidx, position[i], :] = replace_embeds[bidx, i, :]

        return raw_embeds

    def get_query(self, x_h, prompt_tokens, x_t=None):
        max_seq_len = self.model._cfg.encoder_seq_length
        input_token_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenizer.tokenize(' ' + x_h))
        cut = 0
        if len(input_token_ids) + sum(self.template) > max_seq_len:
            logging.warning("Input sequence is longer than the LM model max seq, will cut it off to fit")
            cut = len(input_token_ids) + sum(self.template) - max_seq_len
        return [
            prompt_tokens * self.template[0]
            + input_token_ids[cut:]  # head entity
            + prompt_tokens * self.template[1]
            + (
                self.tokenizer.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_t))
                if x_t is not None
                else []
            )
        ]

    def get_ground_truth_labels(self, batch_size, label_ids):
        returned_label = []
        for i in range(batch_size):
            returned_label.append(self.allowed_vocab[label_ids[i, 0].item()])
        return torch.tensor(returned_label).to(self.device)

    def get_prediction(self, batch_size, label_position, logits):
        top10 = []
        returned_pred = []
        for i in range(batch_size):
            array = []
            for allowed_id in self.allowed_vocab_ids:
                pred_p = logits[i, label_position[i, 0], allowed_id]
                array.append((pred_p, allowed_id))
            sorted_array = sorted(array, key=lambda x: x[0], reverse=True)
            pred = sorted_array[0][1]
            returned_pred.append(self.allowed_vocab[pred])
        return top10, torch.tensor(returned_pred).to(self.device)

    def get_encoder_input(self, sentences):
        batch_size = len(sentences)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]

        queries = [torch.LongTensor(self.get_query(sentences[i], prompt_tokens)).squeeze(0) for i in range(batch_size)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        # attention_mask indicates the boundary of attention
        attention_mask = queries != self.pad_token_id
        # get embedded input
        inputs_embeds = self.embed_input(queries)

        bz, seq_len, _ = inputs_embeds.shape

        # get the GPT causal mask
        causal_mask = torch.tril(torch.ones((bz, seq_len, seq_len), device=self.device)).view(bz, 1, seq_len, seq_len)
        # combine the attention_mask and causal_mask
        r = causal_mask.permute((1, 2, 0, 3)) * attention_mask.int()
        new_atten = r.permute((2, 0, 1, 3))
        # convert it to the boolean
        new_atten = new_atten < 0.5

        # calculate the position embedding based on the seq_len
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs_embeds[:, :, 0])
        position_embeddings = self.model.model.language_model.embedding.position_embeddings(position_ids)

        # get the final input for encoder
        encoder_input = inputs_embeds + position_embeddings

        # calculate the position of the output token
        label_position = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1)
        return encoder_input, new_atten, label_position

    def get_label_input(self, labels, label_position, seq_len):
        batch_size, _ = label_position.shape
        x_ts = [token_wrapper(x_t) for x_t in labels]

        # construct label ids
        label_ids = (
            torch.LongTensor(self.tokenizer.tokenizer.convert_tokens_to_ids(x_ts))
            .reshape((batch_size, -1))
            .to(self.device)
        )
        labels = torch.zeros(batch_size, seq_len).to(self.device).fill_(SMALL_LOGITS).long()  # bz * seq_len
        labels = labels.scatter_(1, label_position, label_ids)
        return labels, label_ids

    def forward_eval(self, sentences):
        encoder_input, new_atten, label_position = self.get_encoder_input(sentences)
        batch_size, _, seq_len, _ = new_atten.shape

        # workaround to do auto-cast
        # get the LM dtype
        dtype = self.model.model.language_model.encoder.layers[0].dtype

        if dtype == torch.float32:
            output = self.model.model(
                None, None, encoder_input=encoder_input.to(self.device), attention_mask=new_atten.to(self.device)
            )
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = self.model.model(
                    None, None, encoder_input=encoder_input.to(self.device), attention_mask=new_atten.to(self.device)
                )
        logits = output

        _, returned_pred = self.get_prediction(batch_size, label_position.to(self.device), logits)
        return returned_pred

    @typecheck()
    def forward(self, sentences, labels):
        encoder_input, new_atten, label_position = self.get_encoder_input(sentences)
        batch_size, _, seq_len, _ = new_atten.shape
        labels_input, label_ids = self.get_label_input(labels, label_position, seq_len)
        # workaround to do auto-cast
        # get the LM dtype
        dtype = self.model.model.language_model.encoder.layers[0].dtype

        if dtype == torch.float32:
            output = self.model.model(
                None, None, encoder_input=encoder_input, attention_mask=new_atten, labels=labels_input
            )
        else:
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = self.model.model(
                    None, None, encoder_input=encoder_input, attention_mask=new_atten, labels=labels_input
                )
        loss, logits = output
        floss = (loss[(labels_input != SMALL_LOGITS)]).mean()

        _, returned_pred = self.get_prediction(batch_size, label_position, logits)
        returned_label = self.get_ground_truth_labels(batch_size, label_ids)
        return floss, returned_pred, returned_label

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        sentences, labels = batch
        train_loss, _, _ = self.forward(sentences=sentences, labels=labels)

        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': train_loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        sentences, labels = batch
        val_loss, preds, gt_labels = self.forward(sentences=sentences, labels=labels)

        hit = 0
        for pred, gt_label in zip(preds, gt_labels):
            if pred == gt_label:
                hit += 1

        tp, fn, fp, _ = self.classification_report(preds, gt_labels)

        return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp, 'hit': hit}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return {}
        if self.trainer.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        avg_loss = torch.stack([x[f'val_loss'] for x in outputs]).mean()

        total_hit = sum([x[f'hit'] for x in outputs])
        # calculate metrics and classification report
        precision, recall, f1, report = self.classification_report.compute()

        total_data = torch.sum(self.classification_report.num_examples_per_class)
        accuracy = total_hit / total_data.item()
        logging.info(f'{prefix}_report: {report}')
        logging.info(f'{total_hit} correct out of {total_data}, accuracy: {accuracy*100:.2f}')
        self.log(f'{prefix}_loss', avg_loss, prog_bar=True)
        self.log(f'{prefix}_accuracy', accuracy)
        self.log(f'{prefix}_precision', precision)
        self.log(f'{prefix}_f1', f1)
        self.log(f'{prefix}_recall', recall)

        self.classification_report.reset()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or not test_data_config.file_path:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: Dict) -> 'torch.utils.data.DataLoader':
        input_file = cfg.file_path
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.file_path" and "train_ds.file_path" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        dataset = PTuneTextClassificationDataset(input_file)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=cfg.get("drop_last", False),
            collate_fn=dataset.collate_fn,
        )

    @torch.no_grad()
    def classifytext(self, queries: List[str], batch_size: int = 1, prompt: str = 'Sentiment') -> List[int]:
        """
        Get prediction for the queries
        Args:
            queries: text sequences
            batch_size: batch size to use during inference
            prompt: the prompt string appended at the end of your input sentence
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
            infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, queries, prompt)
            for i, batch in enumerate(infer_datalayer):
                sentences, _ = batch
                preds = self.forward_eval(sentences)
                all_preds.extend([self.id_to_label[i.item()] for i in preds])
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            logging.set_verbosity(logging_level)
        return all_preds

    def _setup_infer_dataloader(self, cfg: Dict, queries: List[str], prompt: str) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            queries: text
            prompt: the prompt string appended at the end of your input sentence
        Returns:
            A pytorch DataLoader.
        """
        dataset = PTuneTextClassificationDataset(None, queries, prompt)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
