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

import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset


class T5LMAdaptedDataset(GPTDataset):
    """
    Dataset for unlearning span corruption (https://arxiv.org/abs/2104.08691) in T5 models.
    Corresponds to the prefix-LM objective in the T5 paper (Table 3 in https://arxiv.org/abs/1910.10683).
    """

    def __init__(
        self, cfg, trainer, tokenizer, name, data_prefix, documents, indexed_dataset, num_samples, seed, **kwargs
    ):
        self.seq_length_encoder = cfg.data.seq_length
        self.seq_length_decoder = cfg.data.seq_length_dec
        self.tokenizer = tokenizer
        super().__init__(
            cfg,
            trainer,
            tokenizer,
            name,
            data_prefix,
            documents,
            indexed_dataset,
            num_samples,
            self.seq_length_encoder + self.seq_length_decoder,
            seed,
        )

    def __getitem__(self, idx):
        text = super()._get_text(idx)

        # Split text sequence into encoder and decoder inputs
        tokens_enc = text[: self.seq_length_encoder]

        # NOTE: Add bos only and not eos because the model will always generate till max seq length.
        tokens_dec = np.concatenate(([self.tokenizer.bos_id], text[self.seq_length_encoder :]))

        # Shift sequences for teacher forcing
        tokens_dec_in = tokens_dec[:-1]
        labels = tokens_dec[1:]

        # Create attention masks
        enc_mask = (tokens_enc != self.tokenizer.pad_id).astype(np.int64)
        dec_mask = (tokens_dec_in != self.tokenizer.pad_id).astype(np.int64)

        loss_mask = dec_mask

        train_sample = {
            'text_enc': tokens_enc,
            'text_dec': tokens_dec_in,
            'labels': labels,
            'loss_mask': loss_mask,
            'truncated': False,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
        }
        return train_sample
