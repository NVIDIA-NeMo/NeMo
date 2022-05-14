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

"""RETRO Style dataset."""

import torch

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import get_train_valid_test_split_
from nemo.utils import logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class MockRETRODataset(torch.utils.data.Dataset):
    def __init__(self, cfg, trainer, tokenizer, name, size):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        self._cfg = cfg
        self.size = size
        seed_val = parallel_state.get_data_parallel_rank() * 131 + 97
        torch.manual_seed(seed_val)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        vocab_size = self.tokenizer.vocab_size

        neighbors = self._cfg.data.neighbors
        input_length = self._cfg.data.seq_length
        chunks = input_length // self._cfg.chunk_size
        chunk_size = self._cfg.chunk_size
        pad_id = self.tokenizer.pad_id

        all_tokens = torch.randint(0, vocab_size, (input_length + 1,))
        # make sure the eod happens at the end of each chunk, can add paddings to it
        # e.g. [..., id, id, pad, pad, pad, eod]  each has chunk_size, each sentence
        # has length of multiple of chunk_size
        hidden = all_tokens[:-1]
        labels = all_tokens[1:]

        hidden_mask = hidden != pad_id
        # to mask out the token ids [id, id,  eod, id, pad, eod, id, id]
        # so attention is not across eod, mask should be:
        # [false, true,  true, true,  true, true,  true,  true]
        # [false, false, true, true,  true, true,  true,  true]
        # [false, false, false,true,  true, true,  true,  true]
        # [true,  true,  true, false, true, true,  true,  true]
        # [true,  true,  true, true,  true, true,  true,  true]
        # [true,  true,  true, false, true, false, true,  true]
        # [true,  true,  true, true,  true, true,  false, true]
        # [true,  true,  true, true,  true, true,  false, false]
        retrieved = torch.randint(0, vocab_size, (chunks, neighbors, 2 * chunk_size))

        context_mask = retrieved != pad_id

        return {
            'tokens': hidden,
            'labels': labels,
            'tokens_mask': hidden_mask,
            'loss_mask': hidden_mask,
            'retrieved_emb_mask': context_mask,
            'retrieved_ids': retrieved,
        }


def build_mock_train_valid_test_datasets(
    cfg, trainer, splits_string, tokenizer, mock_data_size,
):
    """Build train, valid, and test datasets."""

    splits = get_train_valid_test_split_(splits_string, mock_data_size)

    # Print stats about the splits.
    logging.info(' > dataset split:')

    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            dataset = MockRETRODataset(cfg, trainer, tokenizer, name, splits[index + 1] - splits[index],)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)
