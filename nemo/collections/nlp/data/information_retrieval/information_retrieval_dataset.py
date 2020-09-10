# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Pytorch Dataset for training information retrieval models."""

import multiprocessing as mp
import os
import pickle
import random
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = [
    "BertInformationRetrievalDatasetTrain",
    "BertInformationRetrievalDatasetEval",
    "BertDensePassageRetrievalDatasetInfer",
]


class BaseInformationRetrievalDataset(Dataset):
    """
    Base information retrieval dataset on which other datasets are built.

    Args:
        tokenizer: tokenizer
        max_query_length: maximum length of query in tokens
        max_passage_length: maximum length of passage in tokens
    """

    def __init__(
        tokenizer: TokenizerSpec, max_query_length: Optional[int] = 31, max_passage_length: Optional[int] = 190,
    ):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def parse_npz(self, file, max_seq_length):
        cached_collection = file + ".npz"
        if os.path.isfile(cached_collection):
            dataset_npz = np.load(cached_collection)["data"]
        else:
            dataset_dict = self.tokenize_dataset(file, max_seq_length)
            dataset_npz = np.zeros((len(dataset_dict), max_seq_length + 1))
            for key in dataset_dict:
                dataset_npz[key][0] = len(dataset_dict[key])
                dataset_npz[key][1 : len(dataset_dict[key]) + 1] = dataset_dict[key]
            np.savez(cached_collection, data=dataset_npz)
        return dataset_npz

    def parse_pkl(self, file, max_seq_length):
        cached_collection = file + ".pkl"
        if os.path.isfile(cached_collection):
            dataset_dict = pickle.load(open(cached_collection, "rb"))
        else:
            dataset_dict = self.tokenize_dataset(file, max_seq_length)
            pickle.dump(dataset_dict, open(cached_collection, "wb"))
        return dataset_dict

    def tokenize_dataset(self, file, max_seq_length):
        lines = open(file, "r").readlines()
        with mp.Pool() as pool:
            dataset_dict = pool.map(self.preprocess_line, lines)
        dataset_dict = {id_: tokens[:max_seq_length] for (id_, tokens) in dataset_dict}
        return dataset_dict

    def preprocess_line(self, line):
        id_, text = line.split("\t")
        token_ids = self.tokenizer.text_to_ids(text.strip())
        return int(id_), token_ids

    def construct_input(self, token_ids1, max_seq_length, token_ids2=None):
        input_ids = [self.tokenizer.pad_id] * max_seq_length
        bert_input = [self.tokenizer.cls_id] + token_ids1 + [self.tokenizer.sep_id]
        sentence1_length = len(bert_input)
        if token_ids2 is not None:
            bert_input = bert_input + token_ids2 + [self.tokenizer.sep_id]

        bert_input = bert_input[:max_seq_length]

        num_nonpad_tokens = len(bert_input)

        input_ids[:num_nonpad_tokens] = bert_input
        input_ids = np.array(input_ids, dtype=np.long)
        input_mask = input_ids != self.tokenizer.pad_id
        input_type_ids = np.ones_like(input_ids)
        input_type_ids[:sentence1_length] = 0

        return input_ids, input_mask, input_type_ids

    def preprocess_bert(self, query_id, psg_ids):
        """
        Transforms query id (Q) and a list of passages ids (P1, ..., Pk)
        into a tensor of size [k, max_length] with the following rows:
        [CLS] Q_text [SEP] Pi_text [SEP], i = 1, ..., k
        """

        max_seq_length = self.max_query_length + self.max_passage_length + 3
        input_ids, input_mask, input_type_ids = [], [], []
        for psg_id in psg_ids:
            inputs = self.construct_input(self.queries[query_id], max_seq_length, self._psgid2tokens(psg_id))
            input_ids.append(inputs[0])
            input_mask.append(inputs[1])
            input_type_ids.append(inputs[2])

        input_ids = np.stack(input_ids)
        input_mask = np.stack(input_mask)
        input_type_ids = np.stack(input_type_ids)

        return input_ids, input_mask, input_type_ids

    def preprocess_dpr(self, query_id, psg_ids):
        """
        Transforms query id (Q) and a list of passages ids (P1, ..., Pk)
        into two tensors of sizes [1, max_q_length] and [k, max_p_length]
        with the following rows:
        1) [CLS] Q_text [SEP]
        2) [CLS] Pi_text [SEP], i = 1, ..., k
        """

        q_input_ids, q_input_mask, q_type_ids = self.construct_input(self.queries[query_id], self.max_query_length + 2)
        input_ids, input_mask, input_type_ids = [], [], []
        for psg_id in psg_ids:
            inputs = self.construct_input(self._psgid2tokens(psg_id), self.max_passage_length + 2)
            input_ids.append(inputs[0])
            input_mask.append(inputs[1])
            input_type_ids.append(inputs[2])
        input_ids = np.stack(input_ids)
        input_mask = np.stack(input_mask)
        input_type_ids = np.stack(input_type_ids)
        return (
            q_input_ids[None, ...],
            q_input_mask[None, ...],
            q_type_ids[None, ...],
            input_ids,
            input_mask,
            input_type_ids,
        )

    def _psgid2tokens(self, psg_id):
        pass

    def psgid2tokens_npz(self, psg_id):
        seq_len = self.passages[psg_id][0]
        return self.passages[psg_id][1 : seq_len + 1].tolist()

    def psgid2tokens_pkl(self, psg_id):
        return self.passages[psg_id]


class BertInformationRetrievalDatasetTrain(BaseInformationRetrievalDataset):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        passages: str,
        queries: str,
        query_to_passages: str,
        max_query_length: Optional[int] = 31,
        max_passage_length: Optional[int] = 190,
        num_negatives: Optional[int] = 10,
        preprocess_fn: Optional[str] = "preprocess_bert",
        psg_cache_format: Optional[str] = "npz",
    ):
        """
        Dataset for training information retrieval models.
        
        Args:
            tokenizer: tokenizer
            passages: path to tsv with [psg_id, psg_text] entries
            queries: path to tsv with [query_id, query_text] entries
            query_to_passages: path to tsv with
                [query_id, pos_psg_id, neg_psg_id_1, ..., neg_psg_id_k] entries
            max_query_length: maximum length of query in tokens
            max_passage_length: maximum length of passage in tokens
            num_negatives: number of negative passages per positive to use for training
            preprocess_fn: either preprocess_bert or preprocess_dpr
                preprocess_bert: joint input: [CLS] query [SEP] passage [SEP]
                preprocess_dpr: separate inputs: [CLS] query [SEP], [CLS] passage [SEP]
            psg_cache_format: either pkl or npz
        """

        super().__init__(tokenizer, max_query_length, max_passage_length)
        self.num_negatives = num_negatives

        self.passages = getattr(self, f"parse_{psg_cache_format}")(passages, max_passage_length)
        self._psgid2tokens = getattr(self, f"psgid2tokens_{psg_cache_format}")
        self.queries = self.parse_pkl(queries, max_query_length)
        self.idx2psgs = self.parse_query_to_passages(query_to_passages)
        self._preprocess_fn = getattr(self, preprocess_fn)

    def __getitem__(self, idx):
        query_and_psgs = self.idx2psgs[idx]
        query_id, psg_ids = query_and_psgs[0], query_and_psgs[1:]
        inputs = self._preprocess_fn(query_id, psg_ids)
        return inputs

    def __len__(self):
        return len(self.idx2psgs)

    def parse_query_to_passages(self, file):
        idx2psgs = {}
        idx = 0
        for line in open(file, "r").readlines():
            query_and_psgs = line.split("\t")
            query_and_psgs_ids = [int(id_) for id_ in query_and_psgs]
            query_and_rel_psg_ids, irrel_psgs_ids = query_and_psgs_ids[:2], query_and_psgs_ids[2:]
            random.shuffle(irrel_psgs_ids)
            num_samples = len(irrel_psgs_ids) // self.num_negatives
            for j in range(num_samples):
                left = self.num_negatives * j
                right = self.num_negatives * (j + 1)
                idx2psgs[idx] = query_and_rel_psg_ids + irrel_psgs_ids[left:right]
                idx += 1
        return idx2psgs


class BertInformationRetrievalDatasetEval(BaseInformationRetrievalDataset):
    def __init__(
        self,
        tokenizer: TokenizerSpec,
        passages: str,
        queries: str,
        query_to_passages: str,
        max_query_length: Optional[int] = 31,
        max_passage_length: Optional[int] = 190,
        num_candidates: Optional[int] = 10,
        preprocess_fn: Optional[str] = "preprocess_bert",
        psg_cache_format: Optional[str] = "pkl",
    ):
        """
        Dataset for evaluating information retrieval models.
        
        Args:
            tokenizer: tokenizer
            passages: path to tsv with [psg_id, psg_text] entries
            queries: path to tsv with [query_id, query_text] entries
            query_to_passages: path to tsv with
                [query_id, pos_psg_id, neg_psg_id_1, ..., neg_psg_id_k] entries
            max_query_length: maximum length of query in tokens
            max_passage_length: maximum length of passage in tokens
            num_candidates: number of candidates for evaluation
            preprocess_fn: either preprocess_bert or preprocess_dpr
                preprocess_bert: joint input: [CLS] query [SEP] passage [SEP]
                preprocess_dpr: separate inputs: [CLS] query [SEP], [CLS] passage [SEP]
            psg_cache_format: either pkl or npz
        """

        super().__init__(tokenizer, max_query_length, max_passage_length)
        self.num_candidates = num_candidates

        self.passages = getattr(self, f"parse_{psg_cache_format}")(passages, max_passage_length)
        self._psgid2tokens = getattr(self, f"psgid2tokens_{psg_cache_format}")
        self.queries = self.parse_pkl(queries, max_query_length)
        self.idx2topk = self.parse_topk_list(query_to_passages)
        self._preprocess_fn = getattr(self, preprocess_fn)

    def __getitem__(self, idx):
        query_and_psgs = self.idx2topk[idx]
        query_id, psg_ids = query_and_psgs[0], query_and_psgs[1:]
        inputs = self._preprocess_fn(query_id, psg_ids)
        return [*inputs, query_id, np.array(psg_ids)]

    def __len__(self):
        return len(self.idx2topk)

    def parse_topk_list(self, file):
        idx2topk = {}
        idx = 0
        for line in open(file, "r").readlines():
            query_and_psgs = [int(id_) for id_ in line.split("\t")][: self.num_candidates + 1]
            num_samples = int(np.ceil((len(query_and_psgs) - 1) / self.num_candidates))
            for j in range(num_samples):
                left = self.num_candidates * j + 1
                right = self.num_candidates * (j + 1) + 1
                idx2topk[idx] = [query_and_psgs[0]] + query_and_psgs[left:right]
                idx += 1
        return idx2topk
