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

import abc
import time
from typing import List, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import MMapRetrievalIndexedDataset
from nemo.utils import logging


class RetrievalService:
    @abc.abstractmethod
    def get_knn(self, query: Union[List[str], str, torch.Tensor]):
        pass


class FaissRetrievalService(RetrievalService):
    def __init__(
        self,
        faiss_index: str,
        faiss_devices: str,
        nprobe: int,
        retrieval_index: str,
        tokenizer: TokenizerSpec,
        sentence_bert: str = 'all-mpnet-base-v2',
        sentence_bert_batch: int = 4,
        neighbors: int = 4,
    ):
        self.neighbors = neighbors
        has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")

        if faiss_devices is None or not torch.cuda.is_available():
            device_list = None
        else:
            device_list = ['cuda:' + str(device) for device in faiss_devices.split(',')]

        self.index = faiss.read_index(faiss_index)
        if has_gpu and device_list is not None:
            beg = time.time()
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co, ngpu=len(device_list))
            end = time.time()
            logging.info('convert Faiss db to GPU takes', end - beg)
        self.index.nprobe = nprobe

        self.bert_model = SentenceTransformer(sentence_bert)
        self.tokenizer = tokenizer
        self.ds = MMapRetrievalIndexedDataset(retrieval_index)
        self.pool = self.bert_model.start_multi_process_pool(device_list)
        self.sentence_bert_batch = sentence_bert_batch

    def get_knn(self, query: Union[List[str], str, torch.Tensor]):
        single_sentence = False
        if isinstance(query, str):
            single_sentence = True
            query = [query]
        elif isinstance(query, torch.Tensor):
            sentence_list = []
            for q in query:
                text = self.tokenizer.ids_to_text(q)
                sentence_list.append(text)
            query = sentence_list
        emb = self.bert_model.encode_multi_process(
            sentences=query, pool=self.pool, batch_size=self.sentence_bert_batch
        )
        D, knn = self.index.search(emb, self.neighbors)
        results = []
        for sentence_neighbors in knn:
            chunks = []
            for neighbor_chunk_id in sentence_neighbors:
                chunk_id = self.ds.get_chunk(neighbor_chunk_id)
                chunks.append(chunk_id)
            chunks = np.stack(chunks, axis=0).astype(np.int64)
            results.append(chunks)
        if single_sentence:
            # unpack the single sentence input
            return results[0]
        return np.stack(results, axis=0).astype(np.int64)
