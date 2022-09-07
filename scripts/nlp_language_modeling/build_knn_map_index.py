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
"""
This is the script to build KNN index map from Training dataset to Retrieval dataset.
For example, it maps chunk_id i from training dataset to K chunk ids in the nearest neighbor in the retrieval dataset.

It requires the training text data to be converted into `bin` and `idx` files by `preprocess_data_for_megatron.py` script.
It also requires the Faiss Index file for the Retrieval dataset built by `build_retrieval_index.py` script.

Here is an example to using it:

```python
python scripts/nlp_language_modeling/build_knn_map_index.py \
    --input_file=PATH_TO_INPUT_TRAINING_DATA \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=tokenizer.model \
    --process_chunk_size=51200 \
    --K_neighbors=16 \
    --faiss_index=PATH_TO_FAISS_INDEX_FILE \
    --devices=0,1,2,3 \
    --batch_size=1280 \
    --remove_duplicate \
    --output_file=knn_map.idx 
```
Use `--remove_duplicate` flag if the data and retrieval dataset are the same. It will remove the neighbors from the same document.
It creates a knn_map.idx KNNIndex file.
During training of RETRO model, it can look up the KNN chunk ids of the
DB dataset given the input training data chunk id. 

For large dataset, we can build the KNN index in multiple stages

stage-1: build sharding indexes, each containing a fraction of the dataset. This can be done in parallel on several machines. example,

```python
python scripts/nlp_language_modeling/build_knn_map_index.py \
    --input_file=PATH_TO_INPUT_TRAINING_DATA \
    --tokenizer-library=megatron \
    --tokenizer-type=GPT2BPETokenizer \
    --merge-file=/dataset/gpt2-merges.txt \
    --vocab-file=/dataset/gpt2-vocab.json \
    --process_chunk_size=10000 \
    --K_neighbors=16 \
    --remove_duplicate \
    --workers=2 \
    --shard_id=0 \
    --total_shards=2 \
    --devices=0,1,2 \
    --stage=1 \
    --nprobe=10 \
    --output_file=knn_shard0.save \
    --faiss_index=faiss.index
```

stage-2: merge the sharding indexes into one that is written directly to disk, example

```python
python scripts/nlp_language_modeling/build_knn_map_index.py  \
    --stage=2 \
    --output_file=knn_final.save \
    --shard_index_input=knn_shard
```
"""

import argparse
import multiprocessing
import pathlib
import sys
import time
from multiprocessing import Pool

import faiss
import numpy as np
import torch
from numba import njit, prange
from sentence_transformers import SentenceTransformer

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    KNNIndex,
    MMapRetrievalIndexedDataset,
    merge_knn_files,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

QUEUE_SIZE = 30

queue = multiprocessing.Queue(QUEUE_SIZE)
emb_queue = multiprocessing.Queue(QUEUE_SIZE)


@njit(parallel=True)
def build_map(chunk_start, result, total_chunks, start_id, end_id):
    """
    Build the map from chunk_id to a range of chunk ids that are from the same document.
    The chunk_id is in range [start_id, end_id)
    """
    size = len(chunk_start)
    for i in prange(size):
        beg = chunk_start[i]
        end = chunk_start[i + 1] if i < size - 1 else total_chunks
        if start_id < end and beg < end_id:  # [beg, end) intersect  [start_id, end_id)
            result[max(beg - start_id, 0) : (end - start_id), 0] = beg
            result[max(beg - start_id, 0) : (end - start_id), 1] = end


@njit(parallel=True)
def _dedup(chunk_id_to_range, I, tmp_neighbors, chunk_id_start, offset):
    for cid in prange(len(I)):
        if chunk_id_start + cid - offset >= 0 and chunk_id_start + cid - offset < len(chunk_id_to_range):
            beg, end = chunk_id_to_range[chunk_id_start + cid - offset]
        position = 0
        for target_chunk_id in I[cid]:
            if beg <= target_chunk_id < end:
                # target chunk is from the same document
                continue
            tmp_neighbors[cid, position] = target_chunk_id
            position += 1


def dedup(chunk_id_to_range, I, tmp_neighbors, chunk_id_start, offset):
    """
    deduplicate the KNN who are from the same document as the data chunks.
    chunk_id_to_range is calculated by build_map function, which maps chunk_id - offset to range of ids of the same document
    I is original KNN search result from Faiss.
    chunk_id_start is the chunk_id offset.
    offset is the map offset

    filtered KNN will be stored in the tmp_neighbors

    """
    if chunk_id_start < offset or chunk_id_start + len(I) - offset > len(chunk_id_to_range):
        raise ValueError('chunk_id_start out side the range')
    _dedup(chunk_id_to_range, I, tmp_neighbors, chunk_id_start, offset)


def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        vocab_file=args.vocab_file,
        merges_file=args.merge_file,
        delimiter=args.delimiter,
    )
    if not hasattr(tokenizer, "pad_id"):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    elif hasattr(tokenizer, "pad_id") and (tokenizer.pad_id is None or tokenizer.pad_id < 0):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer


def calculate_start_end(total_chunks, total_shards, shard_id):
    shard_size = total_chunks // total_shards
    splits = list(range(0, total_chunks, shard_size))
    if shard_id < total_shards - 1:
        start = splits[shard_id]
        total_chunks = splits[shard_id + 1]
    elif shard_id == total_shards - 1:
        start = splits[shard_id]
        total_chunks = total_chunks
    else:
        raise ValueError(f'{shard_id} bigger than {total_shards}')
    return start, total_chunks


def process_sentence_chunks(
    ds: MMapRetrievalIndexedDataset,
    tokenizer,
    chunk_size: int,
    stage: int,
    workers: int,
    shard_id: int,
    total_shards: int,
):
    """
    This function takes chunked tokens from the retrieval dataset and map it back to text.
    In stage 1, it divides the total work into `total_shards`, and process only at the `shard_id`.  
    If the stage is None, it process all the chunks.
    """
    total_chunks = ds.chunks
    start = 0
    threshold = 0

    if stage == 1:
        start, total_chunks = calculate_start_end(
            total_chunks=total_chunks, total_shards=total_shards, shard_id=shard_id
        )
        logging.info(f'shard_id {shard_id}, create index from chunk {start} to {total_chunks}')

    with Pool(workers) as p:
        while start < total_chunks:
            if start / total_chunks > threshold:
                logging.info(f"sentence processing {start / total_chunks} is done")
                threshold += 0.1
            slice_id = (start, min(start + chunk_size, total_chunks))
            beg = time.time()
            id_slices = ds.get_chunk(slice(*slice_id), force_no_cont_ids=True)
            end = time.time()
            logging.info(f"load {chunk_size} chunks takes {end-beg}")
            start = min(start + chunk_size, total_chunks)
            sentences = p.map(tokenizer.ids_to_text, id_slices)
            end2 = time.time()
            logging.info(f"tokenize {chunk_size} chunks takes {end2-end}")
            queue.put((sentences, slice_id))
    queue.put((None, None))


def get_sentence_chunks():
    return queue.get()


def calculate_embedding(pool, batch_size):
    while True:
        sentences, slice_id = get_sentence_chunks()
        if sentences is None:
            break
        beg = time.time()
        emb = model.encode_multi_process(sentences=sentences, pool=pool, batch_size=batch_size)
        end = time.time()
        logging.info(f"one embedding {len(emb)} batch size takes {end-beg}")
        emb_queue.put((emb, slice_id))
    emb_queue.put((None, None))


def get_emb():
    return emb_queue.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build Faiss index",)
    parser.add_argument(
        '--input_file', type=str, required=False, help='Input file',
    )
    parser.add_argument("--faiss_index", type=str, required=False, help='faiss index file for retrieval dataset')
    parser.add_argument(
        '--process_chunk_size',
        type=int,
        default=10000,
        help='The sentences in chunks that is queries to build map index',
    )
    parser.add_argument(
        '--remove_duplicate',
        action='store_true',
        help='Remove the knn neighbors that is from the same document as the data.',
    )
    parser.add_argument(
        '--K_neighbors', type=int, default=16, help='The number of neighbors to query',
    )
    parser.add_argument(
        '--dedup_margin',
        type=int,
        default=2,
        help='extra neighbors to fill the spaces of the chunks in the duplicated documents',
    )
    parser.add_argument(
        '--sentence_transformer_model',
        type=str,
        default='bert-base-nli-mean-tokens',
        help='sentence transformer to load',
    )
    parser.add_argument('--shard_id', type=int, default=None, help='run the job to create the shard_id index')
    parser.add_argument('--total_shards', type=int, default=None, help='total number of knn index shards')
    parser.add_argument(
        '--output_file', type=str, required=True, help='Output KNN Map index file',
    )
    parser.add_argument(
        '--devices', type=str, default=None, help='delimited list input with cuda devices. Specify like 0,1,2'
    )
    parser.add_argument(
        "--batch_size", type=int, default=4000, help="Batch size for encoding. Use max according to GPU MEM"
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-library',
        type=str,
        required=False,
        choices=['yttm', 'sentencepiece', 'megatron', 'huggingface', 'tabular'],
        help='What tokenizer library to use.',
    )
    group.add_argument(
        '--tokenizer-type', type=str, default=None, help='What type of tokenizer to use.',
    )
    group.add_argument(
        '--tokenizer-model', type=str, default=None, help='Path to tokenizer model.',
    )
    group.add_argument('--vocab-file', type=str, default=None, help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None, help='Path to the BPE merge file (if necessary).')
    group.add_argument('--delimiter', type=str, default=None, help='delimiter used for tabular tokenizer')
    group.add_argument(
        '--stage',
        type=int,
        default=None,
        help='used for building the large knn index in multiple stages',
        choices=[1, 2],
    )
    group.add_argument('--workers', type=int, default=None, help='number of workers to run tokenizer')
    group.add_argument(
        '--nprobe',
        type=int,
        default=10,
        help='number of probes, higher number of probes renders better results but runs slower',
    )
    group.add_argument(
        '--shard_index_input',
        type=str,
        default=None,
        help='the knn sharding index files, which are created at stage 1',
    )

    args = parser.parse_args()

    has_gpu = torch.cuda.is_available() and hasattr(faiss, "index_gpu_to_cpu")

    if not hasattr(faiss, "index_gpu_to_cpu"):
        logging.warning(
            "faiss doesn't support gpu index. Please check https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
        )

    if args.stage == 2:
        # combine shard index files into one
        input_file = pathlib.Path(args.shard_index_input)
        path = input_file.parent
        fname = input_file.name
        all_files = [str(i) for i in pathlib.Path(path).glob(fname + '*')]
        merge_knn_files(all_files, args.output_file)
        f = KNNIndex(args.output_file)
        logging.info(f'Write to {args.output_file},  Size of Index : {f.len}')
        logging.info(f'Index neighbors: {f.K}')
        logging.info(f'Index chunk start id: {f.chunk_start_id}')
        logging.info(f'Index chunk end id: {f.chunk_end_id}')
        sys.exit(0)

    model = SentenceTransformer(args.sentence_transformer_model)
    tokenizer = get_tokenizer(args)
    ds = MMapRetrievalIndexedDataset(args.input_file)

    if args.devices is None or not torch.cuda.is_available():
        device_list = None
    else:
        device_list = ['cuda:' + str(device) for device in args.devices.split(',')]

    index = faiss.read_index(args.faiss_index)
    if has_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.usePrecomputed = False
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co, ngpu=len(device_list))

    index.nprobe = args.nprobe

    start = 0
    total_chunks = ds.chunks
    if args.stage == 1:
        start, total_chunks = calculate_start_end(
            total_chunks=total_chunks, total_shards=args.total_shards, shard_id=args.shard_id
        )

    process = multiprocessing.Process(
        target=process_sentence_chunks,
        args=(ds, tokenizer, args.process_chunk_size, args.stage, args.workers, args.shard_id, args.total_shards),
    )
    process.start()

    pool = model.start_multi_process_pool(device_list)

    emb_process = multiprocessing.Process(target=calculate_embedding, args=(pool, args.batch_size))
    emb_process.start()

    if ds._index.retrieval_db and args.remove_duplicate:
        neighbors = args.K_neighbors + args.dedup_margin
        # build the id maps for quick dedup
        id_start = np.array(ds._index._chunk_id_start)
        chunk_id_to_doc_id_map = np.zeros((total_chunks - start, 2), dtype=np.int64)
        build_map(id_start, chunk_id_to_doc_id_map, ds.chunks, start, total_chunks)
    else:
        neighbors = args.K_neighbors

    chunk_id_start = start
    with KNNIndex.writer(args.output_file, args.K_neighbors, offset=start) as w:
        while True:
            emb, slice_id = get_emb()
            if emb is None:
                break
            beg = time.time()
            D, I = index.search(emb, neighbors)
            end = time.time()
            logging.info(f'search {slice_id[0]} - {slice_id[1]} takes {end-beg}')
            assert chunk_id_start == slice_id[0]
            if ds._index.retrieval_db and args.remove_duplicate:
                beg = time.time()
                tmp_neighbors = np.ones_like(I) * -1
                dedup(chunk_id_to_doc_id_map, I, tmp_neighbors, chunk_id_start, start)
                I = tmp_neighbors[:, : args.K_neighbors]
                end = time.time()
                logging.info(f'dedup {slice_id[0]} - {slice_id[1]} takes {end-beg}')
            beg = time.time()
            w.write(I)
            end = time.time()
            logging.info(f'write {slice_id[0]} - {slice_id[1]} takes {end-beg}')
            chunk_id_start += len(I)

    process.join()
    emb_process.join()
    model.stop_multi_process_pool(pool)
