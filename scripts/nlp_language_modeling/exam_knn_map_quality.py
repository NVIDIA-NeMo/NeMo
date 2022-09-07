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
This is the script to exam the KNN mapping quality between indexed data and indexed retrieval database.

It requires the training text data to be converted into `bin` and `idx` files by `preprocess_data_for_megatron.py` script.
It also requires KNNIndex built by `build_retrieval_index.py` script.

Here is an example to using it:

```python
python scripts/nlp_language_modeling/exam_knn_map_quality.py \
    --input_data_prefix=PATH_TO_DATA \
    --input_retrieval_prefix=PATH_TO_RETRIEVAL_DATA \
    --knn_index=PATH_TO_KNN_MAP_INDEX \
    --chunk_ids 2 3000 4000 5000 6000 \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=tokenizer.model
```

"""
import argparse

from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    KNNIndex,
    MMapRetrievalIndexedDataset,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build Faiss index",)
    parser.add_argument(
        '--input_data_prefix', type=str, required=True, help='Input data prefix',
    )
    parser.add_argument(
        '--input_retrieval_prefix', type=str, required=True, help='Input retrieval data prefix',
    )
    parser.add_argument(
        '--knn_index', type=str, required=True, help='Input knn map index file',
    )
    parser.add_argument(
        '--neighbors', type=int, default=None, help='number of neighbors',
    )
    parser.add_argument(
        '--chunk_ids',
        nargs='+',
        default=[1, 3, 5, 7],
        type=int,
        help='space separate listed of chunk ids in input data',
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-library',
        type=str,
        required=True,
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

    args = parser.parse_args()

    tokenizer = get_tokenizer(args)
    data_ds = MMapRetrievalIndexedDataset(args.input_data_prefix)
    retrieval_ds = MMapRetrievalIndexedDataset(args.input_retrieval_prefix)
    knn_index = KNNIndex(args.knn_index)
    assert knn_index.len == data_ds.chunks
    logging.info(f'Data index has {data_ds.chunks} chunks')
    logging.info(f'Retrieval Data index has {retrieval_ds.chunks} chunks')
    logging.info(f'KNN index has {knn_index.K} neighbors')
    assert data_ds._index.chunk_size == retrieval_ds._index.chunk_size
    print_num_neighbors = knn_index.K
    if args.neighbors is not None:
        assert args.neighbors <= knn_index.K
        print_num_neighbors = args.neighbors

    for chunk_id in args.chunk_ids:
        token_ids = data_ds.get_chunk(chunk_id, force_no_cont_ids=True)
        assert token_ids.shape[0] == data_ds._index.chunk_size
        query_text = tokenizer.ids_to_text(token_ids)
        neighbor_chunk_ids = knn_index.get_KNN_chunk_ids(chunk_id)
        neighbor_chunk_ids = neighbor_chunk_ids[:print_num_neighbors]
        print(f'Query: {query_text}')
        for i, neighbor in enumerate(neighbor_chunk_ids):
            token_ids = retrieval_ds.get_chunk(neighbor)
            half = token_ids.shape[0] // 2
            assert half == data_ds._index.chunk_size
            neighbor_match = tokenizer.ids_to_text(token_ids[:half])
            neighbor_extend = tokenizer.ids_to_text(token_ids[half:])
            print(f' ->K{i}: {neighbor_match} --- {neighbor_extend}')
        print('         ---------------         ')
