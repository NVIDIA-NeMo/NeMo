# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import io
import json
import math
import os
import pprint
import random
import re
import sys
from collections import Counter

import jsonlines
import numpy as np
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model
import torch
from datasets import Dataset, IterableDataset, load_dataset
from tokenization_helper import *
from tokenizers import (
    SentencePieceBPETokenizer,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast


def get_token_cnt_spm(data_root, tokenizer, batchsize, keys):
    """
    Function to get number of tokens generated from a given dataset

    Args:
            data_root (str): Path to folder containing data files in jsonl format.
            tokenizer (AutoTokenizer): Tokenizer to create tokens from data
            batchsize (int): batch size used for the text_iterator that generates of batches of text.
            keys (list): Keys/metadata to extract from jsonl files

    Returns:
            A new tokenizer of the same type as the original one, trained on data_root
    """
    readers = []
    for f in glob.glob(data_root + "**/*.jsonl", recursive=True):
        f = open(f, mode="r")
        readers.append(jsonlines.Reader(f))

    def gen():
        data = []
        cnt = 0
        for reader in readers:
            for obj in reader:
                for key in keys:
                    data.append(obj[key])
                cnt += 1
                if cnt >= batchsize:
                    yield data
                    cnt = 0
                    data = []
        if len(data) > 0:
            yield data

    ds = IterableDataset.from_generator(gen)
    total_cnt = 0
    for d in ds:
        ids = tokenizer.encode(d)  # for spm model
        total_cnt += sum([len(i) for i in ids])
    print("total token cnt", total_cnt)


def extend_tokenizer(vocab_size, split, model_type):
    """
    Expand the general-purpose tokenizer with the newly identified tokens to get an extended Tokenizer
    Args:
            vocab_size (int): The target size of the vocabulary you want for domain specific tokenizer.
            split (int): Number of splits used for original model weights (model parallelism)
            model_type (str): Model type/family

    Returns:
           Extended tokenizer is created and saved in the paths specified below

    """
    digit_flag = False
    rm_subword_flag = False
    unseen_flag = True
    init_out_flag = True
    newinit_flag = False

    tag = "code_gen"  # Tag to identify custom_tokenization per use case
    data_root = "./general_data"  # path to general datasets collected from open-source domain
    original_tokenizer_path = (
        f"./models/tokenizer/{model_type}/original_tokenizer/tokenizer.model"  # path to original tokenizer
    )
    domain_tok_vocab_path = f"./models/tokenizer/{model_type}/custom_tokenizer_init_{vocab_size}.json/vocab.json"  # path to domain specific vocab file (created previously)

    # New model file paths that will be created
    new_vocab_path = f"./models/tokenizer/{model_type}/new_tokenizer/" + tag + "_vocab.json"
    new_model_path = f"./models/tokenizer/{model_type}/new_tokenizer/tokenizer_" + tag + ".model"
    old_ebd_path = f"./models/weight/{model_type}/ori_{model_type}-hf_weight/"
    new_ebd_path = f"./models/weight/{model_type}/new_{model_type}-hf_weight/"

    extend_tokenizer_llama(
        data_root,
        original_tokenizer_path,
        domain_tok_vocab_path,
        new_vocab_path,
        new_model_path,
        old_ebd_path,
        new_ebd_path,
        split,
    )

    print("Vocabulary path for extended tokenizer: ", new_vocab_path)
    print("Tokenizer model path for extended tokenizer: ", new_model_path)
    print("Modified embedding weights path for extended tokenizer: ", new_ebd_path)


def extend_tokenizer_high_freq_tokens(
    data_root,
    original_tokenizer_path,
    new_tokens,
    new_vocab_path,
    new_model_path,
    old_ebd_path=None,
    new_ebd_path=None,
    split=8,
):
    """
    Expand the original llama tokenizer with the newly identified high frequency tokens to get a customized tokenizer
    Args:
            data_root (str): Path to general/domain specific data to identify tokens and extend tokenizer
            original_tokenizer_path (str): Path to original tokenizer (llama 2 tokenizer downlaoded from hf)
            new_tokens (List(str)): List of idenitfied high frequency tokens
            new_vocab_path (str): Path to new vocabulary file
            new_model_path (str): Path to new/customized tokenizer
            old_ebd_path (str): Path to original llama2 embedding weights downlaoded from hf
            new_ebd_path (str): Path to new embedding weights (modified due to tokenizer changes)
            split (int): Number of splits used for original model weights (model parallelism)

    Returns:
           New model files created and saved in the paths specified below

    """
    m = model.ModelProto()
    m.ParseFromString(open(original_tokenizer_path, 'rb').read())
    ori_vocab_size = len(m.pieces)

    print("token_cnt with original tokenizer: ")
    sp = spm.SentencePieceProcessor()
    sp.load(original_tokenizer_path)
    get_token_cnt_spm(data_root, sp, batchsize=1000, keys=["text"])

    add_normal_cnt = len(new_tokens)
    add_dummy_cnt = (len(new_tokens) // 1024 + 1) * 1024 - len(new_tokens)
    total_add_cnt = add_normal_cnt + add_dummy_cnt
    new_vocab_size = total_add_cnt + ori_vocab_size
    total_cnt = new_vocab_size + 768  ## consider 768 padding vocab in llama/mixtral tokenizer
    print("original vocab_size: ", ori_vocab_size)
    print("added normal vocab: ", add_normal_cnt)
    print("added dummy vocab: ", add_dummy_cnt)
    print("new vocab_size: ", new_vocab_size)
    print("padded vocab: ", 768)
    print("total cnt (with padding vocab): ", total_cnt)
    assert add_dummy_cnt >= 3, "there should be at least 3 extra tokens for finetuning"

    record = []
    N = len(m.pieces)
    for i, sym in enumerate(new_tokens):
        new_sym = m.SentencePiece()
        new_sym.piece = sym
        new_sym.score = 0.0  # default score for USER_DEFINED
        new_sym.type = 4  # type value for USER_DEFINED
        m.pieces.insert(N + i, new_sym)  # position after default control symbols ("<unk>", "<s>", "</s>")
        record.append([sym, N + i])

    N = len(m.pieces)
    for i in range(add_dummy_cnt):
        new_sym = m.SentencePiece()
        new_sym.piece = f"<extra_id_{i}>"
        new_sym.score = 0.0  # default score for USER_DEFINED
        new_sym.type = 4  # type value for USER_DEFINED
        m.pieces.insert(N + i, new_sym)  # position after default control symbols ("<unk>", "<s>", "</s>")
        record.append([new_sym.piece, N + i])

    with open(new_vocab_path, "w", encoding="utf8") as fp:
        json.dump(record, fp)

    with open(new_model_path, 'wb') as f:
        f.write(m.SerializeToString())

    print("token_cnt with customized tokenizer: ")
    sp = spm.SentencePieceProcessor()
    sp.load(new_model_path)
    get_token_cnt_spm(data_root, sp, batchsize=1000, keys=["text"])

    old_ebd_paths = []
    for f in glob.glob(old_ebd_path + "/*.pt"):
        old_ebd_paths.append(f)

    def myFunc(s):
        return int(s.split("embedding_")[-1].split(".")[0])

    old_ebd_paths.sort(key=myFunc)
    word_embeddings = []
    output_layers = []
    for f in old_ebd_paths:
        temp = torch.load(f)
        word_embeddings.append(temp['word_embeddings'])
        output_layers.append(temp['output_layer'])
    word_embedding = torch.cat(word_embeddings, dim=1)
    output_layer = torch.cat(output_layers, dim=0)
    print("word_embedding shape: ", word_embedding.shape)
    print("output_layer shape: ", output_layer.shape)

    N_ori_emb, N = word_embedding.shape
    add_weight = torch.zeros(total_add_cnt, N)
    word_embedding = torch.cat((word_embedding[:ori_vocab_size], add_weight, word_embedding[ori_vocab_size:]), 0)

    _, M = output_layer.shape
    add_out = torch.zeros(total_add_cnt, M)
    output_layer = torch.cat((output_layer[:ori_vocab_size], add_out, output_layer[ori_vocab_size:]), 0)

    sp = spm.SentencePieceProcessor()
    sp.load(original_tokenizer_path)

    for r in record:
        token = r[0]
        idx = r[1]
        ids = sp.encode_as_ids(token)
        word_embedding[idx] = torch.mean(word_embedding[ids], dim=0)
        output_layer[idx] = torch.mean(output_layer[ids], dim=0)

    word_embedding = word_embedding.bfloat16()
    output_layer = output_layer.bfloat16()

    vocab_size, dimension = word_embedding.shape
    split_dimension = dimension // (split)
    split_vocab_size = vocab_size // split
    prefix = new_ebd_path + "/embedding_"
    for i in range(split):
        start = i * split_dimension
        end = (i + 1) * split_dimension
        st = i * split_vocab_size
        ed = (i + 1) * split_vocab_size
        save_name = prefix + f"{i}" + ".pt"
        temp = {}
        temp['word_embeddings'] = word_embedding[:, start:end]  # split word_embedding
        temp['output_layer'] = output_layer[st:ed, :]  # split output_layer
        torch.save(temp, save_name)

    print("Completed saving new embeddings")


if __name__ == "__main__":
    original_tokenizer_path = sys.argv[1]  # original sentencepiece model
    new_tokens = sys.argv[2]  # new tokens to be added
    new_model_path = sys.argv[3]  # augmented sentencepiece model
    old_ebd_path = sys.argv[4]  # original embeddings
    new_ebd_path = sys.argv[5]  # augmented embeddings
    new_vocab_path = sys.argv[6]  # path to record added new tokens
    split = int(sys.argv[7])  # num of partitions to split the augmented embeddings
    data_root = sys.argv[8]

    extend_tokenizer_high_freq_tokens(
        data_root,
        original_tokenizer_path,
        new_tokens,
        new_vocab_path,
        new_model_path,
        old_ebd_path,
        new_ebd_path,
        split,
    )
