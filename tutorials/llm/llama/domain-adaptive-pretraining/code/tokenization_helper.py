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
import os
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


def check_parent_directory_exists(directory_path):
    parent_directory = os.path.dirname(directory_path)
    if not os.path.exists(parent_directory):
        raise FileNotFoundError(f"Parent directory '{parent_directory}' does not exist. Please create it.")
    else:
        print(f"Parent directory '{parent_directory}' exists.")


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_token_cnt(data_root, tokenizer, batchsize, keys):
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
        ids = tokenizer(d).input_ids  # tokenizer.encode(d)
        total_cnt += sum([len(i) for i in ids])
    print("total token cnt", total_cnt)


def train_tokenizer(data_root, batchsize, vocab_size, tokenizer, keys):
    """
    Train tokenizer from scratch and evaluate number of tokens both before and after
    Args:
            data_root (str): Path to folder containing data files in jsonl format.
            batchsize (int): batch size used for the text_iterator that generates of batches of text.
            vocab_size (int): The target size of the vocabulary you want for your tokenizer.
            tokenizer (AutoTokenizer): Tokenizer to create tokens from data
            keys (list): Keys/metadata to extract from jsonl files

    Returns:
            A new tokenizer of the same type as the original one, trained on data_root

    """
    print("Before Training: ")
    get_token_cnt(data_root, tokenizer, batchsize, keys)

    def gen():
        data = []
        cnt = 0
        for f in glob.glob(data_root + "*.jsonl", recursive=True):
            f = open(f, mode="r")
            reader = jsonlines.Reader(f)
            for obj in reader:
                for key in keys:
                    data.append(obj[key])
                cnt += 1
                if cnt >= batchsize:
                    yield data
                    cnt = 0
                    data = []
            f.close()
        if len(data) > 0:
            yield data

    ds = IterableDataset.from_generator(gen)
    tokenizer = tokenizer.train_new_from_iterator(ds, vocab_size)
    print("After Training: ")
    get_token_cnt(data_root, tokenizer, batchsize, keys)
    return tokenizer


def extend_tokenizer_llama(
    data_root,
    original_tokenizer_path,
    domain_tok_vocab_path,
    new_vocab_path,
    new_model_path,
    old_ebd_path,
    new_ebd_path,
    split=1,
):
    """
    Expand the general-purpose llama tokenizer with the newly identified tokens to get an extended Tokenizer
    Args:
            data_root (str): Path to general/domain specific data to identify tokens and extend tokenizer
            original_tokenizer_path (str): Path to original tokenizer (llama 2 tokenizer downlaoded from hf)
            domain_tok_vocab_path (str): Path to domain specific vocab file (created from training a tokenizer from scratch)
            new_vocab_path (str): Path to new vocabulary file
            new_model_path (str): Path to new/extended tokenizer
            old_ebd_path (str): Path to original llama2 embedding weights downlaoded from hf
            new_ebd_path (str): Path to new embedding weights (modified due to tokenizer changes)
            split (int): Number of splits used for original model weights (model parallelism)

    Returns:
           Extended/new model files created and saved in the paths specified below

    """
    keys = ["text"]
    occur_limit = 3

    token_pattern = '[a-zA-Z]'  # or [a-zA-Z0-9]

    # Read data from data path and store
    readers = []
    for f in glob.glob(data_root + "**/*.jsonl", recursive=True):
        f = open(f, mode="r")
        readers.append(jsonlines.Reader(f))
    data = []
    for reader in readers:
        for obj in reader:
            for key in keys:
                if key in obj:
                    data.append(" " + obj[key])

    # Read domain specific voacb file and analyze added tokens
    f = open(domain_tok_vocab_path)
    vocab = json.load(f)
    print("Domain vocab size:", len(vocab))

    tokens = []
    drop_tokens = []
    print("token pattern: ", token_pattern)
    for v in vocab:
        if re.search(token_pattern, v):
            tokens.append(v.replace("Ġ", "▁"))
        else:
            drop_tokens.append(v)
    print("Num of added tokens and dropped tokens", len(tokens), len(drop_tokens))

    m = model.ModelProto()
    m.ParseFromString(open(original_tokenizer_path, 'rb').read())
    print(f'Original model pieces: {len(m.pieces)}')
    print(m.trainer_spec)
    ori_vol = []
    for piece in m.pieces:
        ori_vol.append(piece.piece)
    print("original vocab size: ", len(ori_vol))
    ori_vol = set(ori_vol)
    data = set(data)

    new_tokens = []
    for token in tokens:
        if token not in ori_vol:
            token1 = token.replace("▁", " ")
            occur_cnt = 0
            flag = True
            for s in data:
                if token1 in s:
                    occur_cnt += 1
                    if occur_cnt > occur_limit:
                        flag = False
                        break
            if flag:
                new_tokens.append(token)
    print("new token cnt: ", len(new_tokens))

    normal_cnt = len(new_tokens)
    dummy_cnt = (len(new_tokens) // 1024 + 1) * 1024 - len(new_tokens)
    add_cnt = normal_cnt + dummy_cnt
    print("add token cnt: ", add_cnt)
    print("add normal token cnt: ", normal_cnt)
    print("add dummy token cnt: ", dummy_cnt)
    assert dummy_cnt >= 3, "should be at least 3 extra tokens for finetuning"

    dummy_tokens = []
    for i in range(dummy_cnt):
        dummy_tokens.append(f"<extra_id_{i}>")

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
    for i, sym in enumerate(dummy_tokens):
        new_sym = m.SentencePiece()
        new_sym.piece = sym
        new_sym.score = 0.0  # default score for USER_DEFINED
        new_sym.type = 4  # type value for USER_DEFINED
        m.pieces.insert(N + i, new_sym)  # position after default control symbols ("<unk>", "<s>", "</s>")
        record.append([sym, N + i])

    print(f'New model pieces: {len(m.pieces)}')
    print(m.trainer_spec)

    check_parent_directory_exists(new_vocab_path)
    with open(new_vocab_path, "w", encoding="utf8") as fp:
        json.dump(record, fp)

    check_parent_directory_exists(new_model_path)
    with open(new_model_path, 'wb') as f:
        f.write(m.SerializeToString())

    if split > 1:
        old_ebd_paths = []
        for f in glob.glob(old_ebd_path + "/*.pt"):
            old_ebd_paths.append(f)

        def myFunc(s):
            return int(s.split("embedding_")[-1].split(".")[0])  ### embedding_0.pt

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

        _, N = word_embedding.shape
        add_weight = torch.zeros(add_cnt, N)
        word_embedding = torch.cat((word_embedding, add_weight), 0)
    else:
        old_ebd = torch.load(old_ebd_path)
        _, N = old_ebd['word_embeddings'].shape
        add_weight = torch.zeros(add_cnt, N)
        old_ebd['word_embeddings'] = torch.cat((old_ebd['word_embeddings'], add_weight), 0)

    if split > 1:
        _, M = output_layer.shape
        add_out = torch.zeros(add_cnt, M)
        output_layer = torch.cat((output_layer, add_out), 0)
    else:
        _, M = old_ebd['output_layer'].shape
        add_out = torch.zeros(add_cnt, M)
        old_ebd['output_layer'] = torch.cat((old_ebd['output_layer'], add_out), 0)

    sp = spm.SentencePieceProcessor()
    sp.load(original_tokenizer_path)

    for r in record:
        token = r[0]
        idx = r[1]
        ids = sp.encode_as_ids(token)
        if split > 1:
            word_embedding[idx] = torch.mean(word_embedding[ids], dim=0)
            output_layer[idx] = torch.mean(output_layer[ids], dim=0)
        else:
            old_ebd['word_embeddings'][idx] = torch.mean(old_ebd['word_embeddings'][ids], dim=0)
            old_ebd['output_layer'][idx] = torch.mean(old_ebd['output_layer'][ids], dim=0)

    if split > 1:
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
            check_parent_directory_exists(save_name)
            torch.save(temp, save_name)
    else:
        torch.save(old_ebd, new_ebd_path + str(len(m.pieces)) + ".pt")

    print("Completed saving new embeddings")


def analyze_token_usage(data_root, tokenizer_path, batchsize, keys, save_path):
    """
    Function to analyze domain tokens using frequency analysis
    Args:
            data_root (str): Path to general/domain specific data to identify tokens
            tokenizer_path (str): Path to original tokenizer (llama 2 tokenizer downlaoded from hf)
            batchsize (int): batch size used for the text_iterator that generates of batches of text.
            keys (list): Keys/metadata to extract from jsonl files
            save_path (str): path to save token usage frequency analysis results

    Returns:
           None, saves frequency analysis results to the provided path

    """
    extra_id = 32000
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    vocab_size = sp.get_piece_size()
    print("vocab_size: ", vocab_size)
    results = {}

    for name in glob.glob(data_root + "**/*.jsonl", recursive=True):
        readers = []
        f = open(name, mode="r")
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
        cnt_np = np.zeros(vocab_size)
        for d in ds:
            ids = sp.encode(d)
            ids = flatten(ids)
            counts = Counter(ids)
            for key in counts:
                cnt_np[key] += counts[key]
        ori_cnt = cnt_np[0:extra_id].sum()
        new_cnt = cnt_np[extra_id:].sum()
        total_cnt = ori_cnt + new_cnt
        print("ori cnt and new cnt: ", ori_cnt, new_cnt)
        indices = np.flip(cnt_np.ravel().argsort()[-vocab_size:])
        flag = indices >= extra_id
        cnts = cnt_np[indices]
        old_freq = []
        new_freq = []
        for i in range(len(indices)):
            if cnts[i] < 1:
                break
            id = indices[i]
            if flag[i]:
                new_freq.append([int(id), str(sp.id_to_piece(int(id))), int(flag[i]), int(cnts[i])])
            else:
                old_freq.append([int(id), str(sp.id_to_piece(int(id))), int(flag[i]), int(cnts[i])])
        results[name] = {}
        results[name]["ori_cnt"] = [int(ori_cnt), float(ori_cnt / total_cnt)]
        results[name]["new_cnt"] = [int(new_cnt), float(new_cnt / total_cnt)]
        results[name]["old_freq"] = old_freq
        results[name]["new_freq"] = new_freq
        f.close()

    with open(save_path, "w") as outfile:
        json.dump(results, outfile)
