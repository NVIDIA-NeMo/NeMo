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
import re
from typing import Optional

import pandas as pd

from nemo.collections.common.tokenizers.char_tokenizer import TokenizerSpec
from nemo.utils import logging

__all__ = ['RegExTokenizer']

DEFAULT_MASK_TOKEN = '<MASK>'
DEFAULT_BOS_TOKEN = '^'
DEFAULT_EOS_TOKEN = '&'
DEFAULT_PAD_TOKEN = '<PAD>'
DEFAULT_SEP_TOKEN = '<SEP>'
DEFAULT_UNK_TOKEN = '?'


class RegExTokenizer(TokenizerSpec):
    """
    A regular expression-based tokenizer at word boundary.
    This tokenizer default to support MegaMolBART.
    <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/megamolbart>
    """

    def __init__(
        self,
        regex: Optional[str] = "",
        mask_token: Optional[str] = DEFAULT_MASK_TOKEN,
        bos_token: Optional[str] = DEFAULT_BOS_TOKEN,
        eos_token: Optional[str] = DEFAULT_EOS_TOKEN,
        pad_token: Optional[str] = DEFAULT_PAD_TOKEN,
        sep_token: Optional[str] = DEFAULT_SEP_TOKEN,
        unk_token: Optional[str] = DEFAULT_UNK_TOKEN,
    ):
        """
        Args:
            regex: regular expression that defined tokenization rules
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
        """
        self.regex = regex
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.unk_token = unk_token

        # holds base name of .model/.vocab files
        self.base_fname = None

        # initialize with default vocab
        self.vocab = {
            DEFAULT_PAD_TOKEN: 0,  # pad_token
            DEFAULT_UNK_TOKEN: 1,  # unk_token
            DEFAULT_BOS_TOKEN: 2,  # begin_token
            DEFAULT_EOS_TOKEN: 3,  # end_token
            DEFAULT_MASK_TOKEN: 4,  # mask_token
            DEFAULT_SEP_TOKEN: 5,  # sep_token
        }
        self._update_cache()

        # Computed attributes
        self._compile_regex()

    def _update_cache(self):
        # Cache data/attributes required for tokenization
        self._unk_id = self.vocab.get(self.unk_token, DEFAULT_UNK_TOKEN)
        self._decode_vocab = {i: t for t, i in self.vocab.items()}

    def _compile_regex(self):
        regex_string = r"("
        regex_string += self.regex + r"|"
        regex_string += r".)"
        self._compiled_regex = re.compile(regex_string)

    def text_to_tokens(self, text):
        # Begin token
        tokens = [self.bos_token]
        tokens.extend(self._compiled_regex.findall(text))
        # End token
        tokens.append(self.eos_token)

        return tokens

    def tokens_to_text(self, tokens):
        tokens_list = []
        for token in tokens:
            if token[0] == self.bos_token:
                token = token[1:]

            # Remove end token and the following values
            if self.eos_token in token:
                eos_idx = token.index(self.eos_token)
                token = token[:eos_idx]

            tokens_list.append(token)

        text = ["".join(tokens) for tokens in tokens_list]
        return text

    def token_to_ids(self, tokens):
        ids_list = []
        for token in tokens:
            ids_list.append(self.vocab.get(token, self._unk_id))
        return ids_list

    def tokens_to_ids(self, token_data):
        if isinstance(token_data, str):
            token_data = [token_data]

        ids_list = []
        for tokens in token_data:
            ids = self.token_to_ids(tokens)
            ids_list.append(ids)
        return ids_list

    def ids_to_tokens(self, ids):
        tokens_list = []
        for ids in ids:
            for token_id in ids:
                token = self._decode_vocab.get(token_id)
                if token is None:
                    raise ValueError(f"Token id {token_id} is not recognised")

            tokens = [self._decode_vocab.get(token_id) for token_id in ids]
            tokens_list.append(tokens)

        return tokens_list

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        tokens = [tokens]
        return self.tokens_to_ids(tokens)[0]

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        return self.tokens_to_text(tokens)

    def save_tokenizer(self, base_fname=None):
        """
        Saves tokenizer's regex (base_fname.model) and vocab (base_fname.vocab) files
        """
        if base_fname.endswith(".model"):
            base_fname = os.path.splitext(base_fname)[0]

        if base_fname:
            self.base_fname = base_fname

        if not self.base_fname:
            raise ValueError(f"base_fname must be specified")

        vocab_file = self.base_fname + '.vocab'
        regex_file = self.base_fname + '.model'

        logging.debug(f"Saving vocabulary to file = {vocab_file}")
        with open(vocab_file, 'w') as fp:
            for token in self.vocab:
                fp.write(f"{token[0]}\n")

        logging.debug(f"Saving regex to file = {regex_file}")
        open(regex_file, 'w').write(self.regex)

    def load_tokenizer(self, base_fname):
        """
        Loads tokenizer's regex (base_fname.model) and vocab (base_fname.vocab) files
        """
        if base_fname.endswith(".model"):
            base_fname = os.path.splitext(base_fname)[0]

        if base_fname:
            self.base_fname = base_fname

        if not self.base_fname:
            raise ValueError(f"base_fname must be specified")

        vocab_file = self.base_fname + '.vocab'
        regex_file = self.base_fname + '.model'

        # load vocab file
        # vocab_file: path to file with vocabulary which consists
        # of characters separated by \n (None/"" for empty vocab)

        logging.debug(f"Loading vocabulary from file = {vocab_file}")
        if os.path.exists(vocab_file):
            vocab = {}
            with open(vocab_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        vocab[line] = len(vocab)
            self.vocab = vocab
        else:
            raise RuntimeError(f"Missing vocab_file = {vocab_file}")

        # load regex from a file
        if os.path.exists(regex_file):
            logging.debug(f"Loading regex from file = {regex_file}")
            self.regex = open(regex_file, encoding="utf-8").read().strip()
        else:
            raise RuntimeError(f"Missing regex_file = {regex_file}")

        return self

    def build_vocab_from_csv(self, data_csv_file, col="smiles"):
        """
        Learns vocabulary from a CSV file. Can be called multiple times to update vocabulary.
        """
        logging.debug(f"Building vocabulary from CSV col = {col} file = {data_csv_file}")

        # NOTE this has to be run on each CSV file
        if not os.path.exists(data_csv_file):
            raise ValueError(f"Data file: {data_csv_file} is missing")

        df = pd.read_csv(data_csv_file)

        vocab = self.vocab
        for d in df[col]:
            tokens = self.text_to_tokens(d)
            logging.debug(f"Text: {d}, Tokens: {tokens}")
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        sorted_vocab = sorted(vocab.items(), key=lambda k_v: k_v[1])
        logging.debug(f"Vocab: {sorted_vocab}")

        self.vocab = vocab
        self._update_cache()

    def build_vocab_from_text(self, data_text_file):
        """
        Learns vocabulary from a text file. Can be called multiple times to update vocabulary.
        """
        logging.debug(f"Building vocabulary from TEXT file = {data_text_file}")

        # NOTE this has to be run on each text file
        if not os.path.exists(data_text_file):
            raise ValueError(f"Data file: {data_text_file} is missing")

        vocab = self.vocab
        for d in open(data_text_file, encoding="utf-8").readlines():
            d = d.rstrip()
            tokens = self.text_to_tokens(d)
            logging.debug(f"Text: {d}, Tokens: {d}")
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        sorted_vocab = sorted(vocab.items(), key=lambda k_v: k_v[1])
        logging.debug(f"Vocab: {sorted_vocab}")

        self.vocab = vocab
        self._update_cache()
