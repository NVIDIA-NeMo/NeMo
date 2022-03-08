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

from typing import Dict, List, Union

import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = ['AggregateTokenizer']


class DummyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    # minimum compatibility
    # since all the monolingual tokenizers have a vocab
    # additional methods could be added here
    def get_vocab(self):
        return self.vocab


class AggregateTokenizer(TokenizerSpec):
    '''
    AggregateTokenizer, allowing one to combine multiple regular monolongual tokenizers into one tokenizer.
    The intuition is that we can use existing tokenizers "as is", without retraining, and associate each tokenizer with a language id
    during text processing (language id will be used to route the incoming text sample to the right tokenizer)
    as well as a token id range for detokenization (e.g. [0..127] for tokenizer A, [128..255] for tokenizer B) so
    that the orignal text could be reconstructed. Note that we assume that the incoming dict of langs / tokenizers
    is ordered, e.g. the first tokenizer will be assigned a lower interval of token ids
        Args:
        tokenizers: dict of tokenizers, keys are lang ids, values are actual tokenizers
    '''

    def __init__(self, tokenizers: Dict):

        self.tokenizers_dict = tokenizers
        self.vocabulary = []

        # the tokenizers should produce non-overlapping, ordered token ids
        # keys are language ids
        self.token_id_offset = {}

        # keys are tokenizer numbers
        self.token_id_offset_by_tokenizer_num = {}
        offset = 0
        i = 0
        for lang, tokenizer in self.tokenizers_dict.items():
            self.token_id_offset[lang] = offset
            self.token_id_offset_by_tokenizer_num[i] = offset
            offset += len(tokenizer.vocab)
            i += 1

        for tokenizer in self.tokenizers_dict.values():
            self.vocabulary.extend(tokenizer.vocab)

        self.vocab_size = len(self.vocabulary)
        logging.info(f'Aggregate vocab size: {self.vocab_size}')

        # for compatibility purposes only -- right now only the get_vocab method
        # is supported, returning the joint vocab across all tokenizers
        self.tokenizer = DummyTokenizer(self.vocabulary)

        # lookup tables to speed up token to text operations
        # if there are two tokenizers, [0,1] each with 128 tokens, the aggregate tokenizer
        # token range will be [0,255]. The below method provides two look up tables:
        # one, to convert the incoming token id -- e.g. 200 into its real id (200-127 = 73)
        # and to compute the tokenizer id that should process that token (1)
        offset_token_ids_by_token_id, tokenizers_by_token_id = self._calculate_offsets()

        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id
        self.tokenizers_by_token_id = tokenizers_by_token_id

    def _calculate_offsets(self):
        offsets = {}
        tokenizers = {}
        cur_num = 0
        tot = len(self.tokenizers_dict)
        for id in range(len(self.vocabulary)):
            off_id = id - list(self.token_id_offset.values())[cur_num]
            if cur_num + 1 < tot:
                if id >= list(self.token_id_offset.values())[cur_num + 1]:
                    cur_num += 1
                    off_id = id - list(self.token_id_offset.values())[cur_num]
            offsets[id] = off_id
            tokenizers[id] = list(self.tokenizers_dict.values())[cur_num]

        return offsets, tokenizers

    def text_to_tokens(self, text, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.text_to_tokens(text)

    def text_to_ids(self, text, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        token_ids = tokenizer.text_to_ids(text)
        token_ids[:] = [t + self.token_id_offset[lang_id] for t in token_ids]

        return token_ids

    def tokens_to_text(self, tokens, lang_id):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        tokens = []
        for id in ids:
            offset_id = self.offset_token_ids_by_token_id[id]
            tokenizer = self.tokenizers_by_token_id[id]
            tokens.extend(tokenizer.ids_to_tokens([offset_id]))
        text = ''.join(tokens).replace('▁', ' ')

        return text

    def token_to_id(self, token, lang_id):
        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.token_to_id(token) + self.token_id_offset[lang_id]

    def ids_to_tokens(self, ids):
        tokens = []

        for id in ids:
            offset_id = self.offset_token_ids_by_token_id[id]
            tokenizer = self.tokenizers_by_token_id[id]
            token = tokenizer.ids_to_tokens([offset_id])[0]
            tokens.append(token)

        return tokens

    def tokens_to_ids(self, tokens: Union[str, List[str]], langs: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        if isinstance(langs, str):
            langs = [langs]

        ids = []
        for i, token in enumerate(tokens):
            lang_id = langs[i]
            ids.append(self.token_to_id(token, lang_id))
        return ids

    @property
    def vocab(self):
        return self.vocabulary

    @property
    def langs(self):
        return list(self.tokenizers_dict.keys())
