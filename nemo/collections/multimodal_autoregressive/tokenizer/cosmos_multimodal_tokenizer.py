# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes Cosmos (Visual tokens and Text tokens)."""

import base64
import logging
import os
import unicodedata
from typing import Collection, Dict, List, Optional, Set, Tuple, Union

import tiktoken
from transformers import AddedToken, PreTrainedTokenizer

logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "emu3.tiktoken",
    "special_tokens_file": "cosmos_vision_tokens.txt",
}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # pylint: disable=line-too-long
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
# changed to use actual index to avoid misconfiguration with vocabulary expansion
SPECIAL_START_ID = 151643


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)
    }


class CosmosMultiModalTokenizer(PreTrainedTokenizer):
    """Emu3 tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        special_tokens_file,
        errors="replace",
        bos_token="<|extra_203|>",
        eos_token="<|extra_204|>",
        pad_token="<|endoftext|>",
        img_token="<|image token|>",
        boi_token="<|image start|>",
        eoi_token="<|image end|>",
        eol_token="<|extra_200|>",
        eof_token="<|extra_201|>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # how to handle errors in decoding UTF-8 byte sequences
        # use ignore if you are in streaming inference
        self.errors = errors

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)

        vision_tokens = [t.strip() for t in open(special_tokens_file).readlines() if len(t.strip()) > 0]
        SPECIAL_TOKENS = tuple(
            enumerate(
                (
                    (
                        ENDOFTEXT,
                        IMSTART,
                        IMEND,
                    )
                    + EXTRAS
                    + tuple(vision_tokens)
                ),
                start=SPECIAL_START_ID,
            )
        )
        self.special_tokens = {token: index for index, token in SPECIAL_TOKENS}
        self.special_tokens_set = set(t for _, t in SPECIAL_TOKENS)

        enc = tiktoken.Encoding(
            "Emu3",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc

        self.eod_id = self.tokenizer.eot_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.img_token = img_token
        self.boi_token = boi_token
        self.eoi_token = eoi_token
        self.eol_token = eol_token
        self.eof_token = eof_token

    def __getstate__(self):
        # for pickle lovers
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "Emu3",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        """Returns the tokenizer vocab"""
        return self.mergeable_ranks

    def convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes, str]]]) -> List[int]:
        """Converts the computed tokens to their corresponding ids"""
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)

        ids = []
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")

        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in self.special_tokens_set:
                raise ValueError("Adding unknown special tokens is not supported")

        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        regular_file_path = os.path.join(save_directory, self.vocab_files_names["vocab_file"])
        with open(regular_file_path, 'w', encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)

        excluded_special_tokens = set(
            (
                ENDOFTEXT,
                IMSTART,
                IMEND,
            )
            + EXTRAS
        )
        special_file_path = os.path.join(save_directory, self.vocab_files_names["special_tokens_file"])
        with open(special_file_path, 'w', encoding="utf8") as w:
            for k in self.special_tokens:
                if k not in excluded_special_tokens:
                    print(k, file=w)

        return (regular_file_path, special_file_path)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])

        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        """Returns the vocab size"""
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: Optional[str] = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]

        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
