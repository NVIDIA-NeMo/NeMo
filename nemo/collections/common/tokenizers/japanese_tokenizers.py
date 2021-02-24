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

import re
from typing import List
from sacremoses import MosesDetokenizer, MosesTokenizer
from nemo.collections.common.tokenizers import SentencePieceDetokenizer, SentencePieceTokenizer

class JapaneseDetokenizer:

    def __init__(self):
        self.moses_detokenizer = MosesDetokenizer(lang='ja')

    def sp_detokenize(self, tokens: List[str]) -> str:
        return re.sub('▁', ' ', re.sub(' ', '', "".join(tokens)))

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenizes a list of sentencepiece tokens in Japanese
        Args:
            tokens: list of strings as tokens
        Returns:
            detokenized Japanese string
        """
        text = self.sp_detokenize(tokens)
        return self.moses_detokenizer.detokenize(text)

class JapaneseTokenizer:

    def __init__(self, sp_tokenizer_model):
        self.moses_tokenizer = MosesTokenizer(lang='ja')
        self.sp_tokenizer = SentencePieceTokenizer(model_path=sp_tokenizer_model)

    def sp_tokenize(self, text: str) -> str:
        return ' '.join(self.sp_tokenizer.text_to_tokens(text))
    
    def tokenize(self, text, escape=False, return_str=False):
        """
        Detokenizes a list of sentencepiece tokens in Japanese
        Args:
            tokens: list of strings as tokens
        Returns:
            detokenized Japanese string
        """
        text = self.moses_tokenizer.tokenize(text, escape=escape, return_str=True)
        text = self.sp_tokenize(text)
        return text if return_str else text.split()

