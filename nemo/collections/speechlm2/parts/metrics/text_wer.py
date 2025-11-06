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

from collections import defaultdict
from typing import List

import jiwer
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.utils import logging


class TextWER:
    """
    Computes WER scores on text predictions.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    """

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = True):
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        self._refs.clear()
        self._hyps.clear()
        return self

    def update(self, name: str, refs: List[str], hyps: List[str]) -> None:
        for ref, hyp in zip(refs, hyps):
            normalized_ref = self.normalizer(ref)
            normalized_hyp = self.normalizer(hyp)
            
            self._refs[name].append(normalized_ref)
            self._hyps[name].append(normalized_hyp)

            if normalized_ref.strip() == "":
                wer_score = -1.0
            else:
                if self.verbose:
                    wer_score = jiwer.wer(normalized_ref, normalized_hyp)
                    logging.info(f"[REF]\t{normalized_ref}\n[HYP]\t{normalized_hyp} [WER: {wer_score:.4f}]")

    def compute(self) -> dict[str, torch.Tensor]:
        corpus_metric = {}
        
        for name in self._refs.keys():
            total_wer = 0.0
            total_ref_words = 0
            
            for ref, hyp in zip(self._refs[name], self._hyps[name]):
                ref_words = ref.split()
                total_ref_words += len(ref_words)
                total_wer += jiwer.wer(ref, hyp) * len(ref_words)
            
            # Weighted average WER
            corpus_wer = total_wer / total_ref_words if total_ref_words > 0 else 0.0
            corpus_metric[f"txt_wer_{name}"] = torch.tensor(corpus_wer)
        
        if corpus_metric:
            corpus_metric["txt_wer"] = torch.stack(list(corpus_metric.values())).mean()
        
        self._refs.clear()
        self._hyps.clear()
        
        return corpus_metric


def _identity(x):
    return x 