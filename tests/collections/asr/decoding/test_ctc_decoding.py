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
from functools import lru_cache

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.metrics.wer import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.metrics.wer_bpe import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.mixins import mixins
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


def char_vocabulary():
    return [' ', 'a', 'b', 'c', 'd', 'e', 'f']


@pytest.fixture()
@lru_cache(maxsize=8)
def tmp_tokenizer(test_data_dir):
    cfg = DictConfig({'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'})

    class _TmpASRBPE(mixins.ASRBPEMixin):
        def register_artifact(self, _, vocab_path):
            return vocab_path

    asrbpe = _TmpASRBPE()
    asrbpe._setup_tokenizer(cfg)
    return asrbpe.tokenizer


def check_char_timestamps(hyp: Hypothesis, decoding: CTCDecoding):
    assert hyp.timestep is not None
    assert isinstance(hyp.timestep, dict)
    assert 'timestep' in hyp.timestep
    assert 'char' in hyp.timestep
    assert 'word' in hyp.timestep

    words = hyp.text.split(decoding.word_seperator)
    words = list(filter(lambda x: x != '', words))
    assert len(hyp.timestep['word']) == len(words)


def check_subword_timestamps(hyp: Hypothesis, decoding: CTCBPEDecoding):
    assert hyp.timestep is not None
    assert isinstance(hyp.timestep, dict)
    assert 'timestep' in hyp.timestep
    assert 'char' in hyp.timestep
    assert 'word' in hyp.timestep

    chars = list(hyp.text)
    chars = list(filter(lambda x: x not in ['', ' ', '#'], chars))
    all_chars = [list(decoding.tokenizer.tokens_to_text(data['char'])) for data in hyp.timestep['char']]
    all_chars = [char for subword in all_chars for char in subword]
    all_chars = list(filter(lambda x: x not in ['', ' ', '#'], all_chars))
    assert len(chars) == len(all_chars)


class TestCTCDecoding:
    @pytest.mark.unit
    def test_constructor(self):
        cfg = CTCDecodingConfig()
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)
        assert decoding is not None

    @pytest.mark.unit
    def test_constructor_subword(self, tmp_tokenizer):
        cfg = CTCBPEDecodingConfig()
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)
        assert decoding is not None

    @pytest.mark.unit
    def test_char_decoding_greedy_forward(self,):
        cfg = CTCDecodingConfig(strategy='greedy')
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)

        B, T = 4, 20
        V = len(char_vocabulary()) + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            texts, _ = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=False
            )

            for text in texts:
                assert isinstance(text, str)

    @pytest.mark.unit
    @pytest.mark.parametrize('alignments', [False, True])
    @pytest.mark.parametrize('timestamps', [False, True])
    def test_char_decoding_greedy_forward_hypotheses(self, alignments, timestamps):
        cfg = CTCDecodingConfig(strategy='greedy', preserve_alignments=alignments, compute_timestamps=timestamps)
        vocab = char_vocabulary()
        decoding = CTCDecoding(decoding_cfg=cfg, vocabulary=vocab)

        B, T = 4, 20
        V = len(char_vocabulary()) + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            hyps, _ = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=True
            )

            for idx, hyp in enumerate(hyps):
                assert isinstance(hyp, Hypothesis)
                assert torch.is_tensor(hyp.y_sequence)
                assert isinstance(hyp.text, str)

                # alignments check
                if alignments:
                    assert hyp.alignments is not None
                    assert isinstance(hyp.alignments, tuple)
                    assert len(hyp.alignments[0]) == length[idx]
                    assert len(hyp.alignments[1]) == length[idx]

                # timestamps check
                if timestamps:
                    check_char_timestamps(hyp, decoding)

    @pytest.mark.unit
    def test_subword_decoding_greedy_forward(self, tmp_tokenizer):
        cfg = CTCBPEDecodingConfig(strategy='greedy')
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        B, T = 4, 20
        V = decoding.tokenizer.tokenizer.vocab_size + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            texts, _ = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=False
            )

            for text in texts:
                assert isinstance(text, str)

    @pytest.mark.unit
    @pytest.mark.parametrize('alignments', [False, True])
    @pytest.mark.parametrize('timestamps', [False, True])
    def test_subword_decoding_greedy_forward_hypotheses(self, tmp_tokenizer, alignments, timestamps):
        cfg = CTCBPEDecodingConfig(strategy='greedy', preserve_alignments=alignments, compute_timestamps=timestamps)
        decoding = CTCBPEDecoding(decoding_cfg=cfg, tokenizer=tmp_tokenizer)

        B, T = 4, 20
        V = decoding.tokenizer.tokenizer.vocab_size + 1
        input_signal = torch.randn(size=(B, T, V))
        length = torch.randint(low=1, high=T, size=[B])

        with torch.no_grad():
            hyps, _ = decoding.ctc_decoder_predictions_tensor(
                input_signal, length, fold_consecutive=True, return_hypotheses=True
            )

            for idx, hyp in enumerate(hyps):
                assert isinstance(hyp, Hypothesis)
                assert torch.is_tensor(hyp.y_sequence)
                assert isinstance(hyp.text, str)

                # alignments check
                if alignments:
                    assert hyp.alignments is not None
                    assert isinstance(hyp.alignments, tuple)
                    assert len(hyp.alignments[0]) == length[idx]
                    assert len(hyp.alignments[1]) == length[idx]

                # timestamps check
                if timestamps:
                    check_subword_timestamps(hyp, decoding)
