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

import pytest
from nemo_text_processing.g2p.modules import IPAG2P


class TestModules:

    PHONEME_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict", "test_dict.txt")

    @staticmethod
    def _create_g2p(
        phoneme_dict=PHONEME_DICT_PATH,
        apply_to_oov_word=lambda x: x,
        use_chars=False,
        phoneme_probability=None,
        set_graphemes_upper=True,
    ):
        return IPAG2P(
            phoneme_dict,
            apply_to_oov_word=apply_to_oov_word,
            use_chars=use_chars,
            phoneme_probability=phoneme_probability,
            set_graphemes_upper=set_graphemes_upper,
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_parse_dict(self):
        # fmt: off
        expected_symbols = {
            'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ', 'ˈ',
            'w', 'ɝ', 'ɫ', 'd'
        }
        # fmt: on
        g2p = self._create_g2p()

        assert expected_symbols == g2p.symbols
        assert len(g2p.phoneme_dict["HELLO"]) == 1
        assert len(g2p.phoneme_dict["WORLD"]) == 1
        assert g2p.phoneme_dict["HELLO"][0] == [char for char in "həˈɫoʊ"]
        assert g2p.phoneme_dict["WORLD"][0] == [char for char in "ˈwɝɫd"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_parse_dict_with_chars(self):
        # fmt: off
        expected_symbols = {
            'H', 'E', 'L', 'L', 'O',
            'W', 'O', 'R', 'L', 'D',
            'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
            'ˈ', 'w', 'ɝ', 'ɫ', 'd'
        }
        # fmt: on
        g2p = self._create_g2p(use_chars=True)

        assert expected_symbols == g2p.symbols
        assert len(g2p.phoneme_dict["HELLO"]) == 1
        assert len(g2p.phoneme_dict["WORLD"]) == 1
        assert g2p.phoneme_dict["HELLO"][0] == [char for char in "həˈɫoʊ"]
        assert g2p.phoneme_dict["WORLD"][0] == [char for char in "ˈwɝɫd"]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p(self):
        input_text = "Hello world."
        expected_output = [char for char in "həˈɫoʊ ˈwɝɫd."]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_with_oov(self):
        input_text = "Hello Kitty!"
        expected_output = [char for char in "həˈɫoʊ KITTY!"]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_with_oov_func(self):
        input_text = "Hello Kitty!"
        expected_output = [char for char in "həˈɫoʊ test!"]
        g2p = self._create_g2p(apply_to_oov_word=lambda x: "test")

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_graphemes(self):
        input_text = "Hello world."
        expected_output = [char for char in input_text.upper()]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=0.0)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_graphemes_lower(self):
        input_text = "Hello world."
        expected_output = [char for char in input_text.lower()]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=0.0, set_graphemes_upper=False)

        phonemes = g2p(input_text)
        assert phonemes == expected_output
