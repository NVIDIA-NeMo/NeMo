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


class TestIPAG2P:

    PHONEME_DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict")
    PHONEME_DICT_PATH_DE = os.path.join(PHONEME_DICT_DIR, "test_dict_de.txt")
    PHONEME_DICT_PATH_EN = os.path.join(PHONEME_DICT_DIR, "test_dict.txt")
    PHONEME_DICT_PATH_ES = os.path.join(PHONEME_DICT_DIR, "test_dict_es.txt")

    @staticmethod
    def _create_g2p(
        phoneme_dict=PHONEME_DICT_PATH_EN,
        locale=None,
        apply_to_oov_word=lambda x: x,
        use_chars=False,
        phoneme_probability=None,
        set_graphemes_upper=True,
    ):
        return IPAG2P(
            phoneme_dict,
            locale=locale,
            apply_to_oov_word=apply_to_oov_word,
            use_chars=use_chars,
            phoneme_probability=phoneme_probability,
            set_graphemes_upper=set_graphemes_upper,
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_dict_with_phonemes(self):
        # fmt: off
        expected_symbols = {
            'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
            'ˈ', 'w', 'ɝ', 'ɫ', 'd',
            'ˈ', 'l', 'ɛ', 'd',
            'ˈ', 'l', 'i', 'd',
            'ɛ', 'n', 'ˈ', 'v', 'ɪ', 'd', 'i', 'ə'
        }
        # fmt: on
        g2p = self._create_g2p()

        assert expected_symbols == g2p.symbols
        assert len(g2p.phoneme_dict["HELLO"]) == 1
        assert len(g2p.phoneme_dict["WORLD"]) == 1
        assert len(g2p.phoneme_dict["LEAD"]) == 2
        assert len(g2p.phoneme_dict["NVIDIA"]) == 1
        assert g2p.phoneme_dict["HELLO"][0] == list("həˈɫoʊ")
        assert g2p.phoneme_dict["WORLD"][0] == list("ˈwɝɫd")
        assert g2p.phoneme_dict["LEAD"] == [list("ˈlɛd"), list("ˈlid")]
        assert g2p.phoneme_dict["NVIDIA"][0] == list("ɛnˈvɪdiə")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_dict_with_graphemes_and_phonemes(self):
        # fmt: off
        expected_symbols = {
            'H', 'E', 'L', 'L', 'O',
            'W', 'O', 'R', 'L', 'D',
            'L', 'E', 'A', 'D',
            'N', 'V', 'I', 'D', 'I', 'A',
            'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
            'ˈ', 'w', 'ɝ', 'ɫ', 'd',
            'ˈ', 'l', 'ɛ', 'd',
            'ˈ', 'l', 'i', 'd',
            'ɛ', 'n', 'ˈ', 'v', 'ɪ', 'd', 'i', 'ə'
        }
        # fmt: on
        g2p = self._create_g2p(use_chars=True)

        assert expected_symbols == g2p.symbols
        assert len(g2p.phoneme_dict["HELLO"]) == 1
        assert len(g2p.phoneme_dict["WORLD"]) == 1
        assert len(g2p.phoneme_dict["LEAD"]) == 2
        assert len(g2p.phoneme_dict["NVIDIA"]) == 1
        assert g2p.phoneme_dict["HELLO"][0] == list("həˈɫoʊ")
        assert g2p.phoneme_dict["WORLD"][0] == list("ˈwɝɫd")
        assert g2p.phoneme_dict["LEAD"] == [list("ˈlɛd"), list("ˈlid")]
        assert g2p.phoneme_dict["NVIDIA"][0] == list("ɛnˈvɪdiə")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call(self):
        input_text = "Hello world."
        expected_output = [char for char in "həˈɫoʊ ˈwɝɫd."]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_file_or_object_dict_type(self):
        input_text = "Hello world."
        expected_output = [char for char in "həˈɫoʊ ˈwɝɫd."]

        phoneme_dict = {"HELLO": ["həˈɫoʊ"], "WORLD": ["ˈwɝɫd"], "LEAD": ["ˈlɛd", "ˈlid"], "NVIDIA": ["ɛnˈvɪdiə"]}

        g2p_file = self._create_g2p()
        g2p_dict = self._create_g2p(phoneme_dict=phoneme_dict)

        phonemes_file = g2p_file(input_text)
        phonemes_dict = g2p_dict(input_text)
        assert phonemes_dict == expected_output
        assert phonemes_file == phonemes_dict

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_oov_word(self):
        input_text = "Hello Kitty!"
        expected_output = [char for char in "həˈɫoʊ KITTY!"]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_oov_func(self):
        input_text = "Hello Kitty!"
        expected_output = [char for char in "həˈɫoʊ test!"]
        g2p = self._create_g2p(apply_to_oov_word=lambda x: "test")

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_graphemes_uppercase(self):
        input_text = "Hello world."
        expected_output = [char for char in input_text.upper()]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=0.0)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_graphemes_lowercase(self):
        input_text = "Hello world."
        expected_output = [char for char in input_text.lower()]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=0.0, set_graphemes_upper=False)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_escaped_characters(self):
        input_text = "Hello |wo rld|."
        expected_output = ["h", "ə", "ˈ", "ɫ", "o", "ʊ", " ", "wo", "rld", "."]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_instantiate_unsupported_locale(self):
        with pytest.raises(ValueError, match="Unsupported locale"):
            self._create_g2p(locale="en-USA")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_de_de(self):
        input_text = "Hallo „welt“!"
        expected_output = [char for char in "hˈaloː „vˈɛlt“!"]
        g2p = self._create_g2p(phoneme_dict=self.PHONEME_DICT_PATH_DE, locale="de-DE")

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_en_us(self):
        input_text = "Hello Kitty!"
        expected_output = [char for char in "həˈɫoʊ KITTY!"]
        g2p = self._create_g2p(locale="en-US")

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_es_es(self):
        input_text = "¿Hola mundo, amigo?"
        expected_output = [char for char in "¿ˈola mˈundo, AMIGO?"]
        g2p = self._create_g2p(phoneme_dict=self.PHONEME_DICT_PATH_ES, locale="es-ES")

        phonemes = g2p(input_text)
        assert phonemes == expected_output
