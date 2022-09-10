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

import ctypes.util
import os
from unittest.mock import Mock, patch

import pytest

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import PhonemizerTokenizer


class TestTTSTokenizers:

    ESPEAK_AVAILABLE = ctypes.util.find_library('espeak-ng') or ctypes.util.find_library('espeak')

    @staticmethod
    def _create_tokenizer(language, phoneme_probability=None):
        phoneme_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict", "test_dict.txt")
        tokenizer = PhonemizerTokenizer(
            language=language, phoneme_dict=phoneme_dict_path, phoneme_probability=phoneme_probability
        )
        return tokenizer

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_espeak_not_installed(self):
        with patch('ctypes.util.find_library', Mock(return_value=False)):
            with pytest.raises(ImportError, match="requires eSpeak to be installed"):
                self._create_tokenizer("en-us")

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer(self):
        input_text = "NVIDIA NeMo"
        expected_output = "ɛnˈvidiə ˈnimoʊ"
        tokenizer = self._create_tokenizer("en-us")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_with_oov(self):
        input_text = "NVIDIA NeMo is awesome"
        expected_output = "ɛnˈvidiə ˈnimoʊ ɪz ˈɔːsʌm"
        tokenizer = self._create_tokenizer("en-us")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_with_punctuation(self):
        input_text = "NVIDIA NeMo, is awesome!"
        expected_output = "ɛnˈvidiə ˈnimoʊ, ɪz ˈɔːsʌm!"
        tokenizer = self._create_tokenizer("en-us")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_with_accented_characters_and_digits(self):
        input_text = "There is 1 é in café"
        # E-speak converts standalone é to its formal name "e-acute"
        expected_output = "ðˈɛɹ ɪz wˈʌn ˌiːɐkjˈuːt ˈɪn kæfˈeɪ"
        tokenizer = self._create_tokenizer("en-us")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_with_graphemes(self):
        input_text = "Hello fhwdgads."
        expected_output = "HELLO FHWDGADS."
        tokenizer = self._create_tokenizer("en-us", phoneme_probability=0.0)

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_with_phoneme_probability(self):
        input_text = "NVIDIA NeMo"
        expected_grapheme_output = "NVIDIA NEMO"
        expected_phoneme_output = "ɛnˈvidiə ˈnimoʊ"
        tokenizer = self._create_tokenizer("en-us", phoneme_probability=0.0)

        grapheme_output = tokenizer.text_to_phonemes(input_text)
        with tokenizer.set_phone_prob(prob=1.0):
            phoneme_output = tokenizer.text_to_phonemes(input_text)

        assert grapheme_output == expected_grapheme_output
        assert phoneme_output == expected_phoneme_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_with_unknown_character(self):
        input_text = "NVIDIA🙂 NeMo 🧐"
        expected_output = "ɛnˈvidiə ˈnimoʊ"
        tokenizer = self._create_tokenizer("en-us")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_forward(self):
        input_text = "NVIDIA NeMo, is awesome!"
        tokenizer = self._create_tokenizer("en-us")

        output_tokens = tokenizer(input_text)
        output_phonemes = tokenizer.text_to_phonemes(input_text)
        # Validate tokens (list of integers) is the same length as the output phonemes
        assert len(output_tokens) == len(output_phonemes)

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_unsupported_language(self):
        with pytest.raises(ValueError, match="Language not supported"):
            self._create_tokenizer("pt-BR")

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_wrong_language(self):
        input_text = "こんにちは世界"
        expected_output = ""
        tokenizer = self._create_tokenizer("en-us")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_spanish(self):
        input_text = "¿podrás buscar algún restaurante para hoy en la noche?"
        expected_output = "¿poðɾˈas buskˈaɾ alɣˈun ɾɾˌestaʊɾˈante pˈaɾa ˈoɪ ˈen lˈa nˈotʃe?"
        tokenizer = self._create_tokenizer("es")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output

    @pytest.mark.skipif(
        not ESPEAK_AVAILABLE, reason="eSpeak not installed",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_phonemizer_tokenizer_german(self):
        input_text = "Erzähl mir eine Geschichte."
        expected_output = "ɛɾtsˈɛːl mˈiːɾ ˌaɪnə ɡəʃˈɪçtə."
        tokenizer = self._create_tokenizer("de")

        output = tokenizer.text_to_phonemes(input_text)
        assert output == expected_output
