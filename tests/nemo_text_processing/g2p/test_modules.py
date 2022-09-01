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

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p(self):
        input_text = "Hello world."
        expected_output = [char for char in "həˈɫoʊ ˈwɝɫd."]

        g2p = IPAG2P(self.PHONEME_DICT_PATH)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_g2p_with_oov(self):
        input_text = "Hello Kitty!"
        expected_output = [char for char in "həˈɫoʊ KITTY!"]

        g2p = IPAG2P(self.PHONEME_DICT_PATH)

        phonemes = g2p(input_text)
        assert phonemes == expected_output
