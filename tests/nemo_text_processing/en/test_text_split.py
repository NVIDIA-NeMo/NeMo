# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest

from ..utils import CACHE_DIR

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class TestTextSentenceSplit:
    normalizer_en = (
        Normalizer(input_case='cased', lang='en', cache_dir=CACHE_DIR, overwrite_cache=False, post_process=True)
        if PYNINI_AVAILABLE
        else None
    )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    def test_text_sentence_split(self):
        text = "This happened in 1918 when Mrs. and Mr. Smith paid $111.12 in U.S.A. at 9 a.m. on Dec. 1. 2020. And Jan. 17th. This is an example. He paid $123 for this desk. 123rd, St. Patrick."
        gt_sentences = [
            'This happened in 1918 when Mrs. and Mr. Smith paid $111.12 in U.S.A. at 9 a.m. on Dec. 1. 2020.',
            'And Jan. 17th.',
            'This is an example.',
            'He paid $123 for this desk.',
            '123rd, St. Patrick.',
        ]
        sentences = self.normalizer_en.split_text_into_sentences(text)
        assert gt_sentences == sentences
