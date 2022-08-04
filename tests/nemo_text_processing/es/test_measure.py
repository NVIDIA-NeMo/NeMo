# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from parameterized import parameterized

from ..utils import CACHE_DIR, RUN_AUDIO_BASED_TESTS, parse_test_case_file

try:
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    from nemo_text_processing.text_normalization.normalize import Normalizer
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class TestMeasure:
    inverse_normalizer_es = (
        InverseNormalizer(lang='es', cache_dir=CACHE_DIR, overwrite_cache=False) if PYNINI_AVAILABLE else None
    )

    @parameterized.expand(parse_test_case_file('es/data_inverse_text_normalization/test_cases_measure.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm_es(self, test_input, expected):
        pred = self.inverse_normalizer_es.inverse_normalize(test_input, verbose=False)
        assert pred == expected

    normalizer = (
        Normalizer(input_case='cased', lang='es', cache_dir=CACHE_DIR, overwrite_cache=False)
        if PYNINI_AVAILABLE
        else None
    )

    normalizer_with_audio = (
        NormalizerWithAudio(input_case='cased', lang='es', cache_dir=CACHE_DIR, overwrite_cache=False)
        if CACHE_DIR and RUN_AUDIO_BASED_TESTS and PYNINI_AVAILABLE
        else None
    )

    @parameterized.expand(parse_test_case_file('es/data_text_normalization/test_cases_measure.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        pred = self.normalizer.normalize(test_input, verbose=False)
        assert pred == expected

        if self.normalizer_with_audio:
            pred_non_deterministic = self.normalizer_with_audio.normalize(
                test_input, n_tagged=500, punct_post_process=False
            )
            assert expected in pred_non_deterministic
