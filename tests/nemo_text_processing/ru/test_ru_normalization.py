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

import pytest
from parameterized import parameterized

from ..utils import CACHE_DIR, parse_test_case_file

try:
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class TestRuNormalizeWithAudio:

    normalizer = NormalizerWithAudio(input_case='cased', lang='ru', cache_dir=CACHE_DIR) if PYNINI_AVAILABLE else None
    N_TAGGED = 100

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_cardinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_cardinal(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=50)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_ordinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_ordinal(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_decimal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_decimal(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_measure.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_measure(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_date.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_date(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds, expected not in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_telephone.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_telephone(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds, expected not in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_money.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_money(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds, expected not in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_time.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_time(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds, expected not in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_electronic.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_electronic(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_whitelist.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_whitelist(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds

    @parameterized.expand(parse_test_case_file('ru/data_text_normalization/test_cases_word.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE,
        reason="`pynini` not installed, please install via nemo_text_processing/pynini_install.sh",
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_word(self, expected, test_input):
        preds = self.normalizer.normalize(test_input, n_tagged=self.N_TAGGED)
        assert expected in preds
