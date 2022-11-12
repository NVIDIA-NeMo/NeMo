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

from itertools import permutations

import pytest
import torch

from nemo.collections.asr.metrics.der import calculate_session_cpWER, calculate_session_cpWER_bruteforce


def word_count(spk_transcript):
    return sum([len(w.split()) for w in spk_transcript])


def calculate_wer_count(_ins, _del, _sub, ref_word_count):
    return (_ins + _del + _sub) / ref_word_count


def permuted_input_test(hyp, ref, calculated):
    """
    Randomly permute the input to see if evaluation result stays the same.
    """
    for hyp_permed in permutations(hyp):
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp_permed, spk_reference=ref)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6


class TestConcatMinPermWordErrorRate:
    """
    Tests for cpWER calculation.
    """

    @pytest.mark.unit
    def test_cpwer_oneword(self):
        hyp = ["oneword"]
        ref = ["oneword"]
        _ins, _del, _sub = 0, 0, 0
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

        # Test with a substitution
        hyp = ["wrongword"]
        _ins, _del, _sub = 0, 0, 1
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_perfect(self):
        hyp = ["ff", "aa bb cc", "dd ee"]
        ref = ["aa bb cc", "dd ee", "ff"]
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        calculated = 0
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)

    @pytest.mark.unit
    def test_cpwer_spk_counfusion_and_asr_error(self):
        hyp = ["aa bb c ff", "dd e ii jj kk", "hi"]
        ref = ["aa bb cc ff", "dd ee gg jj kk", "hh ii"]
        _ins, _del, _sub = 0, 1, 4
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_undercount(self):
        hyp = ["aa bb cc", "dd ee gg", "hh ii", "jj kk"]
        ref = ["aa bb cc", "dd ee", "ff", "gg", "hh ii", "jj kk"]
        _ins, _del, _sub = 0, 1, 0
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_overcount(self):
        hyp = ["aa bb cc", "dd ee gg hh", "ii jj kk"]
        ref = ["aa bb cc", "dd ee ff gg hh ii jj kk"]
        _ins, _del, _sub = 0, 1, 0
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6
