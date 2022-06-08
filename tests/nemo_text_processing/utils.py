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

import os

CACHE_DIR = None
RUN_AUDIO_BASED_TESTS = False


def set_cache_dir(path: str = None):
    """
    Sets cache directory for TN/ITN unittests. Default is None, e.g. no cache during tests.
    """
    global CACHE_DIR
    CACHE_DIR = path


def set_audio_based_tests(run_audio_based: bool = False):
    """
    Sets audio-based test mode for TN/ITN unittests. Default is False, e.g. audio-based tests will be skipped.
    """
    global RUN_AUDIO_BASED_TESTS
    RUN_AUDIO_BASED_TESTS = run_audio_based


def parse_test_case_file(file_name: str):
    """
    Prepares tests pairs for ITN and TN tests
    """
    test_pairs = []
    with open(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + file_name, 'r') as f:
        for line in f:
            spoken, written = line.split('~')
            test_pairs.append((spoken, written.strip("\n")))
    return test_pairs


def get_test_cases_multiple(file_name: str = 'data_text_normalization/en/test_cases_normalize_with_audio.txt'):
    """
    Prepares tests pairs for audio based TN tests
    """
    test_pairs = []
    with open(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + file_name, 'r') as f:
        written = None
        normalized_options = []
        for line in f:
            if line.startswith('~'):
                if written:
                    test_pairs.append((written, normalized_options))
                    normalized_options = []
                written = line.strip().replace('~', '')
            else:
                normalized_options.append(line.strip())
    test_pairs.append((written, normalized_options))
    return test_pairs
