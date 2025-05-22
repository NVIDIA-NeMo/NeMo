# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.speechlm2.parts.metrics import BLEU


def test_bleu():
    metric = BLEU(verbose=False)
    metric.update(
        name="dataset_1",
        refs=["a b c d e f g h i j k l", "m n o p r s t u v"],
        hyps=["a b c d e f g h i j k l", "m n o p r s t u v"],
    )
    metric.update(
        name="dataset_2",
        refs=["a b c"],
        hyps=["a b d"],
    )
    ans = metric.compute()
    assert ans["txt_bleu_dataset_1"] == 100.0
    assert ans["txt_bleu_dataset_2"] == 0.0
    assert ans["txt_bleu"] == 50.0  # average across datasets
