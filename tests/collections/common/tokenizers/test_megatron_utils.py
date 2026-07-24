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

"""
Tests for Megatron tokenizer utilities.
"""

from nemo.collections.common.tokenizers.megatron_utils import list_available_models


def test_list_available_models_description_interpolates_vocab():
    """BioMegatron model descriptions should interpolate vocab_size and vocab."""
    models = list_available_models()
    biomegatron_models = [m for m in models if m.pretrained_model_name.startswith("biomegatron345m_biovocab_")]
    assert len(biomegatron_models) == 4  # 50k/30k × cased/uncased

    for model in biomegatron_models:
        # Descriptions should no longer contain literal braces
        assert "{vocab_size}" not in model.description
        assert "{vocab}" not in model.description
        # vocab (cased/uncased) should appear in the description
        assert "cased" in model.description or "uncased" in model.description
