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
Tests for Canary 2.0 prompt formatting.
"""

from unittest.mock import MagicMock

import pytest
from lhotse.testing.dummies import dummy_cut

from nemo.collections.common.prompts.canary2 import canary2


def test_canary2_assert_message_includes_eos_id():
    """When the tokenizer returns -1 for CANARY_EOS, the assertion should report the actual id."""
    from lhotse import SupervisionSegment

    cut = dummy_cut(0, duration=1.0, supervisions=[SupervisionSegment("", "", 0, 1.0, text="hello")])
    cut.custom = {"source_lang": "en", "target_lang": "en"}

    prompt = MagicMock()
    prompt.encode_dialog.return_value = {"answer_ids": MagicMock()}
    prompt.tokenizer.token_to_id.return_value = -1

    with pytest.raises(AssertionError, match="token_to_id.*returned -1"):
        canary2(cut, prompt)
