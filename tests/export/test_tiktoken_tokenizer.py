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


import base64
import json
import tempfile
from pathlib import Path

import pytest

from nemo.export.tiktoken_tokenizer import TiktokenTokenizer, reload_mergeable_ranks


@pytest.fixture
def sample_vocab_file():
    # Create a temporary vocab file for testing
    vocab_data = [
        {"rank": i, "token_bytes": base64.b64encode(bytes([i])).decode('utf-8'), "token_str": f"token_{i}"}
        for i in range(256)
    ]
    # Add a few merged tokens
    vocab_data.extend(
        [
            {"rank": 256, "token_bytes": base64.b64encode(b"Hello").decode('utf-8'), "token_str": "Hello"},
            {"rank": 257, "token_bytes": base64.b64encode(b"World").decode('utf-8'), "token_str": "World"},
        ]
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab_data, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()  # Cleanup after tests


def test_reload_mergeable_ranks(sample_vocab_file):
    ranks = reload_mergeable_ranks(sample_vocab_file)
    assert len(ranks) == 258  # 256 base tokens + 2 merged tokens
    assert ranks[b"Hello"] == 256
    assert ranks[b"World"] == 257


def test_tokenizer_initialization(sample_vocab_file):
    tokenizer = TiktokenTokenizer(sample_vocab_file)
    assert tokenizer.bos_token_id == 1  # <s>
    assert tokenizer.eos_token_id == 2  # </s>
    assert tokenizer.pad_id == 2  # same as eos_token_id


def test_encode_decode(sample_vocab_file):
    tokenizer = TiktokenTokenizer(sample_vocab_file)
    text = "Hello World"
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert isinstance(decoded_text, str)


def test_batch_decode(sample_vocab_file):
    tokenizer = TiktokenTokenizer(sample_vocab_file)
    tokens = [[1000, 1001, 1002]]  # Example token IDs above num_special_tokens
    decoded_text = tokenizer.batch_decode(tokens)
    assert isinstance(decoded_text, str)


def test_special_token_handling(sample_vocab_file):
    tokenizer = TiktokenTokenizer(sample_vocab_file)
    # Test that special tokens are properly filtered during decoding
    tokens = [tokenizer.bos_token_id, 1000, 1001, tokenizer.eos_token_id]
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text != ""  # Should decode the non-special tokens


def test_empty_decode(sample_vocab_file):
    tokenizer = TiktokenTokenizer(sample_vocab_file)
    # Test decoding with only special tokens
    tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id]
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == ""  # Should return empty string


def test_batch_decode_numpy_tensor(sample_vocab_file):
    import numpy as np
    import torch

    tokenizer = TiktokenTokenizer(sample_vocab_file)
    np_tokens = np.array([[1000, 1001, 1002]])
    torch_tokens = torch.tensor([[1000, 1001, 1002]])

    np_decoded = tokenizer.batch_decode(np_tokens)
    torch_decoded = tokenizer.batch_decode(torch_tokens)

    assert isinstance(np_decoded, str)
    assert isinstance(torch_decoded, str)
    assert np_decoded == torch_decoded
