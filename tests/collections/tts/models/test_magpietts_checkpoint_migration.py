# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Unit tests for loading legacy MagpieTTS transcript embeddings."""

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.tts.models.magpietts import MagpieTTSModel


def _minimal_legacy_model(num_embeddings: int, embedding_dim: int, conditioning_tokens: int):
    """Build only the state needed to exercise MagpieTTSModel.load_state_dict()."""
    model = MagpieTTSModel.__new__(MagpieTTSModel)
    torch.nn.Module.__init__(model)
    model.legacy_text_conditioning = True
    model.text_conditioning_tokenizer_name = "context_tokenizer"
    model.tokenizer = SimpleNamespace(
        num_tokens_per_tokenizer={model.text_conditioning_tokenizer_name: conditioning_tokens}
    )
    model.text_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    return model


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_load_legacy_text_embedding_removes_conditioning_slice_and_preserves_bos_eos():
    """A legacy checkpoint may contain conditioning rows immediately before its BOS/EOS rows."""
    # NVBug 6390422: the released checkpoint has 2,362 rows while the legacy model expects 2,317;
    # the difference is the 45-token conditioning vocabulary.
    model_num_embeddings = 2317
    conditioning_tokens = 45
    embedding_dim = 2
    model = _minimal_legacy_model(model_num_embeddings, embedding_dim, conditioning_tokens)

    checkpoint_weight = torch.arange(
        (model_num_embeddings + conditioning_tokens) * embedding_dim, dtype=torch.float32
    ).reshape(model_num_embeddings + conditioning_tokens, embedding_dim)

    model.load_state_dict({'text_embedding.weight': checkpoint_weight})

    expected_weight = torch.cat([checkpoint_weight[: model_num_embeddings - 2], checkpoint_weight[-2:]], dim=0)
    torch.testing.assert_close(model.text_embedding.weight, expected_weight)


@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_load_legacy_text_embedding_rejects_unrecognized_size_mismatch():
    """Do not silently truncate a mismatch that is not exactly the conditioning vocabulary size."""
    model = _minimal_legacy_model(num_embeddings=7, embedding_dim=2, conditioning_tokens=3)
    checkpoint_weight = torch.zeros(9, 2)

    with pytest.raises(RuntimeError, match='size mismatch for weight'):
        model.load_state_dict({'text_embedding.weight': checkpoint_weight})
