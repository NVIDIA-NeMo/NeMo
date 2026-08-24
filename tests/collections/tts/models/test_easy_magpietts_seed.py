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

import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from scripts.magpietts.create_easy_magpietts_oneshot_flow_seed import (
    _defer_dataset_setup,
    transfer_compatible_pretrained_state,
)
from torch import nn


class _TinyEasyMagpie(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_type = "nemotron_h"
        self.decoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.num_semantic_codebooks = 1
        self.frame_stacking_factor = 2
        self.num_all_tokens_per_codebook = 3
        self.final_proj = nn.Linear(4, 6)
        self.conditioning_module = nn.Linear(4, 5)
        self.local_flow = nn.Linear(4, 6)
        self.flow_acoustic_in_projection = nn.Linear(3, 4)
        nn.init.zeros_(self.flow_acoustic_in_projection.weight)
        nn.init.zeros_(self.flow_acoustic_in_projection.bias)


class _TinyCodecHelper:
    semantic_rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def semantic_codes_to_embedding(self, semantic_codes, codes_len):
        del codes_len
        rows = self.semantic_rows.to(semantic_codes.device)[semantic_codes[0, 0]]
        return rows.T.unsqueeze(0)


class _TinyProjectionConditionedTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_type = "nemotron_h"
        self.decoder = nn.Linear(4, 4)
        self.num_semantic_codebooks = 1
        self.frame_stacking_factor = 1
        self.codebook_size = 3
        self.num_all_tokens_per_codebook = 7
        self.semantic_codec_embedding_dim = 2
        self.acoustic_codec_embedding_dim = 3
        self.audio_bos_id = 3
        self.audio_eos_id = 4
        self.context_audio_bos_id = 5
        self.context_audio_eos_id = 6
        self.cfg = SimpleNamespace(embedding_dim=4)
        self.audio_in_projection = nn.Identity()
        self.oneshot_separate_context_input_projection = True
        self.audio_embeddings = nn.ModuleList([nn.Embedding(7, 4)])
        self.flow_acoustic_in_projection = nn.Linear(3, 4)
        self.flow_context_audio_embeddings = nn.ModuleList([nn.Embedding(7, 4)])
        self.flow_context_acoustic_in_projection = nn.Linear(3, 4)
        self.final_proj = nn.Linear(4, 7)
        self.local_flow = nn.Linear(4, 3)
        self._codec_helper = _TinyCodecHelper()


def test_seed_model_defers_all_configured_datasets():
    cfg = OmegaConf.create(
        {
            "train_ds": {"dataset": {"input_cfg": [{"type": "lhotse_shar"}]}},
            "validation_ds": {"dataset": {"input_cfg": [{"type": "nemo"}]}},
            "test_ds": None,
        }
    )

    _defer_dataset_setup(cfg)

    assert cfg.train_ds.defer_setup is True
    assert cfg.validation_ds.defer_setup is True
    assert cfg.test_ds is None


def test_transfers_compatible_tensors_and_reports_new_flow_initialization():
    torch.manual_seed(1)
    source = _TinyEasyMagpie()
    source_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith(("local_flow.", "flow_acoustic_in_projection."))
    }
    torch.manual_seed(2)
    target = _TinyEasyMagpie()
    initial_target_state = copy.deepcopy(target.state_dict())

    report = transfer_compatible_pretrained_state(target, source_state)
    target_state = target.state_dict()

    for key, value in target_state.items():
        if key.startswith(("local_flow.", "flow_acoustic_in_projection.")):
            torch.testing.assert_close(value, initial_target_state[key])
        else:
            torch.testing.assert_close(value, source_state[key])

    assert report["semantic_projection_rows_copied"] == 6
    assert report["projection_rows_copied"] == 6
    assert report["acoustic_projection_rows_copied"] == 0
    assert report["acoustic_projection_rows_left_random"] == 0
    assert report["compatible_tensor_count"] == len(target_state) - 4
    assert report["target_state_tensor_keys_left_random"] == [
        "local_flow.bias",
        "local_flow.weight",
    ]
    assert report["target_state_tensor_keys_left_zero_initialized"] == [
        "flow_acoustic_in_projection.bias",
        "flow_acoustic_in_projection.weight",
    ]
    assert report["target_state_tensor_keys_left_initialized"] == [
        "flow_acoustic_in_projection.bias",
        "flow_acoustic_in_projection.weight",
        "local_flow.bias",
        "local_flow.weight",
    ]


def test_accepts_lightning_model_prefix_and_backbone_name():
    source = _TinyEasyMagpie()
    source_state = {
        (f"model.backbone.{key.removeprefix('decoder.')}" if key.startswith("decoder.") else f"model.{key}"): value
        for key, value in source.state_dict().items()
    }
    target = _TinyEasyMagpie()

    transfer_compatible_pretrained_state(target, source_state)

    for key, value in target.decoder.state_dict().items():
        torch.testing.assert_close(value, source.decoder.state_dict()[key])


def test_validates_all_backbone_shapes_before_copying():
    source = _TinyEasyMagpie()
    target = _TinyEasyMagpie()
    initial_target_state = copy.deepcopy(target.state_dict())
    source_state = dict(source.state_dict())
    source_state["decoder.0.weight"] = torch.empty(5, 4)

    with pytest.raises(ValueError, match="Backbone shape mismatch"):
        transfer_compatible_pretrained_state(target, source_state)

    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, initial_target_state[key])


def test_copies_only_semantic_projection_rows_from_legacy_full_head():
    source = _TinyEasyMagpie()
    target = _TinyEasyMagpie()
    source_state = dict(source.state_dict())
    source_state["final_proj.weight"] = torch.randn(12, 4)
    source_state["final_proj.bias"] = torch.randn(12)

    report = transfer_compatible_pretrained_state(target, source_state)

    torch.testing.assert_close(target.final_proj.weight, source_state["final_proj.weight"][:6])
    torch.testing.assert_close(target.final_proj.bias, source_state["final_proj.bias"][:6])
    assert report["projection_rows_copied"] == 6
    assert report["acoustic_projection_rows_copied"] == 0
    assert report["acoustic_projection_rows_left_random"] == 0


def test_rejects_incompatible_semantic_vocabulary_packing():
    source = _TinyEasyMagpie()
    target = _TinyEasyMagpie()
    source_state = dict(source.state_dict())
    source_state["final_proj.weight"] = torch.empty(13, 4)
    source_state["final_proj.bias"] = torch.empty(13)

    with pytest.raises(ValueError, match="not divisible"):
        transfer_compatible_pretrained_state(target, source_state)


def test_materializes_projection_conditioned_decoder_and_context_inputs():
    torch.manual_seed(5)
    target = _TinyProjectionConditionedTarget()
    initial_state = copy.deepcopy(target.state_dict())
    decoder_weight = torch.arange(20, dtype=torch.float32).view(4, 5) / 10
    context_weight = decoder_weight + 2
    decoder_bias = torch.arange(4, dtype=torch.float32)
    context_bias = decoder_bias + 4
    source_state = {
        "decoder.weight": torch.randn_like(target.decoder.weight),
        "decoder.bias": torch.randn_like(target.decoder.bias),
        "final_proj.weight": torch.randn_like(target.final_proj.weight),
        "final_proj.bias": torch.randn_like(target.final_proj.bias),
        "decoder_code_proj.weight": decoder_weight,
        "decoder_code_proj.bias": decoder_bias,
        "context_code_proj.weight": context_weight,
        "context_code_proj.bias": context_bias,
        "audio_bos_emb": torch.full((4,), 11.0),
        "audio_eos_emb": torch.full((4,), 12.0),
        "context_bos_emb": torch.full((4,), 13.0),
        "context_eos_emb": torch.full((4,), 14.0),
    }

    report = transfer_compatible_pretrained_state(
        target,
        source_state,
        projection_conditioned_source=True,
    )
    state = target.state_dict()
    semantic_rows = _TinyCodecHelper.semantic_rows

    torch.testing.assert_close(
        state["audio_embeddings.0.weight"][:3],
        semantic_rows @ decoder_weight[:, :2].T + decoder_bias,
    )
    torch.testing.assert_close(
        state["flow_context_audio_embeddings.0.weight"][:3],
        semantic_rows @ context_weight[:, :2].T + context_bias,
    )
    torch.testing.assert_close(state["flow_acoustic_in_projection.weight"], decoder_weight[:, 2:])
    torch.testing.assert_close(state["flow_context_acoustic_in_projection.weight"], context_weight[:, 2:])
    assert torch.count_nonzero(state["flow_acoustic_in_projection.bias"]) == 0
    assert torch.count_nonzero(state["flow_context_acoustic_in_projection.bias"]) == 0
    torch.testing.assert_close(state["audio_embeddings.0.weight"][3], source_state["audio_bos_emb"])
    torch.testing.assert_close(state["audio_embeddings.0.weight"][4], source_state["audio_eos_emb"])
    torch.testing.assert_close(state["flow_context_audio_embeddings.0.weight"][5], source_state["context_bos_emb"])
    torch.testing.assert_close(state["flow_context_audio_embeddings.0.weight"][6], source_state["context_eos_emb"])
    torch.testing.assert_close(state["audio_embeddings.0.weight"][5:], initial_state["audio_embeddings.0.weight"][5:])
    torch.testing.assert_close(
        state["flow_context_audio_embeddings.0.weight"][3:5],
        initial_state["flow_context_audio_embeddings.0.weight"][3:5],
    )
    assert report["projection_conditioned_source"] is True
    assert report["input_projection_semantic_dim"] == 2
    assert report["input_projection_acoustic_dim"] == 3
    assert report["input_projection_special_rows_initialized"] == {
        "decoder": [3, 4],
        "context": [5, 6],
    }
