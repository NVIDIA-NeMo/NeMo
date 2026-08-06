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

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.magpietts.create_easy_magpietts_oneshot_flow_seed import (
    _defer_dataset_setup,
    transfer_backbone_and_semantic_projection,
)


class _TinyEasyMagpie(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_type = "nemotron_h"
        self.decoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.num_semantic_codebooks = 1
        self.frame_stacking_factor = 2
        self.num_all_tokens_per_codebook = 3
        self.final_proj = nn.Linear(4, 12)
        self.random_module = nn.Linear(4, 5)


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


def test_transfers_only_backbone_and_semantic_projection_rows():
    torch.manual_seed(1)
    source = _TinyEasyMagpie()
    torch.manual_seed(2)
    target = _TinyEasyMagpie()
    initial_target_state = copy.deepcopy(target.state_dict())

    report = transfer_backbone_and_semantic_projection(target, source.state_dict())
    target_state = target.state_dict()
    source_state = source.state_dict()
    semantic_rows = 6

    for key in target_state:
        if key.startswith("decoder."):
            torch.testing.assert_close(target_state[key], source_state[key])
    torch.testing.assert_close(
        target_state["final_proj.weight"][:semantic_rows], source_state["final_proj.weight"][:semantic_rows]
    )
    torch.testing.assert_close(
        target_state["final_proj.bias"][:semantic_rows], source_state["final_proj.bias"][:semantic_rows]
    )
    torch.testing.assert_close(
        target_state["final_proj.weight"][semantic_rows:], initial_target_state["final_proj.weight"][semantic_rows:]
    )
    torch.testing.assert_close(
        target_state["final_proj.bias"][semantic_rows:], initial_target_state["final_proj.bias"][semantic_rows:]
    )
    torch.testing.assert_close(target_state["random_module.weight"], initial_target_state["random_module.weight"])
    torch.testing.assert_close(target_state["random_module.bias"], initial_target_state["random_module.bias"])

    assert report["semantic_projection_rows_copied"] == semantic_rows
    assert report["acoustic_projection_rows_left_random"] == 6


def test_accepts_lightning_model_prefix_and_backbone_name():
    source = _TinyEasyMagpie()
    source_state = {
        (f"model.backbone.{key.removeprefix('decoder.')}" if key.startswith("decoder.") else f"model.{key}"): value
        for key, value in source.state_dict().items()
    }
    target = _TinyEasyMagpie()

    transfer_backbone_and_semantic_projection(target, source_state)

    for key, value in target.decoder.state_dict().items():
        torch.testing.assert_close(value, source.decoder.state_dict()[key])


def test_validates_all_backbone_shapes_before_copying():
    source = _TinyEasyMagpie()
    target = _TinyEasyMagpie()
    initial_target_state = copy.deepcopy(target.state_dict())
    source_state = dict(source.state_dict())
    source_state["decoder.0.weight"] = torch.empty(5, 4)

    with pytest.raises(ValueError, match="Backbone shape mismatch"):
        transfer_backbone_and_semantic_projection(target, source_state)

    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, initial_target_state[key])


def test_rejects_incompatible_semantic_vocabulary_packing():
    source = _TinyEasyMagpie()
    target = _TinyEasyMagpie()
    source_state = dict(source.state_dict())
    source_state["final_proj.weight"] = torch.empty(13, 4)
    source_state["final_proj.bias"] = torch.empty(13)

    with pytest.raises(ValueError, match="not divisible"):
        transfer_backbone_and_semantic_projection(target, source_state)
