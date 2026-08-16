# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.speechlm2.parts import pretrained


def test_setup_speech_encoder_hydrates_missing_config_without_weights():
    model = SimpleNamespace(
        cfg=DictConfig(
            {
                "pretrained_asr": "fake-asr",
                "perception": {
                    "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                    "output_dim": 1,
                    "modality_adapter": {"output_dim": 1},
                },
            }
        ),
        llm=SimpleNamespace(config=SimpleNamespace(hidden_size=8)),
    )
    asr_cfg = DictConfig(
        {
            "preprocessor": {"_target_": "fake.Preprocessor"},
            "encoder": {"d_model": 4, "n_layers": 2},
        }
    )

    with (
        patch.object(pretrained, "load_pretrained_nemo_config", return_value=asr_cfg) as load_config,
        patch.object(pretrained, "AudioPerceptionModule") as perception,
    ):
        pretrained.setup_speech_encoder(model, pretrained_weights=False)

    load_config.assert_called_once_with(pretrained.ASRModel, "fake-asr")
    perception.assert_called_once()
    assert model.cfg.perception.preprocessor._target_ == "fake.Preprocessor"
    assert model.cfg.perception.encoder.n_layers == 2
    assert model.cfg.perception.output_dim == 8
    assert model.cfg.perception.modality_adapter.output_dim == 8


def test_setup_parallel_expert_encoder_mounts_two_branch_bridge_and_disables_outer_normalization():
    pe_encoder = torch.nn.Linear(1, 1)
    pe_encoder.d_model = 32
    pe_encoder.n_spk = 4
    pe_encoder._feat_in = 80
    pe_encoder.freeze_asr = False
    pe_encoder.freeze_diar = True
    pe_encoder.spk_kernel_scale = 1.0
    pe_encoder.asr_encoder_type = "transformer"
    pe_encoder.asr_chunk_size_seconds = None
    pe_encoder.diar_chunk_size_seconds = None
    pe_encoder._validate_chunk_size = lambda _name, value: value
    pe_encoder.asr_encoder = SimpleNamespace(sync_max_audio_length=True)
    pe_encoder.diarization_model = SimpleNamespace(
        encoder=SimpleNamespace(sync_max_audio_length=True)
    )

    perception = SimpleNamespace(
        encoder=SimpleNamespace(d_model=32),
        preprocessor=SimpleNamespace(
            featurizer=SimpleNamespace(normalize="per_feature", hop_length=160, sample_rate=16000)
        ),
        spec_augmentation=None,
        proj=torch.nn.Identity(),
    )
    model = SimpleNamespace(
        cfg=DictConfig(
            {
                "pe_encoder_path": "fake/pee-two-branch",
                "perception": {
                    "preprocessor": {"features": 80, "normalize": "per_feature"},
                    "modality_adapter": {"d_model": 32},
                },
            }
        ),
        perception=perception,
    )

    with patch.object(pretrained.ParallelExpertEncoderPT, "load_from_nemo", return_value=pe_encoder) as load:
        pretrained.setup_parallel_expert_encoder(model)

    load.assert_called_once_with("fake/pee-two-branch", map_location="cpu", strict=True)
    assert model.perception.encoder is pe_encoder
    assert model.perception.preprocessor.featurizer.normalize is None
    assert model.cfg.perception.preprocessor.normalize is None
    assert pe_encoder.frame_shift_seconds == pytest.approx(0.01)
    assert pe_encoder.asr_encoder.sync_max_audio_length is False
    assert pe_encoder.diarization_model.encoder.sync_max_audio_length is False

    model.cfg.pe_encoder_path = None
    model.cfg.pe_encoder_config = {"target": "fake.ParallelExpertEncoderPT"}
    model.cfg.pretrained_weights = False
    model.perception.encoder = SimpleNamespace(d_model=32)
    with patch.object(
        pretrained.ParallelExpertEncoderPT, "from_inline_config", return_value=pe_encoder
    ) as from_inline:
        pretrained.setup_parallel_expert_encoder(model)

    from_inline.assert_called_once_with(model.cfg.pe_encoder_config, map_location="cpu")
    assert model.perception.encoder is pe_encoder
