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
