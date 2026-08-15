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

import pytest
import torch

from nemo.collections.tts.modules.magpietts_diffusion import OneShotLocalDiffusion
from nemo.collections.tts.modules.magpietts_flow import OneShotLocalFlow
from nemo.collections.tts.modules.magpietts_flow_matching import (
    OneShotLocalFlowMatching,
)
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType
from nemo.collections.tts.modules.magpietts_oneshot import (
    OneShotLocalPredictor,
    create_oneshot_local_predictor,
)


pytestmark = pytest.mark.unit


def test_normalizing_flow_implements_oneshot_predictor_contract():
    predictor = create_oneshot_local_predictor(
        "normalizing_flow",
        acoustic_channels=12,
        condition_channels=20,
        cfg={
            "local_flow_hidden_dim": 16,
            "local_flow_n_layers": 2,
            "local_flow_n_flows": 2,
        },
    )

    assert isinstance(predictor, OneShotLocalPredictor)
    assert isinstance(predictor, OneShotLocalFlow)

    condition = torch.randn(2, 20, 5)
    lengths = torch.tensor([5, 3])
    torch.manual_seed(42)
    prediction = predictor.predict(condition, lengths)
    torch.manual_seed(42)
    legacy_prediction = predictor.sample(condition, lengths)

    assert prediction.shape == (2, 12, 5)
    assert torch.equal(prediction, legacy_prediction)
    assert torch.count_nonzero(prediction[1, :, 3:]) == 0


def test_flow_matching_implements_oneshot_predictor_contract():
    predictor = create_oneshot_local_predictor(
        "flow_matching",
        acoustic_channels=12,
        condition_channels=20,
        cfg={
            "local_flow_matching_hidden_dim": 16,
            "local_flow_matching_n_layers": 2,
            "local_flow_matching_time_embedding_dim": 8,
            "local_flow_matching_inference_steps": 2,
            "local_flow_matching_train_num_noise_samples": 4,
        },
    )

    assert isinstance(predictor, OneShotLocalPredictor)
    assert isinstance(predictor, OneShotLocalFlowMatching)
    assert predictor.inference_steps == 2
    assert predictor.solver == "midpoint"
    assert predictor.num_noise_samples == 4


def test_diffusion_implements_oneshot_predictor_contract():
    predictor = create_oneshot_local_predictor(
        "diffusion",
        acoustic_channels=12,
        condition_channels=20,
        cfg={
            "local_diffusion_hidden_dim": 16,
            "local_diffusion_n_layers": 2,
            "local_diffusion_time_embedding_dim": 8,
            "local_diffusion_training_timesteps": 20,
            "local_diffusion_inference_steps": 4,
        },
    )

    assert isinstance(predictor, OneShotLocalPredictor)
    assert isinstance(predictor, OneShotLocalDiffusion)
    assert predictor.training_timesteps == 20
    assert predictor.inference_steps == 4
    assert predictor.ddim_eta == 0.0


def test_local_transformer_type_identifies_oneshot_predictors():
    assert LocalTransformerType.FLOW.is_oneshot
    assert LocalTransformerType.FLOW_MATCHING.is_oneshot
    assert LocalTransformerType.DIFFUSION.is_oneshot
    assert not LocalTransformerType.AR.is_oneshot
    assert not LocalTransformerType.NO_LT.is_oneshot
