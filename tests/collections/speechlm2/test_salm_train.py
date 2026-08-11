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

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest


_SALM_TRAIN_PATH = Path(__file__).parents[3] / "examples" / "speechlm2" / "salm_train.py"
_SPEC = importlib.util.spec_from_file_location("salm_train_for_test", _SALM_TRAIN_PATH)
_SALM_TRAIN = importlib.util.module_from_spec(_SPEC)
with patch("torch.cuda.is_available", return_value=False):
    _SPEC.loader.exec_module(_SALM_TRAIN)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("precision", "use_nemo_automodel", "compile_enabled", "activation_checkpointing_llm", "expected"),
    [
        ("bf16-true", True, True, True, True),
        ("bf16-true", False, True, True, False),
        ("bf16-true", True, False, True, False),
        ("bf16-true", True, True, False, False),
        ("bf16-flash", True, True, True, False),
    ],
)
def test_pin_bf16_default_dtype_only_for_compiled_activation_recompute(
    precision, use_nemo_automodel, compile_enabled, activation_checkpointing_llm, expected
):
    cfg = _SALM_TRAIN.OmegaConf.create(
        {
            "model": {
                "use_nemo_automodel": use_nemo_automodel,
                "compile": {"enabled": compile_enabled},
            },
            "trainer": {
                "precision": precision,
                "strategy": {"activation_checkpointing_llm": activation_checkpointing_llm},
            },
        }
    )

    assert _SALM_TRAIN._should_pin_bf16_default_dtype(cfg) is expected


@pytest.mark.unit
def test_create_salm_dataset_omits_unset_multispeaker_config(monkeypatch):
    class LegacySALMDataset:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    tokenizer = object()
    monkeypatch.setattr(_SALM_TRAIN, "SALMDataset", LegacySALMDataset)

    dataset = _SALM_TRAIN._create_salm_dataset(tokenizer, {})

    assert dataset.tokenizer is tokenizer


@pytest.mark.unit
def test_create_salm_dataset_forwards_configured_multispeaker_config(monkeypatch):
    multispeaker_cfg = {"num_speakers": 2}

    class MultiSpeakerSALMDataset:
        def __init__(self, tokenizer, multispeaker_cfg=None):
            self.tokenizer = tokenizer
            self.multispeaker_cfg = multispeaker_cfg

    tokenizer = object()
    monkeypatch.setattr(_SALM_TRAIN, "SALMDataset", MultiSpeakerSALMDataset)

    dataset = _SALM_TRAIN._create_salm_dataset(tokenizer, {"multispeaker_cfg": multispeaker_cfg})

    assert dataset.tokenizer is tokenizer
    assert dataset.multispeaker_cfg is multispeaker_cfg


@pytest.mark.unit
def test_train_uses_compatible_dataset_factory(monkeypatch, tmp_path):
    tokenizer = object()
    dataset = object()
    calls = []

    class FakeSALM:
        def __init__(self, model_cfg):
            self.tokenizer = tokenizer

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = []

        def init_module(self):
            return nullcontext()

        def fit(self, model, datamodule):
            pass

    class FakeDataModule:
        def __init__(self, data_cfg, tokenizer, dataset):
            pass

    def create_salm_dataset(tokenizer_arg, data_cfg):
        calls.append((tokenizer_arg, data_cfg))
        return dataset

    monkeypatch.setattr(_SALM_TRAIN, "SALM", FakeSALM)
    monkeypatch.setattr(_SALM_TRAIN, "Trainer", FakeTrainer)
    monkeypatch.setattr(_SALM_TRAIN, "DataModule", FakeDataModule)
    monkeypatch.setattr(_SALM_TRAIN, "_create_salm_dataset", create_salm_dataset)
    monkeypatch.setattr(_SALM_TRAIN, "seed_everything", lambda seed: None)
    monkeypatch.setattr(_SALM_TRAIN, "resolve_trainer_cfg", lambda cfg: {})
    monkeypatch.setattr(_SALM_TRAIN, "exp_manager", lambda trainer, cfg: tmp_path)
    monkeypatch.setattr(_SALM_TRAIN.OmegaConf, "save", lambda cfg, path: None)
    monkeypatch.setattr(_SALM_TRAIN.torch.cuda, "is_available", lambda: False)

    cfg = _SALM_TRAIN.OmegaConf.create(
        {
            "data": {"train_ds": {"seed": 0}},
            "model": {},
            "trainer": {},
        }
    )
    _SALM_TRAIN.train.__wrapped__(cfg)

    assert calls == [(tokenizer, cfg.data)]
