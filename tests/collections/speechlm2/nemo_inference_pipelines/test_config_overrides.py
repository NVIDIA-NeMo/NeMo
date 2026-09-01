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

"""Model config overrides and the logit boosts both backends share."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2.inference.model_wrappers.config_overrides import (
    COMPONENT_OF,
    LLM,
    TTS,
    VLLM_FORCES_TRUE,
    VLLM_IGNORES,
    apply_model_cfg_overrides,
)
from nemo.collections.speechlm2.parts.logit_boosts import LogitBoosts, apply_logit_boosts


@pytest.fixture
def model():
    """VoiceChat stand-in exposing the two config objects overrides land in.

    Real ``DictConfig`` instances, because bridging goes through
    ``OmegaConf.update``, which only accepts an OmegaConf container.
    """
    return SimpleNamespace(
        stt_model=SimpleNamespace(cfg=OmegaConf.create({})),
        tts_model=SimpleNamespace(cfg=OmegaConf.create({})),
    )


@pytest.fixture
def warnings(monkeypatch):
    from nemo.collections.speechlm2.inference.model_wrappers import config_overrides

    recorded: list[str] = []
    monkeypatch.setattr(config_overrides.logging, "warning", recorded.append)
    return recorded


def test_overrides_land_in_the_config_their_consumer_reads(model, warnings):
    """Each key reaches the submodel that reads it, and absent keys are left
    alone so whatever the checkpoint carries stays in effect."""
    model.stt_model.cfg["inference_pad_boost"] = 1.5

    effective = apply_model_cfg_overrides(
        model,
        {
            "inference_user_pad_boost": 0.8,
            "force_turn_taking": True,
        },
        llm_engine_type="native",
        tts_engine_type="native",
    )

    # DuplexSTTModel reads its own cfg.
    assert model.stt_model.cfg["inference_user_pad_boost"] == 0.8
    assert model.stt_model.cfg["force_turn_taking"] is True
    # Untouched by this call, and still reported as the effective value.
    assert model.stt_model.cfg["inference_pad_boost"] == 1.5
    assert effective["inference_pad_boost"] == 1.5
    assert warnings == []


@pytest.mark.parametrize(
    ("overrides", "tts_engine_type", "expected"),
    [
        # It forces codec silence on EOS unconditionally: True is honoured,
        # False cannot be.
        ({"inference_force_speech_silence_on_eos": False}, "vllm_omni", "force_speech_silence"),
        ({"inference_force_speech_silence_on_eos": True}, "vllm_omni", None),
    ],
)
def test_a_backend_reports_exactly_the_keys_it_ignores(model, warnings, overrides, tts_engine_type, expected):
    """No silent no-ops, and no noise about keys that do work."""
    apply_model_cfg_overrides(
        model,
        overrides,
        llm_engine_type="native",
        tts_engine_type=tts_engine_type,
    )

    if expected is None:
        assert warnings == []
    else:
        assert any(expected in warning for warning in warnings)
        assert any(f"tts_engine_type={tts_engine_type}" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("overrides", "engines", "expected"),
    [
        # Wrapper-consumed knobs are reported by the same table as the
        # model-consumed ones, so there is one place to look and one format.
        ({"use_llm_cache": True}, ("vllm_omni", "native"), "use_llm_cache"),
        ({"use_llm_cache": True}, ("native", "native"), None),
        ({"use_tts_torch_compile": True}, ("native", "vllm_omni"), "use_tts_torch_compile"),
        # A falsy value is already a no-op, so it needs no warning.
        ({"use_tts_torch_compile": False}, ("native", "vllm_omni"), None),
    ],
)
def test_wrapper_knobs_report_through_the_same_table(model, warnings, overrides, engines, expected):
    """Ignored wrapper knobs are reported through the same table as bridged keys."""
    llm_engine_type, tts_engine_type = engines
    apply_model_cfg_overrides(
        model,
        overrides,
        llm_engine_type=llm_engine_type,
        tts_engine_type=tts_engine_type,
    )

    if expected is None:
        assert warnings == []
    else:
        assert any(expected in warning for warning in warnings)


def test_every_support_entry_is_well_formed_and_claimed_once():
    """Guards the table itself, which is now the single source of truth.

    A key in two tables would get two different verdicts, and a key bridged
    into a model config must agree with that config's owning component -- both
    are silent inconsistencies rather than crashes.
    """
    tables = (VLLM_IGNORES, VLLM_FORCES_TRUE)

    seen: set[str] = set()
    for table in tables:
        for key, entry in table.items():
            component, why = entry
            assert component in (LLM, TTS), f"{key} names an unknown component {component!r}"
            assert why and not why.endswith("."), f"{key} reason is interpolated mid-sentence"
            assert key not in seen, f"{key} appears in more than one support table"
            seen.add(key)
            # A bridged key must be attributed to the component whose config it
            # is written into, or the warning names the wrong engine type.
            if key in COMPONENT_OF:
                assert COMPONENT_OF[key] == component


@pytest.mark.parametrize("shape", [(1, 2, 5), (5,)])
def test_logit_boosts_are_read_and_applied_the_same_way_by_both_runtimes(shape):
    """(B, T, V) for the PyTorch heads, (V,) for the vLLM logits processor."""
    from_mapping = LogitBoosts.agent_from_cfg({"inference_pad_boost": 0.8, "inference_eos_boost": None})
    assert (from_mapping.pad, from_mapping.bos, from_mapping.eos) == (0.8, None, None)

    # The converted Nemotron reads an HF PretrainedConfig, which has no .get().
    from_attrs = LogitBoosts.user_from_cfg(
        SimpleNamespace(
            inference_user_pad_boost=0.5,
            inference_user_bos_boost=None,
            inference_user_eos_boost=1.5,
        )
    )
    assert (from_attrs.pad, from_attrs.bos, from_attrs.eos) == (0.5, None, 1.5)

    # Matches the truthiness gate the model heads have always used.
    assert not LogitBoosts.agent_from_cfg({"inference_pad_boost": 0.0})

    logits = torch.zeros(*shape)
    apply_logit_boosts(logits, LogitBoosts(pad=0.5, eos=1.5), pad_id=0, bos_id=1, eos_id=2)
    assert logits.reshape(-1, 5)[0].tolist() == [0.5, 0.0, 1.5, 0.0, 0.0]

    # Empty boosts touch nothing and need no token ids.
    untouched = torch.zeros(*shape)
    apply_logit_boosts(untouched, LogitBoosts(), pad_id=None, bos_id=None, eos_id=None)
    assert untouched.reshape(-1, 5)[0].tolist() == [0.0] * 5
