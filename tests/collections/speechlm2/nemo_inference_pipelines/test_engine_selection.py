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

"""What the s2s config resolves to, and which native weights that skips."""

import pytest

from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import (
    native_weight_skip_prefixes,
    reject_unimplemented_vllm,
    resolve_engine_types,
)

NATIVE, VLLM = "native", "vllm_omni"


@pytest.mark.parametrize(
    ("cfg", "engines", "skipped", "not_skipped"),
    [
        ({}, (NATIVE, NATIVE), set(), {"stt_model.llm.", "tts_model.tts_model."}),
        (
            {"llm_engine_type": VLLM, "tts_engine_type": VLLM},
            (VLLM, VLLM),
            {"stt_model.llm.", "tts_model.tts_model."},
            set(),
        ),
        (
            {"llm_engine_type": NATIVE, "tts_engine_type": VLLM},
            (NATIVE, VLLM),
            {"tts_model.tts_model."},
            {"stt_model.llm."},
        ),
        (
            {"tts_engine_type": VLLM},
            (NATIVE, VLLM),
            {"tts_model.tts_model."},
            {"stt_model.llm."},
        ),
        (
            {"llm_engine_type": VLLM, "tts_engine_type": None},
            (VLLM, NATIVE),
            {"stt_model.llm."},
            {"tts_model.tts_model."},
        ),
    ],
)
def test_config_resolves_to_backends_and_the_weights_they_skip(cfg, engines, skipped, not_skipped):
    """Each component key is independent; omitted keys default to native.

    A component on vLLM must also skip loading its native weights, so the two
    decisions are checked together -- disagreeing would waste a full weight
    load or, worse, leave a component with no weights at all.
    """
    assert resolve_engine_types(cfg) == engines

    prefixes = native_weight_skip_prefixes(*engines)
    assert skipped <= prefixes
    assert prefixes.isdisjoint(not_skipped)
    # The auxiliary RNN-T decoder is never needed by the streaming path.
    assert {"stt_model.rnnt_decoder.", "stt_model.rnnt_joint."} <= prefixes


def test_unusable_engine_selection_is_named():
    with pytest.raises(ValueError, match="llm_engine_type='other'"):
        resolve_engine_types({"llm_engine_type": "other"})
    with pytest.raises(ValueError, match="not a config key"):
        resolve_engine_types({"engine_type": VLLM})


def test_vllm_selection_is_named_as_not_implemented():
    """vllm_omni stays a legal config value so the native loop is already the
    combined-form loop; constructing that backend is rejected rather than
    silently running native."""
    with pytest.raises(NotImplementedError, match="not implemented in this PR"):
        reject_unimplemented_vllm(VLLM, NATIVE)
    with pytest.raises(NotImplementedError, match="not implemented in this PR"):
        reject_unimplemented_vllm(NATIVE, VLLM)

    from nemo.collections.speechlm2.inference.model_wrappers.backend.vllm.llm import VllmLLM

    with pytest.raises(NotImplementedError, match="not implemented in this PR"):
        VllmLLM()
