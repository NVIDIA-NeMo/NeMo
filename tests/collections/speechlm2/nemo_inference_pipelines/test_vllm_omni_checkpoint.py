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

"""Wrapper-checkpoint assembly on the filesystem.

A real vLLM/vLLM ``pipeline.run()`` lives in
``test_nemotron_voicechat_pipeline_vllm.py``.
"""

from types import SimpleNamespace

import pytest

from nemo.collections.speechlm2.inference.vllm_omni.checkpoint import build_wrapper_checkpoint


def _stub_source_and_partial_wrapper(tmp_path, ready_component):
    """A stub source checkpoint plus a wrapper with one component already built."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"source")

    wrapper = tmp_path / "wrapper"
    (wrapper / ready_component).mkdir(parents=True)
    (wrapper / ready_component / "config.json").write_text("{}")
    (wrapper / ready_component / "model.safetensors").write_bytes(b"ready")
    (wrapper / "config.json").write_text('{"model_type": "nemotron_voicechat"}')
    return source, wrapper


@pytest.mark.parametrize("component", ["nemotron", "eartts"])
def test_wrapper_checkpoint_converts_only_the_requested_component(tmp_path, component):
    """Reusing a ready component must not drag the other one along."""
    source, wrapper = _stub_source_and_partial_wrapper(tmp_path, component)

    result = build_wrapper_checkpoint(
        str(source),
        str(wrapper),
        include_nemotron=component == "nemotron",
        include_eartts=component == "eartts",
    )

    assert result == str(wrapper)
    assert not (wrapper / ("eartts" if component == "nemotron" else "nemotron")).exists()
    # Records which source it came from, which is what makes it verifiable.
    assert (wrapper / ".nemo_source.json").is_file()


def test_wrapper_checkpoint_refuses_to_extend_an_unverified_partial_wrapper(tmp_path):
    """A wrapper with no ``.nemo_source.json`` could have come from any source,
    so adding a second component to it might silently mix two checkpoints.
    """
    source, wrapper = _stub_source_and_partial_wrapper(tmp_path, "nemotron")

    with pytest.raises(ValueError, match="Cannot safely add a component"):
        build_wrapper_checkpoint(str(source), str(wrapper), include_nemotron=False, include_eartts=True)


def test_converter_applies_voicechat_special_token_overrides():
    """The converted config must carry VoiceChat's BOS/EOS/PAD, not the LLM
    backbone's; otherwise conversion succeeds with incorrect system-prompt
    prefill token IDs.
    """
    from nemo.collections.speechlm2.inference.vllm_omni.scripts.convert_duplex_stt_checkpoint import (
        _apply_source_special_tokens,
    )

    class FakeTokenizer:
        def __init__(self):
            self.vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<SPECIAL_12>": 12}

        def get_vocab(self):
            return self.vocab

        def add_special_tokens(self, values):
            for name, token in values.items():
                setattr(self, name, token)
            return 0

        def convert_tokens_to_ids(self, token):
            return self.vocab[token]

    config = SimpleNamespace(bos_token_id=1, eos_token_id=12, pad_token_id=0)
    source = {
        "model": {
            "stt": {
                "model": {
                    "override_tokens": {
                        "bos_token": "<s>",
                        "eos_token": "</s>",
                        "pad_token": "<SPECIAL_12>",
                    }
                }
            }
        }
    }

    _apply_source_special_tokens(config, FakeTokenizer(), source)

    assert (config.bos_token_id, config.eos_token_id, config.pad_token_id) == (1, 2, 12)
