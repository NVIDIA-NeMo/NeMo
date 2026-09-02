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

"""No-crash pipeline tests for NemotronVoiceChat streaming inference.

The config sweep runs on a tiny random-weight model (CI). One extra case
loads ``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`` (downloaded into the
Hugging Face cache if needed).

Each test verifies only that the pipeline completes without raising — no
output quality checks.

Run from the NeMo repo root::

    CUDA_VISIBLE_DEVICES=0 pytest tests/collections/speechlm2/nemo_inference_pipelines/test_nemotron_voicechat_pipeline_nocrash.py -v -s
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
    NemotronVoicechatInferenceWrapper,
)
from nemo.collections.speechlm2.inference.utils.stepprogressbar import StepProgressBar

MOCK_SYSTEM_PROMPT = "This is a mock prompt for the test"

_TEST_DEFAULTS = {
    "s2s": {
        "llm_engine_type": "native",
        "tts_engine_type": "native",
        "compute_dtype": "float32",
        "deterministic": False,
        "decode_audio": False,
        "use_perception_cache": False,
        "use_perception_cudagraph": False,
        "system_prompt": None,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
    },
    "streaming": {
        "chunk_size_in_secs": 0.08,
        "buffer_size_in_secs": 71 * 0.08,
    },
}

# ---------------------------------------------------------------------------
# Parametrized configs — each entry is a single overrides dict
# ---------------------------------------------------------------------------

# Text-only configs (decode_audio=False): minimal STT-path smoke checks.
_TEXT_CONFIGS = [
    pytest.param({}, id="baseline"),
    pytest.param(
        {"s2s": {"use_perception_cache": True}},
        id="perception_cache",
    ),
    pytest.param({"pad_audio_by_sec": 2}, id="pad_by_sec"),
]

# Audio configs (decode_audio=True): exercises the full STT + TTS pipeline.
_AUDIO_CONFIGS = [
    pytest.param({}, id="baseline"),
    pytest.param(
        {
            "s2s": {"use_perception_cache": True, "system_prompt": MOCK_SYSTEM_PROMPT},
            "streaming": {"chunk_size_in_secs": 0.24},
            "pad_audio_to_sec": 5,
        },
        id="perception_cache_prompt_multiframe_pad_to_sec",
    ),
    pytest.param(
        {
            "s2s": {"top_p": 0.9, "temperature": 0.7, "repetition_penalty": 1.1},
            "pad_silence_ratio": 0.5,
        },
        id="sampling_pad_silence_ratio",
    ),
    pytest.param(
        {
            "s2s": {"use_tts_subword_cache": True, "use_tts_torch_compile": True},
            "pad_audio_by_sec": 2,
        },
        id="tts_optimizations_pad_by_sec",
    ),
    pytest.param(
        {"s2s": {"deterministic": True, "temperature": 0.0}},
        id="deterministic",
    ),
    pytest.param(
        {"s2s": {"profile_timing": True}},
        id="profile_timing",
    ),
]


def _run(pipeline, audio_path):
    progress_bar = StepProgressBar.from_audio_filepaths(
        [audio_path],
        chunk_size_in_secs=pipeline.chunk_size_in_secs,
        pad_audio_to_sec=pipeline.pad_audio_to_sec,
        pad_silence_ratio=pipeline.pad_silence_ratio,
        pad_audio_by_sec=pipeline.pad_audio_by_sec,
    )
    result = pipeline.run([audio_path], progress_bar=progress_bar)
    assert result is not None
    return result


def test_speaker_reference_is_rejected():
    """Cloning from a wav is not a supported inference path."""
    cfg = OmegaConf.create(
        {
            "model_path": "unused",
            "decode_audio": True,
            "speaker_name": "Aria",
            "speaker_reference": "/path/to/speaker.wav",
            "llm_engine_type": "native",
            "tts_engine_type": "native",
        }
    )
    with pytest.raises(ValueError, match="speaker_reference is not supported"):
        NemotronVoicechatInferenceWrapper(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("overrides", _TEXT_CONFIGS)
def test_pipeline_no_crash(build_pipeline, tiny_model_artifacts, overrides):
    """Run the streaming pipeline with various configs and verify it doesn't crash."""
    model_dir, audio_path, _ = tiny_model_artifacts
    pipeline = build_pipeline(
        model_dir, audio_path, tempfile.mkdtemp(prefix="no-crash-text-"), _TEST_DEFAULTS, overrides
    )
    _run(pipeline, audio_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("overrides", _AUDIO_CONFIGS)
def test_pipeline_no_crash_decode_audio(build_pipeline, tiny_model_artifacts, overrides):
    """Run the streaming pipeline with decode_audio=True and verify it doesn't crash."""
    model_dir, audio_path, speaker_name = tiny_model_artifacts
    pipeline = build_pipeline(
        model_dir,
        audio_path,
        tempfile.mkdtemp(prefix="no-crash-audio-"),
        _TEST_DEFAULTS,
        {"s2s": {"decode_audio": True, "speaker_name": speaker_name}},
        overrides,
    )
    _run(pipeline, audio_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_pipeline_function_channel_without_asr(build_pipeline, tiny_function_model_artifacts):
    """The public checkpoint's function channel is distinct from absent ASR."""
    model_dir, audio_path, _ = tiny_function_model_artifacts
    pipeline = build_pipeline(
        model_dir,
        audio_path,
        tempfile.mkdtemp(prefix="no-crash-function-no-asr-"),
        _TEST_DEFAULTS,
        {"s2s": {"decode_audio": False, "force_turn_taking": True}},
    )
    assert pipeline.s2s_model.model.stt_model.use_function_head
    assert not pipeline.s2s_model.model.stt_model.predict_user_text
    assert not pipeline.s2s_model.model.stt_model.cfg.force_turn_taking

    result = pipeline.run([audio_path])
    assert result is not None
    assert result[0].token_asr_text is None
    assert result[0].raw_asr_text is None
    assert result[0].token_function is not None
    assert result[0].raw_function_text is not None
    assert result[0].capabilities.has_function_head
    assert not result[0].capabilities.has_asr_head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_pipeline_no_crash_hf_11b(
    build_pipeline, hf_voicechat_11b, voicechat_audio_path, voicechat_speaker_name
):
    """One native ``pipeline.run()`` on the public 11B: function channel, audio.

    The config sweep above stays on the tiny model so CI does not pay an 11B
    load per case. This is the real-weight smoke check.
    """
    pipeline = build_pipeline(
        hf_voicechat_11b,
        voicechat_audio_path,
        tempfile.mkdtemp(prefix="no-crash-11b-"),
        _TEST_DEFAULTS,
        {
            "s2s": {
                "decode_audio": True,
                "speaker_name": voicechat_speaker_name,
                "system_prompt": MOCK_SYSTEM_PROMPT,
                "force_turn_taking": True,
            }
        },
    )
    assert pipeline.s2s_model.model.stt_model.use_function_head
    assert not pipeline.s2s_model.model.stt_model.predict_user_text
    assert not pipeline.s2s_model.model.stt_model.cfg.force_turn_taking

    result = _run(pipeline, voicechat_audio_path)
    output = result[0]
    assert output.token_asr_text is None
    assert output.raw_asr_text is None
    assert output.token_function is not None
    assert output.raw_function_text is not None
    assert output.capabilities.has_function_head
    assert not output.capabilities.has_asr_head
    audio = output.audio_buffer
    assert audio is not None and audio.numel() > 0
