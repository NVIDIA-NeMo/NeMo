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

"""One live vLLM-Omni pipeline run on nvidia/NVIDIA-NemotronLabs-VoiceChat-11B.

Skipped when ``vllm_omni`` is not installed. The 11B snapshot is downloaded
into the Hugging Face cache if needed.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

pytest.importorskip("vllm_omni")

from nemo.collections.speechlm2.inference.utils.stepprogressbar import StepProgressBar

MOCK_SYSTEM_PROMPT = "This is a mock prompt for the test"
_VLLM = "vllm_omni"

_VLLM_DEFAULTS = {
    "s2s": {
        "llm_engine_type": _VLLM,
        "tts_engine_type": _VLLM,
        "deterministic": False,
        "decode_audio": True,
        "system_prompt": MOCK_SYSTEM_PROMPT,
    },
    "streaming": {
        "chunk_size_in_secs": 0.08,
        "buffer_size_in_secs": 71 * 0.08,
    },
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_pipeline_no_crash_vllm(
    build_pipeline, hf_voicechat_11b, voicechat_audio_path, voicechat_speaker_name, real_vllm_omni_wrapper
):
    """vLLM/vLLM ``pipeline.run()`` on the public 11B: text and audio exist."""
    _, wrapper_dir = real_vllm_omni_wrapper
    output_dir = tempfile.mkdtemp(prefix="no-crash-vllm-")

    pipeline = build_pipeline(
        hf_voicechat_11b,
        voicechat_audio_path,
        output_dir,
        _VLLM_DEFAULTS,
        {
            "s2s": {
                "speaker_name": voicechat_speaker_name,
                "vllm_omni_config": {"wrapper_dir": wrapper_dir},
            }
        },
    )
    wrapper = pipeline.s2s_model
    assert wrapper.llm_engine_type == _VLLM
    assert wrapper.tts_engine_type == _VLLM

    progress_bar = StepProgressBar.from_audio_filepaths(
        [voicechat_audio_path],
        chunk_size_in_secs=pipeline.chunk_size_in_secs,
        pad_audio_to_sec=pipeline.pad_audio_to_sec,
        pad_silence_ratio=pipeline.pad_silence_ratio,
        pad_audio_by_sec=pipeline.pad_audio_by_sec,
    )
    result = pipeline.run([voicechat_audio_path], progress_bar=progress_bar)
    assert result is not None
    assert len(result) == 1
    output = result[0]
    assert output.token_text is not None and output.token_text.numel() > 0
    audio = output.audio_buffer
    assert audio is not None and audio.numel() > 0
