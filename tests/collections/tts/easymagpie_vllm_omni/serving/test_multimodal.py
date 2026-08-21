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
from types import SimpleNamespace

import numpy as np
import pytest

from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.multimodal import EasyMagpieAudioParser, EasyMagpieMultiModalProcessor


def test_audio_parser_requires_explicit_codec_format():
    parser = EasyMagpieAudioParser(sample_rate=16000, expected_hidden_size=None)
    waveform = np.zeros(16, dtype=np.float32)

    parsed, sample_rate = parser._get_audio_with_sr((waveform, 16000))
    assert parsed is waveform
    assert sample_rate is None

    with pytest.raises(ValueError, match="16000 Hz; received 22050 Hz"):
        parser._get_audio_with_sr((waveform, 22050))
    with pytest.raises(ValueError, match="must be mono"):
        parser._get_audio_with_sr((np.zeros((2, 16), dtype=np.float32), 16000))
    with pytest.raises(ValueError, match="must be passed as a"):
        parser._get_audio_with_sr(waveform)


class _AudioItems:
    def __init__(self, lengths):
        self.lengths = lengths

    def get_audio_length(self, item_idx):
        return self.lengths[item_idx]

    def get_count(self):
        return len(self.lengths)


def _processor_for_lengths(lengths):
    processor = EasyMagpieMultiModalProcessor.__new__(EasyMagpieMultiModalProcessor)
    processor.info = SimpleNamespace(arch=EasyMagpieOmniArch(codec_encoder_bundled=True))
    return processor, {"audio": _AudioItems(lengths)}


def test_strict_audio_parser_bypasses_vllm_preprocessing_cache(monkeypatch):
    processor = EasyMagpieMultiModalProcessor.__new__(EasyMagpieMultiModalProcessor)
    expected = object()

    def apply_processor(inputs, timing_ctx):
        return expected

    monkeypatch.setattr(processor, "_apply_hf_processor", apply_processor)

    inputs = SimpleNamespace(hf_processor_mm_kwargs={})

    assert processor._cached_apply_hf_processor(inputs, None) is expected


def test_reference_audio_placeholder_uses_actual_codec_row_count():
    processor, mm_items = _processor_for_lengths([32_000])
    update = processor._get_prompt_updates(
        mm_items,
        {},
        None,
    )[
        0
    ].resolve(0)

    assert update.content.full == [1] * 27
