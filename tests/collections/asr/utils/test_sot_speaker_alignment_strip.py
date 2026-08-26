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

import pytest

from nemo.collections.asr.parts.utils.sot_speaker_alignment import strip_speaker_tags


class TestStripSpeakerTags:
    """``strip_speaker_tags`` feeds forced alignment, which mangles ``<spk:N>`` into a phantom
    word, so the tags must come out and the per-word speaker mapping must survive exactly."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_text,expected_ids",
        [
            ("<spk:0> hello there <spk:1> hi", "hello there hi", [0, 0, 1]),
            ("<spk:0> a <spk:1> b <spk:0> c d", "a b c d", [0, 1, 0, 0]),
            ("<spk:2> only one speaker here", "only one speaker here", [2, 2, 2, 2]),
            # word-level SOT: a tag before every word
            ("<spk:0> a <spk:0> b <spk:1> c", "a b c", [0, 0, 1]),
            # irregular whitespace around tags must not create empty words
            ("<spk:0>   spaced   out  <spk:1>   words ", "spaced out words", [0, 0, 1]),
        ],
    )
    def test_splits_text_and_speaker_ids(self, text, expected_text, expected_ids):
        tag_free, ids = strip_speaker_tags(text)
        assert tag_free == expected_text
        assert ids == expected_ids

    @pytest.mark.unit
    def test_output_is_tag_free_and_parallel(self):
        text = "<spk:0> one two <spk:3> three <spk:0> four five six"
        tag_free, ids = strip_speaker_tags(text)
        assert "<spk:" not in tag_free
        assert len(tag_free.split()) == len(ids), "speaker ids must be parallel to words"

    @pytest.mark.unit
    def test_untagged_text_raises_rather_than_desynchronising(self):
        # `parse_speaker_tokens` drops words before the first tag, so untagged text would yield
        # 0 ids for N words. Raising is the safe contract: a silently empty mapping would let
        # every word be attributed to speaker 0. Callers must gate on `has_speaker_tokens` first.
        with pytest.raises(ValueError, match="desynchronised"):
            strip_speaker_tags("no tags at all")

    @pytest.mark.unit
    def test_leading_untagged_words_raise(self):
        # Words before the first tag are dropped by `parse_speaker_tokens`, so this is the same
        # desynchronisation hazard in its subtler form.
        with pytest.raises(ValueError, match="desynchronised"):
            strip_speaker_tags("stray words <spk:0> then tagged")
