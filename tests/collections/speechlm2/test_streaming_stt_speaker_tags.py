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

"""SOT ``<spk:N>`` tag emission in the interleaved per-chunk targets."""

import re

import pytest

from nemo.collections.speechlm2.data.streaming_stt_dataset import get_llm_messages_for_sample
from nemo.collections.speechlm2.parts.alignments import WordAlignment

TAG = re.compile(r"<spk:(\d+)>")


def _align(spec, step=0.2):
    """``spec`` is [(word, speaker), ...] laid out on a regular grid."""
    return [WordAlignment(w, i * step, i * step + step * 0.9, speaker=s) for i, (w, s) in enumerate(spec)]


def _turns(alignments, transcript, *, chunk_size=2, write_token="", prepend=False, template="<spk:{i}>"):
    messages = get_llm_messages_for_sample(
        system_role="system",
        system_prompt="Transcribe.",
        audio_tag="<audio>",
        blank_token="<blank>",
        chunk_size=chunk_size,
        num_delay_frames=0,
        audio_duration_secs=len(alignments) * 0.2 + 0.5,
        frame_length_in_secs=0.08,
        alignments=alignments,
        transcript=transcript,
        words_per_group=1,
        prepend_write_token=prepend,
        write_token=write_token,
        speaker_token_template=template,
    )
    return [m["content"] for m in messages if m["role"] == "assistant" and m["content"] != "<blank>"]


class TestSpeakerTagEmission:
    @pytest.mark.unit
    def test_tag_emitted_only_on_speaker_change(self):
        al = _align([("a", 0), ("b", 0), ("c", 1), ("d", 1)])
        out = " ".join(_turns(al, "<spk:0> a b <spk:1> c d"))
        assert TAG.findall(out) == ["0", "1"], "one tag per change, not per word"

    @pytest.mark.unit
    def test_emitted_tag_sequence_matches_the_transcript(self):
        transcript = "<spk:0> a <spk:1> b <spk:0> c d <spk:2> e"
        al = _align([("a", 0), ("b", 1), ("c", 0), ("d", 0), ("e", 2)])
        assert TAG.findall(" ".join(_turns(al, transcript))) == TAG.findall(transcript)

    @pytest.mark.unit
    def test_write_token_stays_outermost(self):
        # Q11: `prepend_write_token` exists so the LM's first output token is a binary blank/write
        # decision. A tag outside it would make that distribution multi-modal.
        al = _align([("a", 0), ("b", 1)])
        out = _turns(al, "<spk:0> a <spk:1> b", write_token="<|write|>", prepend=True)
        tagged = [c for c in out if "<spk:" in c]
        assert tagged, "expected at least one tagged turn"
        assert all(c.startswith("<|write|><spk:") for c in tagged)

    @pytest.mark.unit
    def test_no_double_space_after_an_injected_tag(self):
        al = _align([("alpha", 0), ("beta", 1)])
        for content in _turns(al, "<spk:0> alpha <spk:1> beta"):
            assert "  " not in content

    @pytest.mark.unit
    def test_template_none_disables_tagging(self):
        al = _align([("a", 0), ("b", 1)])
        out = " ".join(_turns(al, "<spk:0> a <spk:1> b", template=None))
        assert "<spk:" not in out

    @pytest.mark.unit
    def test_words_without_speaker_are_untagged(self):
        # Single-speaker manifests carry no `speaker_ids`; the path must stay inert.
        al = [WordAlignment("a", 0.0, 0.2), WordAlignment("b", 0.3, 0.5)]
        assert "<spk:" not in " ".join(_turns(al, "a b"))

    @pytest.mark.unit
    def test_state_follows_the_last_word_of_a_group(self):
        # A group's transcript slice can already carry a mid-group tag, so the "last emitted
        # speaker" is the group's LAST word, not its first. Tracking the first would re-emit a
        # redundant tag (or drop a needed one) on the following group.
        transcript = "<spk:0> a <spk:1> b c"
        al = _align([("a", 0), ("b", 1), ("c", 1)])
        out = " ".join(_turns(al, transcript, chunk_size=40))  # force one big group
        assert TAG.findall(out) == ["0", "1"]
