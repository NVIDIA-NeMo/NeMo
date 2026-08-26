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

"""Tests for the collapsed-word spreading in ``scripts/speechlm2/align_manifest.py``."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "speechlm2" / "align_manifest.py"


def _load_module():
    """Import the script without its optional aligner backend (``qwen_asr`` is env-specific)."""
    sys.modules.setdefault("qwen_asr", types.ModuleType("qwen_asr"))
    spec = importlib.util.spec_from_file_location("align_manifest_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words(*spans):
    return [{"text": f"w{i}", "start_time": s, "end_time": e} for i, (s, e) in enumerate(spans)]


class TestSpreadCollapsedWords:
    """The aligner collapses runs of words to one instant at speaker changes. Downstream a chunk is
    picked with ``ceil(end_time / frame_length)``, so a collapsed run is emitted in a single chunk
    unless it is spread out."""

    @pytest.mark.unit
    def test_spreads_a_run_into_the_following_gap(self):
        m = _load_module()
        al = _words((1.0, 1.2), (1.6, 1.6), (1.6, 1.6), (1.6, 1.6), (2.1, 2.4))
        assert m.spread_collapsed_words(al, duration=10.0, min_step=0.08) == 3
        assert [round(w["end_time"], 2) for w in al[1:4]] == [1.68, 1.76, 1.84]
        # never pushed past the following word
        assert al[3]["end_time"] <= al[4]["start_time"] + 1e-9

    @pytest.mark.unit
    def test_run_at_end_of_cut_uses_duration_as_the_bound(self):
        m = _load_module()
        al = _words((9.0, 9.1), (9.5, 9.5), (9.5, 9.5))
        assert m.spread_collapsed_words(al, duration=10.0, min_step=0.08) == 2
        assert al[-1]["end_time"] <= 10.0

    @pytest.mark.unit
    def test_no_room_leaves_alignments_untouched(self):
        # Spreading must never break monotonicity to manufacture a duration.
        m = _load_module()
        al = _words((3.0, 3.0), (3.0, 3.4))
        assert m.spread_collapsed_words(al, duration=10.0, min_step=0.08) == 0
        assert al == _words((3.0, 3.0), (3.0, 3.4))

    @pytest.mark.unit
    def test_step_is_capped_so_words_do_not_become_absurdly_long(self):
        m = _load_module()
        al = _words((1.0, 1.0), (9.0, 9.5))  # 8 s of room for one collapsed word
        m.spread_collapsed_words(al, duration=10.0, min_step=0.08)
        assert al[0]["end_time"] - al[0]["start_time"] == pytest.approx(0.08)

    @pytest.mark.unit
    def test_already_spaced_words_are_untouched(self):
        m = _load_module()
        al = _words((0.0, 0.3), (0.4, 0.7), (0.8, 1.1))
        before = [dict(w) for w in al]
        assert m.spread_collapsed_words(al, duration=5.0, min_step=0.08) == 0
        assert al == before

    @pytest.mark.unit
    def test_output_stays_monotonic(self):
        m = _load_module()
        al = _words((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.6, 0.9), (1.2, 1.2), (1.4, 1.6))
        m.spread_collapsed_words(al, duration=5.0, min_step=0.08)
        starts = [w["start_time"] for w in al]
        assert starts == sorted(starts)
        assert all(w["end_time"] >= w["start_time"] for w in al)
