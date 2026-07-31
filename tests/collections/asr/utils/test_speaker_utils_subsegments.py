import pytest

from nemo.collections.asr.parts.utils.speaker_utils import get_subsegments


def test_get_subsegments_returns_segments_between_shift_and_window():
    result = get_subsegments(
        offset=0.0,
        window=2.0,
        shift=0.5,
        duration=0.6,
        min_subsegment_duration=0.01,
        decimals=2,
    )
    assert len(result) == 1
    start, dur = result[0]
    assert start == pytest.approx(0.0)
    assert dur == pytest.approx(0.6)
