import pytest

from nemo.collections.asr.parts.utils.speaker_utils import get_new_cursor_for_update


def test_get_new_cursor_handles_all_overlapping_segments():
    frame_start = 0.0
    segment_range_ts = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    cursor, index = get_new_cursor_for_update(frame_start, segment_range_ts)
    assert cursor == 0.0
    assert index == 0
