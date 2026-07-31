import pytest
import torch

from nemo.collections.asr.parts.utils import speaker_utils as su


def test_run_online_segmentation_passes_vad_before_cumulative(monkeypatch):
    captured = {}

    def fake_cursor(frame_start, segment_range_ts):
        return 0.5, len(segment_range_ts)

    def fake_speech_labels(frame_start, buffer_end, vad_timestamps, cumulative_speech_labels, cursor):
        captured["args"] = (frame_start, buffer_end, vad_timestamps, cumulative_speech_labels, cursor)
        return torch.empty(0, 2), torch.empty(0, 2)

    def fake_subsegments(*args, **kwargs):
        return [], [], []

    monkeypatch.setattr(su, "get_new_cursor_for_update", fake_cursor)
    monkeypatch.setattr(su, "get_speech_labels_for_update", fake_speech_labels)
    monkeypatch.setattr(su, "get_online_subsegments_from_buffer", fake_subsegments)

    segmentor = su.OnlineSegmentor(sample_rate=16000)
    segmentor.frame_start = 1.0
    segmentor.buffer_start = 0.0
    segmentor.buffer_end = 2.0

    vad = torch.tensor([[1.0, 2.0]])
    cum = torch.tensor([[0.0, 0.5]])
    segmentor.cumulative_speech_labels = cum

    segmentor.run_online_segmentation(
        audio_buffer=torch.tensor([]),
        vad_timestamps=vad,
        segment_raw_audio=[torch.tensor([1.0])],
        segment_range_ts=[[0.0, 1.5]],
        segment_indexes=[0],
        window=1.5,
        shift=0.75,
