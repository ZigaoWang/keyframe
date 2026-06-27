"""Tests for the online shot segmenter."""
from __future__ import annotations

import numpy as np

from keyframe.config import SegmenterConfig
from keyframe.segmenter import StreamingSegmenter


def _feed(seg: StreamingSegmenter, t: float, vec: list[float]) -> object:
    return seg.ingest(
        source_index=int(t * 30),
        timestamp_sec=t,
        embedding=np.array(vec, dtype=np.float32),
        bgr_path=f"/tmp/f_{t:.2f}.jpg",
        sharpness=100.0,
    )


def test_first_frame_opens_a_shot():
    seg = StreamingSegmenter(SegmenterConfig(sim_threshold=0.9, min_shot_sec=0.0))
    r = _feed(seg, 0.0, [1.0, 0.0, 0.0])
    assert r.sampled.is_shot_start is True
    assert r.closed is None
    assert seg.active_segment is not None
    assert seg.total_segments_so_far == 1


def test_similar_frames_stay_in_one_shot():
    seg = StreamingSegmenter(SegmenterConfig(sim_threshold=0.9, min_shot_sec=0.0))
    _feed(seg, 0.0, [1.0, 0.0, 0.0])
    r = _feed(seg, 1.0, [0.99, 0.01, 0.0])
    assert r.sampled.is_shot_start is False
    assert r.closed is None
    assert len(seg.closed_segments) == 0


def test_dissimilar_frame_opens_new_shot():
    seg = StreamingSegmenter(SegmenterConfig(sim_threshold=0.9, min_shot_sec=0.0))
    _feed(seg, 0.0, [1.0, 0.0, 0.0])
    _feed(seg, 1.0, [1.0, 0.0, 0.0])
    r = _feed(seg, 2.0, [0.0, 1.0, 0.0])
    assert r.sampled.is_shot_start is True
    assert r.closed is not None
    assert r.closed.segment_id == 0
    assert len(seg.closed_segments) == 1


def test_min_shot_sec_blocks_premature_split():
    seg = StreamingSegmenter(SegmenterConfig(sim_threshold=0.9, min_shot_sec=3.0))
    _feed(seg, 0.0, [1.0, 0.0, 0.0])
    r = _feed(seg, 0.5, [0.0, 1.0, 0.0])  # dissimilar, but too soon
    assert r.sampled.is_shot_start is False
    assert r.closed is None


def test_finalise_closes_active_segment():
    seg = StreamingSegmenter(SegmenterConfig(sim_threshold=0.9, min_shot_sec=0.0))
    _feed(seg, 0.0, [1.0, 0.0, 0.0])
    _feed(seg, 1.0, [1.0, 0.0, 0.0])
    closed = seg.finalise(end_sec=2.0)
    assert closed is not None
    assert seg.active_segment is None
    assert len(seg.closed_segments) == 1
    assert seg.closed_segments[0].end_sec == 2.0


def test_ema_caps_drift_in_long_shots():
    seg = StreamingSegmenter(SegmenterConfig(
        sim_threshold=0.9, min_shot_sec=0.0, use_ema=True, ema_alpha=0.25,
    ))
    _feed(seg, 0.0, [1.0, 0.0, 0.0])
    for i in range(1, 6):
        _feed(seg, float(i), [1.0, 0.02 * i, 0.0])
    # still one open shot, none closed
    assert len(seg.closed_segments) == 0
    assert seg.active_segment is not None
