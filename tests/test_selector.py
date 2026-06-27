"""Tests for the per-segment selector and cross-segment diversity filter."""
from __future__ import annotations

import numpy as np

from keyframe.config import SelectorConfig
from keyframe.segmenter import SampledFrame, Segment
from keyframe.selector import select_all_keyframes, select_segment_keyframes


def _frame(idx: int, t: float, embedding: list[float], sharp: float) -> SampledFrame:
    return SampledFrame(
        index=idx,
        source_index=idx * 30,
        timestamp_sec=t,
        embedding=np.array(embedding, dtype=np.float32),
        bgr_path=f"/tmp/f_{idx}.jpg",
        sharpness=sharp,
        sim_to_shot_mean=1.0,
        is_shot_start=(idx == 0),
    )


def _segment(seg_id: int, frames: list[SampledFrame]) -> Segment:
    return Segment(
        segment_id=seg_id,
        start_sec=frames[0].timestamp_sec,
        end_sec=frames[-1].timestamp_sec,
        frames=frames,
    )


def test_single_frame_segment_returns_one_keyframe():
    seg = _segment(0, [_frame(0, 0.0, [1.0, 0.0], 50.0)])
    out = select_segment_keyframes(seg, SelectorConfig(), starting_keyframe_id=1)
    assert len(out) == 1
    assert out[0].keyframe_id == 1


def test_selector_respects_min_temporal_spacing():
    frames = [_frame(i, i * 0.2, [1.0, 0.0], 100.0 - i) for i in range(10)]
    seg = _segment(0, frames)
    cfg = SelectorConfig(
        max_frames_per_segment=3,
        seconds_per_keyframe=0.5,
        min_temporal_spacing_sec=1.0,
        sharpness_weight=1.0,
        representativeness_weight=0.0,
    )
    out = select_segment_keyframes(seg, cfg, starting_keyframe_id=1)
    times = sorted(k.timestamp_sec for k in out)
    for a, b in zip(times, times[1:]):
        assert (b - a) >= cfg.min_temporal_spacing_sec - 1e-9


def test_selector_targets_more_frames_for_long_segments():
    n = 20
    frames = [_frame(i, float(i), [1.0, 0.0], 50.0) for i in range(n)]
    seg = _segment(0, frames)
    cfg = SelectorConfig(max_frames_per_segment=4, seconds_per_keyframe=5.0,
                        min_temporal_spacing_sec=0.5)
    out = select_segment_keyframes(seg, cfg, starting_keyframe_id=1)
    assert 1 < len(out) <= cfg.max_frames_per_segment


def test_diversity_filter_drops_near_duplicate_keyframes():
    s1 = _segment(0, [_frame(0, 0.0, [1.0, 0.0, 0.0], 100.0)])
    s2 = _segment(1, [_frame(1, 5.0, [1.0, 0.0, 0.0], 100.0)])  # duplicate
    s3 = _segment(2, [_frame(2, 10.0, [0.0, 1.0, 0.0], 100.0)])
    cfg = SelectorConfig(diversity_sim_threshold=0.99, min_keyframes=1)
    out = select_all_keyframes([s1, s2, s3], cfg)
    assert 1 <= len(out) < 3


def test_diversity_filter_respects_min_keyframes_floor():
    segs = [_segment(i, [_frame(i, float(i), [1.0, 0.0, 0.0], 100.0)]) for i in range(4)]
    cfg = SelectorConfig(diversity_sim_threshold=0.5, min_keyframes=3)
    out = select_all_keyframes(segs, cfg)
    assert len(out) >= 3
