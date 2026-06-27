"""Online shot segmenter.

Walks frames in time order, maintains a running mean embedding for the active
shot, and emits a new Segment when the next frame's cosine similarity falls
below the configured threshold.

Designed to be incremental: callers feed one (frame, embedding) pair at a time
and receive an optional emitted Segment back. The same algorithm works for
offline pre-extracted frames or a live RTSP feed; only the I/O differs.

Anti-flicker:
  A new shot is only opened once the current shot has accumulated at least
  ``min_shot_sec`` seconds of stream time. Sub-second blips (autofocus, hand
  shake) are absorbed instead of producing throwaway segments.

EMA mean:
  When ``use_ema`` is set the running mean is an exponential moving average:
  ``mean = alpha * frame + (1 - alpha) * mean``. This caps the influence of
  very old frames in long shots and prevents drift on slowly evolving scenes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import SegmenterConfig
from .io_utils import cosine_sim
from .logging_setup import get_logger

log = get_logger("segmenter")


@dataclass
class SampledFrame:
    """A frame that was analyzed (sampled, embedded, decided)."""
    index: int
    """Index among analyzed frames (NOT source frames)."""
    source_index: int
    """Index in the original VideoSource stream."""
    timestamp_sec: float
    embedding: np.ndarray
    bgr_path: str
    """Path on disk where this frame was cached."""
    sharpness: float
    """Laplacian variance. Higher = sharper."""
    sim_to_shot_mean: float
    """cos_sim against the running shot mean BEFORE this frame is absorbed/emitted."""
    is_shot_start: bool
    """True iff this frame opened a new shot."""


@dataclass
class Segment:
    """A finalised contiguous shot of frames."""
    segment_id: int
    start_sec: float
    end_sec: float
    frames: list[SampledFrame] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(self.end_sec - self.start_sec, 0.0)

    @property
    def num_frames(self) -> int:
        return len(self.frames)


class StreamingSegmenter:
    """Stateful segmenter. Call ``ingest(frame)`` and pull ``closed_segment`` if any."""

    def __init__(self, cfg: SegmenterConfig) -> None:
        self.cfg = cfg
        self._segments: list[Segment] = []
        self._active: Optional[Segment] = None
        self._shot_mean: Optional[np.ndarray] = None
        self._shot_count: int = 0
        self._next_segment_id: int = 0
        self._analyzed_count: int = 0

    @property
    def closed_segments(self) -> list[Segment]:
        return list(self._segments)

    def ingest(
        self,
        source_index: int,
        timestamp_sec: float,
        embedding: np.ndarray,
        bgr_path: str,
        sharpness: float,
    ) -> Optional[Segment]:
        """Feed one analyzed frame. Returns the segment just closed (if any).

        The returned segment is also appended to ``closed_segments``.
        """
        cfg = self.cfg
        sim = 1.0
        is_start = False

        if self._shot_mean is None:
            is_start = True
        else:
            sim = cosine_sim(embedding, self._shot_mean)
            active_age = timestamp_sec - (self._active.start_sec if self._active else 0.0)
            if sim < cfg.sim_threshold and active_age >= cfg.min_shot_sec:
                is_start = True

        sampled = SampledFrame(
            index=self._analyzed_count,
            source_index=source_index,
            timestamp_sec=timestamp_sec,
            embedding=embedding.astype(np.float32),
            bgr_path=bgr_path,
            sharpness=float(sharpness),
            sim_to_shot_mean=float(sim),
            is_shot_start=is_start,
        )
        self._analyzed_count += 1

        closed: Optional[Segment] = None
        if is_start:
            if self._active is not None:
                self._active.end_sec = timestamp_sec
                self._segments.append(self._active)
                closed = self._active
            self._active = Segment(
                segment_id=self._next_segment_id,
                start_sec=timestamp_sec,
                end_sec=timestamp_sec,
            )
            self._next_segment_id += 1
            self._shot_mean = embedding.astype(np.float32).copy()
            self._shot_count = 1
        else:
            assert self._active is not None and self._shot_mean is not None
            if cfg.use_ema:
                self._shot_mean = (
                    cfg.ema_alpha * embedding + (1.0 - cfg.ema_alpha) * self._shot_mean
                ).astype(np.float32)
            else:
                self._shot_mean = (
                    self._shot_mean * self._shot_count + embedding
                ) / (self._shot_count + 1)
            self._shot_count += 1
            self._active.end_sec = timestamp_sec

        assert self._active is not None
        self._active.frames.append(sampled)
        return closed

    def finalise(self, end_sec: float | None = None) -> Optional[Segment]:
        """Close the currently open shot. Call once at end of stream."""
        if self._active is None:
            return None
        if end_sec is not None:
            self._active.end_sec = max(end_sec, self._active.end_sec)
        self._segments.append(self._active)
        closed = self._active
        self._active = None
        self._shot_mean = None
        self._shot_count = 0
        log.info("finalised: %d segment(s)", len(self._segments))
        return closed
