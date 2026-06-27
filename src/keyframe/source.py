"""Frame-by-frame video source.

A single iterator interface for both files and live streams (webcam, RTSP).
The pipeline never knows which source it is reading from. That is the whole
trick that lets one algorithm cover offline and online cases.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger("source")


@dataclass(frozen=True)
class Frame:
    """A single decoded frame with timing metadata."""
    index: int
    """Zero-based position in the source stream."""

    timestamp_sec: float
    """Stream time in seconds, relative to source start.
    For files: cv2 reports POS_MSEC. For webcam: wall clock since stream opened."""

    bgr: np.ndarray
    """Raw decoded BGR image, full resolution."""


class VideoSource:
    """Yields Frame objects until the source ends or the consumer stops iterating.

    Constructor accepts either a file path or an integer/URL string for a live
    feed. The 'realtime' flag, when True and the source is a file, sleeps so
    iteration ticks at the file's native fps -- useful for proving the pipeline
    keeps up with a true stream.
    """

    def __init__(
        self,
        spec: str | int | Path,
        realtime: bool = False,
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 2.0,
    ) -> None:
        self.spec = spec
        self.realtime = realtime
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _resolve_arg(self) -> str | int:
        if isinstance(self.spec, int):
            return self.spec
        s = str(self.spec)
        if s.isdigit():
            return int(s)
        if s.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return s
        path = Path(s)
        if not path.exists():
            raise FileNotFoundError(f"video source not found: {s}")
        return str(path)

    def _open(self) -> None:
        arg = self._resolve_arg()
        self._cap = cv2.VideoCapture(arg)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open source: {arg}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_file = isinstance(arg, str) and Path(arg).exists()
        log.info(
            "opened source=%s fps=%.2f frames=%s size=%dx%d is_file=%s",
            arg, self.fps, self.frame_count or "stream", self.width, self.height, self.is_file,
        )

    @property
    def duration_sec(self) -> float | None:
        if self.frame_count > 0 and self.fps > 0:
            return self.frame_count / self.fps
        return None

    def __iter__(self) -> Iterator[Frame]:
        assert self._cap is not None
        wall_start = time.time()
        idx = 0
        broken_streak = 0
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                broken_streak += 1
                if self.is_file or broken_streak > self.reconnect_attempts:
                    log.info("stream ended after %d frames", idx)
                    break
                log.warning(
                    "read failed (attempt %d/%d), reconnecting in %.1fs",
                    broken_streak, self.reconnect_attempts, self.reconnect_delay_sec,
                )
                time.sleep(self.reconnect_delay_sec)
                self.release()
                self._open()
                continue
            broken_streak = 0

            if self.is_file:
                pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = pos_msec / 1000.0 if pos_msec > 0 else idx / max(self.fps, 1.0)
            else:
                ts = time.time() - wall_start

            yield Frame(index=idx, timestamp_sec=float(ts), bgr=bgr)
            idx += 1

            if self.realtime and self.is_file and self.fps > 0:
                target = idx / self.fps
                slack = target - (time.time() - wall_start)
                if slack > 0:
                    time.sleep(slack)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def iter_sampled(self, interval_sec: float) -> Iterator[Frame]:
        """Yield only the frames we actually want to analyze.

        For file sources, seek directly to each sample point using
        ``cap.set(CAP_PROP_POS_FRAMES, ...)`` so we never decode and throw
        away the in-between frames. On a 60fps file with a 1s sample interval
        this skips ~59x more decode work than ``subsample_by_time``.

        For live sources (webcam, RTSP) we cannot seek, so we fall back to
        decode-and-drop semantics: every frame is read but only every Nth is
        yielded.
        """
        if self.is_file and self.frame_count > 0 and self.fps > 0:
            yield from self._iter_sampled_seek(interval_sec)
        else:
            yield from subsample_by_time(self, interval_sec)

    def _iter_sampled_seek(self, interval_sec: float) -> Iterator[Frame]:
        assert self._cap is not None
        duration = self.duration_sec or 0.0
        if duration <= 0.0:
            return
        idx = 0
        t = 0.0
        wall_start = time.time()
        while t < duration - 1e-6:
            target_frame = min(int(round(t * self.fps)), self.frame_count - 1)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
            ok, bgr = self._cap.read()
            if not ok:
                log.warning("seek read failed at frame %d (t=%.2fs)", target_frame, t)
                break
            yield Frame(index=idx, timestamp_sec=float(t), bgr=bgr)
            idx += 1
            t += interval_sec
            if self.realtime and self.fps > 0:
                target = idx * interval_sec
                slack = target - (time.time() - wall_start)
                if slack > 0:
                    time.sleep(slack)

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def subsample_by_time(
    source: VideoSource,
    interval_sec: float,
) -> Iterator[Frame]:
    """Yield only frames spaced at least interval_sec apart in stream time.

    Always yields the first frame. Drops in between to limit downstream work.
    """
    last_yielded: float | None = None
    for frame in source:
        if last_yielded is None or frame.timestamp_sec - last_yielded >= interval_sec:
            last_yielded = frame.timestamp_sec
            yield frame
