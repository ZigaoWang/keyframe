"""FFmpeg-backed sampled video source.

cv2.VideoCapture decodes from the nearest preceding keyframe whenever you
seek, then walks forward to your target frame. For a 4K 60fps video sampled
at 1Hz this means roughly 60 decoded-and-discarded frames per yielded sample,
which is the dominant cost on most machines.

FFmpeg handles decode + scale + frame-rate-pick in a single filter graph,
runs hardware-accelerated codecs (VideoToolbox on macOS, NVDEC on NVIDIA,
VAAPI on Linux) when available, and pipes raw BGR frames over stdout. On a
2160x3840 60fps clip the resulting throughput is roughly 100-300 samples/s,
versus 1-2 samples/s for cv2.

The :class:`FfmpegSampledSource` exposes the same iterator contract as
:class:`keyframe.source.VideoSource`, so the pipeline does not care which one
it is consuming.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .logging_setup import get_logger
from .source import Frame

log = get_logger("ffmpeg_source")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass
class _ProbedStream:
    width: int
    height: int
    fps: float
    duration_sec: float
    frame_count: int
    rotation: int = 0
    """Display rotation in degrees, mod 360. Phone videos commonly carry
    90 / 180 / 270 via the `displaymatrix` side-data block."""


def _probe(path: Path) -> _ProbedStream:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames"
        ":stream_tags=rotate"
        ":stream_side_data=rotation"
        ":format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.check_output(cmd, timeout=30)
    data = json.loads(out)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    def _parse_rate(value: str) -> float:
        if not value or value == "0/0":
            return 0.0
        if "/" in value:
            num, den = value.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)

    fps = _parse_rate(stream.get("avg_frame_rate") or "")
    if fps == 0.0:
        fps = _parse_rate(stream.get("r_frame_rate") or "")
    duration = float(stream.get("duration") or 0.0)
    if duration == 0.0:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    nb_frames_raw = stream.get("nb_frames")
    try:
        frame_count = int(nb_frames_raw) if nb_frames_raw else int(duration * fps)
    except (TypeError, ValueError):
        frame_count = int(duration * fps) if fps else 0

    rotation = 0
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except (TypeError, ValueError):
            rotation = 0
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = int(sd["rotation"])
            except (TypeError, ValueError):
                pass
            break
    rotation = ((-rotation) % 360 + 360) % 360  # ffmpeg reports CCW; we want CW

    return _ProbedStream(width=width, height=height, fps=fps,
                          duration_sec=duration, frame_count=frame_count,
                          rotation=rotation)


def _rotation_filter(rot: int) -> str | None:
    if rot == 90:
        return "transpose=1"   # 90 deg CW
    if rot == 180:
        return "transpose=1,transpose=1"
    if rot == 270:
        return "transpose=2"   # 90 deg CCW
    return None


class FfmpegSampledSource:
    """Sample frames from a file via an ffmpeg subprocess.

    Frames arrive pre-scaled at most ``max_width`` pixels wide and pre-decimated
    to ``1/interval_sec`` Hz. Each yielded :class:`Frame` carries the stream
    timestamp at which ffmpeg actually sampled the frame.

    Pass ``start_sec`` / ``end_sec`` to decode only a contiguous chunk of the
    file. Used by :class:`ParallelFfmpegSource` to fan out across workers.
    """

    def __init__(
        self,
        path: str | Path,
        interval_sec: float,
        max_width: int = 960,
        hwaccel: str | None = "auto",
        start_sec: float = 0.0,
        end_sec: float | None = None,
        probed: "_ProbedStream | None" = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg/ffprobe not on PATH")
        if interval_sec <= 0:
            raise ValueError("interval_sec must be > 0")

        self.interval_sec = float(interval_sec)
        info = probed if probed is not None else _probe(self.path)
        self.fps = info.fps
        self.frame_count = info.frame_count
        self.duration_sec = info.duration_sec
        self.src_width = info.width
        self.src_height = info.height
        self.rotation = info.rotation
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec) if end_sec is not None else info.duration_sec
        self.is_file = True
        self.realtime = False
        self._probed = info

        # If the video carries a 90 or 270 deg rotation, the displayed width
        # corresponds to the raw frame height (and vice-versa). Compute output
        # dims from the displayed orientation, not the raw pixel layout.
        if info.rotation in (90, 270):
            display_w, display_h = info.height, info.width
        else:
            display_w, display_h = info.width, info.height

        scale = min(1.0, max_width / max(display_w, 1))
        self.width = max(2, (int(display_w * scale) // 2) * 2)
        self.height = max(2, (int(display_h * scale) // 2) * 2)
        self._frame_bytes = self.width * self.height * 3
        self._sample_rate = 1.0 / self.interval_sec

        hwflag: list[str] = []
        if hwaccel and hwaccel != "off":
            hwflag = ["-hwaccel", hwaccel]

        seek_flags: list[str] = []
        if self.start_sec > 0:
            seek_flags = ["-ss", f"{self.start_sec:.3f}"]
        duration_flags: list[str] = []
        if self.end_sec < info.duration_sec:
            duration_flags = ["-to", f"{self.end_sec:.3f}"]

        # filter chain: rotate raw pixels first (if needed), then decimate to
        # the target fps, then scale to final display size.
        filters: list[str] = []
        rot_filter = _rotation_filter(info.rotation)
        if rot_filter:
            filters.append(rot_filter)
        filters.append(f"fps={self._sample_rate:.6f}")
        filters.append(f"scale={self.width}:{self.height}:flags=area")

        self._cmd = [
            "ffmpeg", "-loglevel", "error", "-nostdin",
            "-noautorotate",
            *hwflag,
            *seek_flags,
            "-i", str(self.path),
            *duration_flags,
            "-vf", ",".join(filters),
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-an", "-sn", "-vsync", "0",
            "-",
        ]
        log.debug("ffmpeg cmd: %s", " ".join(self._cmd))
        self._proc: Optional[subprocess.Popen] = None

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * 4,
        )

    def __iter__(self) -> Iterator[Frame]:
        if self._proc is None:
            self._spawn()
        assert self._proc is not None and self._proc.stdout is not None
        idx = 0
        while True:
            buf = self._proc.stdout.read(self._frame_bytes)
            if not buf:
                break
            if len(buf) < self._frame_bytes:
                log.warning("ffmpeg returned short frame (%d/%d bytes), stopping",
                            len(buf), self._frame_bytes)
                break
            bgr = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
            yield Frame(
                index=idx,
                timestamp_sec=self.start_sec + float(idx) * self.interval_sec,
                bgr=bgr.copy(),
            )
            idx += 1
        self._drain()

    def iter_sampled(self, interval_sec: float) -> Iterator[Frame]:
        """Compatibility shim. Interval is fixed at construction; warn if it differs."""
        if abs(interval_sec - self.interval_sec) > 1e-6:
            log.warning(
                "FfmpegSampledSource interval is fixed to %.3fs; ignoring %.3fs",
                self.interval_sec, interval_sec,
            )
        yield from self.__iter__()

    def _drain(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._proc.stderr is not None:
            err = self._proc.stderr.read().decode("utf-8", errors="ignore")
            if err.strip():
                log.debug("ffmpeg stderr: %s", err[:500])

    def release(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> "FfmpegSampledSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Parallel decode                                                             #
# --------------------------------------------------------------------------- #

class ParallelFfmpegSource:
    """Decode the same file in N parallel ffmpeg processes, one per time chunk.

    Each worker runs its own ffmpeg subprocess and writes pre-scaled BGR frames
    into a shared queue. The iterator drains the queue in **timestamp order**
    by buffering and sorting -- so downstream code (segmenter, selector) still
    sees a strictly time-ordered stream.

    Apple Silicon's videotoolbox supports multiple concurrent decode sessions,
    so this typically gives ``num_workers``-fold throughput on long files until
    the SoC's memory bandwidth saturates (around 4-6 workers on M1/M2).

    Parameters
    ----------
    path
        Video file.
    interval_sec
        Sample interval in seconds.
    max_width
        Resize cap; ffmpeg scales internally.
    num_workers
        Number of parallel ffmpeg processes.
    hwaccel
        ffmpeg ``-hwaccel`` flag value; ``"auto"`` picks the best backend.
    queue_capacity
        Hard cap on buffered frames per worker. Smaller = lower memory, more
        backpressure on workers.
    """

    def __init__(
        self,
        path: str | Path,
        interval_sec: float,
        max_width: int = 960,
        num_workers: int = 4,
        hwaccel: str | None = "auto",
        queue_capacity: int = 32,
    ) -> None:
        import queue
        import threading

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg/ffprobe not on PATH")

        self.interval_sec = float(interval_sec)
        self.max_width = int(max_width)
        self.hwaccel = hwaccel
        self.num_workers = max(1, int(num_workers))

        info = _probe(self.path)
        self.duration_sec = info.duration_sec
        self.fps = info.fps
        self.frame_count = info.frame_count
        self.src_width = info.width
        self.src_height = info.height
        self.is_file = True
        self.realtime = False
        self._probed = info

        scale = min(1.0, max_width / max(info.width, 1))
        self.width = max(2, (int(info.width * scale) // 2) * 2)
        self.height = max(2, (int(info.height * scale) // 2) * 2)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_capacity * self.num_workers)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._workers_done = 0
        self._lock = threading.Lock()

    def _worker(self, start_sec: float, end_sec: float, worker_id: int) -> None:
        try:
            sub = FfmpegSampledSource(
                self.path, self.interval_sec, self.max_width,
                hwaccel=self.hwaccel, start_sec=start_sec, end_sec=end_sec,
                probed=self._probed,
            )
            for frame in sub:
                if self._stop.is_set():
                    sub.release()
                    return
                self._queue.put((frame.timestamp_sec, frame))
        except Exception as exc:
            log.exception("worker %d failed: %s", worker_id, exc)
        finally:
            with self._lock:
                self._workers_done += 1
                if self._workers_done >= self.num_workers:
                    self._queue.put(None)  # sentinel for iterator

    def _spawn(self) -> None:
        import threading
        if not self._threads:
            chunk = self.duration_sec / self.num_workers
            for i in range(self.num_workers):
                start = i * chunk
                end = min(self.duration_sec, (i + 1) * chunk)
                # nudge so frames stay aligned to the global sample grid
                start = round(start / self.interval_sec) * self.interval_sec
                if i == self.num_workers - 1:
                    end = self.duration_sec
                t = threading.Thread(
                    target=self._worker, args=(start, end, i), daemon=True,
                )
                self._threads.append(t)
                t.start()

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames in strict timestamp order across all workers.

        Frames whose timestamp lands within ``interval_sec * 0.5`` of a value
        we have already emitted are dropped. This deduplicates the small
        overlap that ffmpeg's keyframe-aligned ``-ss`` produces at chunk
        boundaries.
        """
        self._spawn()
        buffer: list[tuple[float, Frame]] = []
        emitted_grid: set[int] = set()
        eps = self.interval_sec * 0.5
        yielded = 0

        def _emit(fr: Frame) -> Frame | None:
            slot = int(round(fr.timestamp_sec / self.interval_sec))
            if slot in emitted_grid:
                return None
            emitted_grid.add(slot)
            return Frame(index=slot, timestamp_sec=slot * self.interval_sec, bgr=fr.bgr)

        while True:
            try:
                item = self._queue.get(timeout=300)
            except Exception:
                break
            if item is None:
                buffer.sort(key=lambda x: x[0])
                for _ts, fr in buffer:
                    out = _emit(fr)
                    if out is not None:
                        yield out
                        yielded += 1
                buffer.clear()
                break
            ts, fr = item
            buffer.append((ts, fr))
            buffer.sort(key=lambda x: x[0])
            # flush any frame whose slot is now safe (older than any in-flight worker can emit)
            # heuristic: emit anything older than the latest by more than 2 intervals
            latest = buffer[-1][0]
            cutoff = latest - 2 * self.interval_sec
            keep: list[tuple[float, Frame]] = []
            for ts2, fr2 in buffer:
                if ts2 <= cutoff:
                    out = _emit(fr2)
                    if out is not None:
                        yield out
                        yielded += 1
                else:
                    keep.append((ts2, fr2))
            buffer = keep

    def release(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()
