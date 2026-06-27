"""Visualizations for inspecting pipeline output.

Three artefacts:
  - ``timeline.jpg``         coloured strip; one cell per analyzed frame, colour = segment id.
                              keyframes get a small white marker above the strip.
  - ``similarity_curve.jpg`` line plot of cos_sim against the shot mean across stream time,
                              with the threshold and keyframe times annotated.
  - ``keyframes_grid.jpg``   contact sheet of all selected keyframes with metadata banners.

Everything draws with OpenCV; no matplotlib dependency.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .io_utils import imread_unicode, imwrite_unicode
from .logging_setup import get_logger
from .segmenter import SampledFrame
from .selector import Keyframe

log = get_logger("visualize")


def _palette(num_colors: int) -> np.ndarray:
    """Distinct-ish colours via the golden-ratio HSV trick."""
    colours = np.zeros((max(num_colors, 1), 3), dtype=np.uint8)
    for i in range(num_colors):
        h = int((i * 137) % 180)
        hsv = np.uint8([[[h, 200, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colours[i] = bgr
    return colours


def draw_film_strip(
    samples: list[SampledFrame],
    keyframes: list["Keyframe"] | list,
    save_path: Path | str,
    thumb_height: int = 140,
    band_height: int = 28,
    marker_height: int = 34,
    label_height: int = 26,
    max_width: int = 2200,
    max_rows: int = 4,
) -> Path | None:
    """Film strip showing every analyzed frame with proper aspect ratio.

    Auto-wraps to multiple rows when the total width exceeds ``max_width`` so
    the thumbnails stay readable on portrait videos and on long recordings.

    Each row has:
        * keyframe marker band on top  (gold dot + KF index)
        * thumbnail row preserving each frame's aspect ratio
        * coloured segment band  (one colour per segment id)
        * time labels every few cells
    """
    if not samples:
        return None
    from .io_utils import imread_unicode

    n = len(samples)

    # detect the dominant aspect ratio from the first decodable thumbnail
    sample_aspect = 16 / 9
    for s in samples:
        try:
            bgr = imread_unicode(s.bgr_path)
        except FileNotFoundError:
            bgr = None
        if bgr is not None:
            h, w = bgr.shape[:2]
            sample_aspect = max(0.3, w / max(h, 1))
            break

    # cell width preserves thumb aspect, padded by 4px gutter
    cell_w = int(thumb_height * sample_aspect) + 4
    cell_w = max(60, cell_w)

    # decide how many rows we need to stay under max_width
    cells_per_row = max(1, max_width // cell_w)
    rows_needed = math.ceil(n / cells_per_row)
    if rows_needed > max_rows:
        # too many frames; shrink cells to fit max_rows rows
        cells_per_row = math.ceil(n / max_rows)
        cell_w = max(40, max_width // cells_per_row)
        rows_needed = max_rows
        # also shrink thumb_height so aspect stays right with smaller cells
        thumb_height = max(70, int((cell_w - 4) / sample_aspect))

    row_h = marker_height + thumb_height + band_height + label_height
    total_w = cells_per_row * cell_w
    total_h = rows_needed * row_h
    img = np.full((total_h, total_w, 3), 248, dtype=np.uint8)

    # segment ids
    seg_ids: list[int] = []
    cur = -1
    for s in samples:
        if s.is_shot_start:
            cur += 1
        seg_ids.append(max(cur, 0))
    palette = _palette(max(seg_ids) + 1)

    kf_lookup = {
        round(float(k.timestamp_sec if hasattr(k, "timestamp_sec") else k["timestamp_sec"]), 3):
        (k.keyframe_id if hasattr(k, "keyframe_id") else k["keyframe_id"])
        for k in keyframes
    }

    for i, s in enumerate(samples):
        r = i // cells_per_row
        c = i % cells_per_row
        x0 = c * cell_w
        y0 = r * row_h

        # segment colour band
        band_y0 = y0 + marker_height + thumb_height
        band_y1 = band_y0 + band_height
        sid = seg_ids[i]
        cv2.rectangle(img, (x0, band_y0), (x0 + cell_w, band_y1),
                      tuple(int(c) for c in palette[sid]), thickness=-1)
        if s.is_shot_start and i > 0 and c > 0:
            cv2.line(img, (x0, y0 + marker_height),
                     (x0, band_y1), (15, 15, 15), 2)

        # thumbnail (preserve aspect). Tolerate not-yet-flushed cache files.
        try:
            bgr = imread_unicode(s.bgr_path)
        except FileNotFoundError:
            bgr = None
        if bgr is not None:
            h, w = bgr.shape[:2]
            target_h = thumb_height
            target_w = int(w * (target_h / max(h, 1)))
            if target_w > cell_w - 4:
                target_w = cell_w - 4
                target_h = int(h * (target_w / max(w, 1)))
            thumb = cv2.resize(bgr, (target_w, target_h),
                               interpolation=cv2.INTER_AREA)
            tx = x0 + (cell_w - target_w) // 2
            ty = y0 + marker_height + (thumb_height - target_h) // 2
            img[ty:ty + target_h, tx:tx + target_w, :] = thumb
            cv2.rectangle(img, (tx, ty), (tx + target_w - 1, ty + target_h - 1),
                          (220, 220, 220), 1)

        # keyframe marker
        kid = kf_lookup.get(round(s.timestamp_sec, 3))
        if kid is not None:
            mx = x0 + cell_w // 2
            cv2.line(img, (mx, y0 + 4),
                     (mx, y0 + marker_height + thumb_height),
                     (60, 200, 240), 2)
            cv2.circle(img, (mx, y0 + marker_height // 2), 10,
                       (60, 200, 240), -1)
            cv2.circle(img, (mx, y0 + marker_height // 2), 10,
                       (30, 80, 110), 1)
            label = f"KF{kid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(img, label, (mx - tw // 2, y0 + marker_height // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

        # time label below the band
        if c == 0 or c == cells_per_row - 1 or i % max(1, cells_per_row // 6) == 0:
            label_y = band_y1 + 18
            cv2.putText(img, f"{s.timestamp_sec:.0f}s",
                        (x0 + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

    imwrite_unicode(save_path, img)
    return Path(save_path)


def draw_timeline(
    samples: list[SampledFrame],
    keyframe_ts: list[float],
    save_path: Path | str,
    height: int = 110,
    target_width: int = 1400,
) -> None:
    if not samples:
        return
    num_segments = max(s.is_shot_start for s in samples)
    seg_ids = []
    cur = -1
    for s in samples:
        if s.is_shot_start:
            cur += 1
        seg_ids.append(max(cur, 0))
    palette = _palette(max(seg_ids) + 1)

    n = len(samples)
    cell_w = max(8, target_width // n)
    width = n * cell_w
    strip_h = height - 24
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    # coloured strip
    for i, sid in enumerate(seg_ids):
        x0 = i * cell_w
        cv2.rectangle(img, (x0, 24), (x0 + cell_w, 24 + strip_h),
                      tuple(int(c) for c in palette[sid]), thickness=-1)
        if samples[i].is_shot_start:
            cv2.line(img, (x0, 24), (x0, 24 + strip_h), (10, 10, 10), 2)

    # keyframe markers
    for kf_t in keyframe_ts:
        # find nearest analyzed frame
        nearest = min(range(n), key=lambda i: abs(samples[i].timestamp_sec - kf_t))
        x = nearest * cell_w + cell_w // 2
        cv2.circle(img, (x, 14), 6, (0, 0, 0), -1)
        cv2.circle(img, (x, 14), 4, (255, 255, 255), -1)

    # time axis labels
    ticks = max(8, min(14, n))
    for i in range(ticks):
        f_idx = int(i * (n - 1) / max(ticks - 1, 1))
        x = f_idx * cell_w
        label = f"{samples[f_idx].timestamp_sec:.1f}s"
        cv2.line(img, (x, 24 + strip_h), (x, 24 + strip_h + 4), (40, 40, 40), 1)
        cv2.putText(img, label, (max(0, x - 18), height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)

    cv2.putText(img, "timeline: colour = segment, dot = selected keyframe",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    imwrite_unicode(save_path, img)


def draw_similarity_curve(
    samples: list[SampledFrame],
    threshold: float,
    keyframe_ts: list[float],
    save_path: Path | str,
    width: int = 1400,
    height: int = 320,
) -> None:
    if not samples:
        return
    pad_l, pad_r, pad_t, pad_b = 60, 30, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    img = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (pad_l, pad_t), (width - pad_r, height - pad_b),
                  (60, 60, 60), 1)

    y_min, y_max = 0.5, 1.0
    for sim in (0.6, 0.7, 0.8, 0.9, 1.0):
        if not (y_min <= sim <= y_max):
            continue
        y = pad_t + int((1 - (sim - y_min) / (y_max - y_min)) * plot_h)
        cv2.line(img, (pad_l, y), (width - pad_r, y), (220, 220, 220), 1)
        cv2.putText(img, f"{sim:.1f}", (8, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)

    y_thr = pad_t + int((1 - (threshold - y_min) / (y_max - y_min)) * plot_h)
    cv2.line(img, (pad_l, y_thr), (width - pad_r, y_thr), (0, 0, 200), 1)
    cv2.putText(img, f"threshold = {threshold:.3f}",
                (width - pad_r - 150, max(y_thr - 6, pad_t + 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    n = len(samples)
    pts: list[tuple[int, int]] = []
    for i, s in enumerate(samples):
        x = pad_l + int((i / max(n - 1, 1)) * plot_w)
        sim_clamped = max(y_min, min(y_max, s.sim_to_shot_mean))
        y = pad_t + int((1 - (sim_clamped - y_min) / (y_max - y_min)) * plot_h)
        pts.append((x, y))
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], (40, 90, 200), 2)
    for x, y in pts:
        cv2.circle(img, (x, y), 2, (40, 90, 200), -1)

    kf_set = set(round(t, 3) for t in keyframe_ts)
    for i, s in enumerate(samples):
        if round(s.timestamp_sec, 3) in kf_set:
            x = pts[i][0]
            cv2.line(img, (x, pad_t), (x, height - pad_b), (0, 160, 60), 1)
            cv2.circle(img, (x, pts[i][1]), 4, (0, 160, 60), -1)

    ticks = max(8, min(12, n))
    for i in range(ticks):
        f_idx = int(i * (n - 1) / max(ticks - 1, 1))
        x = pts[f_idx][0]
        cv2.putText(img, f"{samples[f_idx].timestamp_sec:.0f}s",
                    (max(pad_l, x - 12), height - pad_b + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

    cv2.putText(img, "cosine similarity vs running shot mean.  green = selected keyframe",
                (pad_l, pad_t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    imwrite_unicode(save_path, img)


def draw_keyframes_grid(
    keyframes: list[Keyframe],
    save_path: Path | str,
    cols: int = 5,
    thumb_width: int = 320,
) -> None:
    if not keyframes:
        return
    tiles: list[np.ndarray] = []
    for k in keyframes:
        img = imread_unicode(k.bgr_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_width / w
        thumb_h = int(h * scale)
        thumb = cv2.resize(img, (thumb_width, thumb_h))
        banner_h = 62
        tile = np.full((thumb_h + banner_h, thumb_width, 3), 22, dtype=np.uint8)
        tile[:thumb_h, :, :] = thumb
        cv2.putText(tile, f"KF{k.keyframe_id:02d}  seg #{k.segment_id}  t={k.timestamp_sec:.1f}s",
                    (8, thumb_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(tile, f"sharp={k.sharpness:.0f}  repr={k.representativeness:.3f}",
                    (8, thumb_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
        cv2.putText(tile, f"score={k.composite_score:+.2f}",
                    (8, thumb_h + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 210, 240), 1)
        tiles.append(tile)
    if not tiles:
        return
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * th, cols * tw, 3), 235, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        h_i = min(tile.shape[0], th)
        sheet[r * th:r * th + h_i, c * tw:(c + 1) * tw, :] = tile[:h_i, :, :]
    imwrite_unicode(save_path, sheet)
