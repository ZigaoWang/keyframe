"""Shared I/O helpers. Unicode-safe image read/write, csv helpers, atomic JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def imwrite_unicode(path: Path | str, image: np.ndarray) -> bool:
    """cv2.imwrite that handles non-ASCII paths on every OS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def imread_unicode(path: Path | str) -> np.ndarray | None:
    """cv2.imread that handles non-ASCII paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_csv(rows: list[dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def append_jsonl(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def laplacian_sharpness(bgr: np.ndarray) -> float:
    """Higher = sharper. Used to reject motion-blurred frames."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Defensive cosine similarity. Returns 0.0 if either vector is zero."""
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def resize_max_width(bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    scale = max_width / w
    return cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
