"""Tests for shared I/O helpers."""
from __future__ import annotations

import numpy as np
import pytest

from keyframe.io_utils import (
    cosine_sim,
    imread_unicode,
    imwrite_unicode,
    laplacian_sharpness,
    resize_max_width,
)


def _make_image(h: int = 32, w: int = 48) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_imwrite_then_imread_roundtrip(tmp_path):
    img = _make_image()
    out = tmp_path / "sub" / "frame.jpg"
    assert imwrite_unicode(out, img)
    assert out.exists()
    decoded = imread_unicode(out)
    assert decoded is not None
    assert decoded.shape == img.shape


def test_imread_missing_file_returns_none(tmp_path):
    assert imread_unicode(tmp_path / "missing.jpg") is None


def test_imwrite_unicode_path(tmp_path):
    img = _make_image()
    out = tmp_path / "字幕" / "frame.jpg"
    assert imwrite_unicode(out, img)
    assert out.exists()


def test_cosine_sim_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_sim_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_sim_zero_vector_returns_zero():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert cosine_sim(a, b) == 0.0
    assert cosine_sim(b, a) == 0.0


def test_resize_max_width_shrinks_only_when_wider():
    img = _make_image(h=100, w=400)
    out = resize_max_width(img, max_width=200)
    assert out.shape[1] == 200
    assert out.shape[0] == 50  # aspect preserved

    same = resize_max_width(img, max_width=800)
    assert same.shape == img.shape  # not enlarged


def test_laplacian_sharpness_higher_for_edges():
    blur = np.full((64, 64, 3), 128, dtype=np.uint8)
    edges = blur.copy()
    edges[:, 32:] = 0
    assert laplacian_sharpness(edges) > laplacian_sharpness(blur)
