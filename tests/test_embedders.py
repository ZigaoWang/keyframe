"""Tests for the zero-dependency embedder baselines."""
from __future__ import annotations

import numpy as np

from keyframe.embedders import build_embedder, list_embedders


def _checker_frame(h: int = 64, w: int = 64) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[::4, :, :] = 255
    return img


def test_list_embedders_contains_baselines():
    names = list_embedders()
    assert "phash" in names
    assert "hsv" in names


def test_phash_returns_64d_binary_vector():
    emb = build_embedder("phash")
    v = emb.embed(_checker_frame())
    assert v.shape == (64,)
    assert v.dtype == np.float32
    assert set(np.unique(v).tolist()).issubset({0.0, 1.0})


def test_hsv_returns_96d_normalised_histogram():
    emb = build_embedder("hsv")
    v = emb.embed(_checker_frame())
    assert v.shape == (96,)
    # three concatenated histograms each L1-normalised
    chunks = v.reshape(3, 32)
    sums = chunks.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_embed_batch_default_loop_matches_per_frame():
    emb = build_embedder("phash")
    frames = [_checker_frame() for _ in range(3)]
    batch = emb.embed_batch(frames)
    assert batch.shape == (3, 64)
    for i, f in enumerate(frames):
        assert np.allclose(batch[i], emb.embed(f))


def test_build_embedder_rejects_unknown_name():
    try:
        build_embedder("definitely-not-a-real-embedder")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
