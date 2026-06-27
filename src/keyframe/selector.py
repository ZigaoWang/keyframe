"""Per-segment keyframe picking + cross-segment diversity filter.

Two stages:

1.  ``select_segment_keyframes`` ranks frames inside one segment by

        composite = w_sharp * z(sharpness) + w_repr * z(sim_to_centroid)

    then greedily picks the top frames subject to a minimum temporal spacing.
    Long segments get more keyframes (one per ``seconds_per_keyframe``,
    capped at ``max_frames_per_segment``).

2.  ``select_all_keyframes`` runs stage 1 on every segment, then applies a
    cross-segment **diversity filter**: any candidate whose embedding has
    cosine similarity above ``diversity_sim_threshold`` with an already-kept
    keyframe is dropped. This removes the "ten near-identical corridor
    walks" pathology when EMA segmenting splits one long static shot.

    A ``min_keyframes`` floor reinstates the highest-scoring dropped
    candidates if the filter leaves us with too few.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SelectorConfig
from .io_utils import cosine_sim
from .logging_setup import get_logger
from .segmenter import SampledFrame, Segment

log = get_logger("selector")


@dataclass
class Keyframe:
    """One chosen representative frame from a segment."""
    keyframe_id: int
    segment_id: int
    source_index: int
    timestamp_sec: float
    bgr_path: str
    sharpness: float
    representativeness: float
    composite_score: float
    rank_in_segment: int
    embedding: np.ndarray | None = None
    """Frame embedding. Carried through so the diversity filter can compare
    candidates without re-loading anything."""


def _zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mu = float(values.mean())
    sigma = float(values.std())
    if sigma < 1e-9:
        return np.zeros_like(values)
    return (values - mu) / sigma


def _target_k(duration_sec: float, cfg: SelectorConfig) -> int:
    if cfg.seconds_per_keyframe <= 0:
        return cfg.max_frames_per_segment
    desired = math.ceil(duration_sec / cfg.seconds_per_keyframe)
    return max(1, min(cfg.max_frames_per_segment, desired))


def select_segment_keyframes(
    segment: Segment,
    cfg: SelectorConfig,
    starting_keyframe_id: int,
) -> list[Keyframe]:
    """Pick best frames inside one segment, ordered by time."""
    if segment.num_frames == 0:
        return []
    if segment.num_frames == 1:
        f = segment.frames[0]
        return [Keyframe(
            keyframe_id=starting_keyframe_id,
            segment_id=segment.segment_id,
            source_index=f.source_index,
            timestamp_sec=f.timestamp_sec,
            bgr_path=f.bgr_path,
            sharpness=f.sharpness,
            representativeness=1.0,
            composite_score=0.0,
            rank_in_segment=0,
            embedding=f.embedding,
        )]

    embeds = np.stack([f.embedding for f in segment.frames])
    centroid = embeds.mean(axis=0)
    reprs = np.array([cosine_sim(e, centroid) for e in embeds], dtype=np.float32)
    sharps = np.array([f.sharpness for f in segment.frames], dtype=np.float32)

    composite = (
        cfg.sharpness_weight * _zscore(sharps)
        + cfg.representativeness_weight * _zscore(reprs)
    )

    k = _target_k(segment.duration_sec, cfg)
    order = np.argsort(-composite)

    picked: list[int] = []
    for idx in order:
        candidate_t = segment.frames[int(idx)].timestamp_sec
        if any(
            abs(candidate_t - segment.frames[p].timestamp_sec) < cfg.min_temporal_spacing_sec
            for p in picked
        ):
            continue
        picked.append(int(idx))
        if len(picked) >= k:
            break

    if not picked:
        picked = [int(order[0])]

    picked.sort(key=lambda i: segment.frames[i].timestamp_sec)
    rank_lookup = {idx: rank for rank, idx in enumerate(order.tolist())}

    keyframes: list[Keyframe] = []
    for offset, idx in enumerate(picked):
        f = segment.frames[idx]
        keyframes.append(Keyframe(
            keyframe_id=starting_keyframe_id + offset,
            segment_id=segment.segment_id,
            source_index=f.source_index,
            timestamp_sec=f.timestamp_sec,
            bgr_path=f.bgr_path,
            sharpness=float(sharps[idx]),
            representativeness=float(reprs[idx]),
            composite_score=float(composite[idx]),
            rank_in_segment=rank_lookup[idx],
            embedding=f.embedding,
        ))
    return keyframes


def _diversity_filter(
    candidates: list[Keyframe],
    cfg: SelectorConfig,
) -> list[Keyframe]:
    """Drop near-duplicates across segments. Always keeps the highest-scoring
    of any two similar candidates. Re-numbers ``keyframe_id`` after filtering."""
    if len(candidates) <= 1 or cfg.diversity_sim_threshold >= 1.0:
        return candidates

    # rank by composite score desc, walk down, drop if too close to any kept
    by_score = sorted(candidates, key=lambda k: -k.composite_score)
    kept: list[Keyframe] = []
    dropped: list[Keyframe] = []
    for cand in by_score:
        if cand.embedding is None:
            kept.append(cand)
            continue
        too_close = False
        for k in kept:
            if k.embedding is None:
                continue
            if cosine_sim(cand.embedding, k.embedding) > cfg.diversity_sim_threshold:
                too_close = True
                break
        if too_close:
            dropped.append(cand)
        else:
            kept.append(cand)

    # restore min_keyframes floor by re-adding highest-scoring dropped
    if len(kept) < cfg.min_keyframes and dropped:
        need = cfg.min_keyframes - len(kept)
        kept.extend(dropped[:need])

    # re-order by time, renumber
    kept.sort(key=lambda k: k.timestamp_sec)
    out: list[Keyframe] = []
    for i, k in enumerate(kept, start=1):
        out.append(Keyframe(
            keyframe_id=i,
            segment_id=k.segment_id,
            source_index=k.source_index,
            timestamp_sec=k.timestamp_sec,
            bgr_path=k.bgr_path,
            sharpness=k.sharpness,
            representativeness=k.representativeness,
            composite_score=k.composite_score,
            rank_in_segment=k.rank_in_segment,
            embedding=k.embedding,
        ))
    if dropped:
        log.info("diversity filter: kept %d, dropped %d as near-duplicates (sim>%.2f)",
                 len(out), len(dropped), cfg.diversity_sim_threshold)
    return out


def select_all_keyframes(
    segments: list[Segment],
    cfg: SelectorConfig,
) -> list[Keyframe]:
    """Run the per-segment selector across every segment, then apply the
    cross-segment diversity filter."""
    candidates: list[Keyframe] = []
    next_id = 1
    for seg in segments:
        chosen = select_segment_keyframes(seg, cfg, next_id)
        candidates.extend(chosen)
        next_id += len(chosen)
    log.info(
        "selector: %d segments -> %d candidates",
        len(segments), len(candidates),
    )
    final = _diversity_filter(candidates, cfg)
    log.info("selector: final keyframes after diversity filter: %d", len(final))
    return final
