"""LLM caption stage.

Sends the selected keyframes to an OpenAI-compatible Chat Completions endpoint
in a single multi-image request. The endpoint is configured via environment
variables (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``), which makes this transparent
to OpenAI itself, OpenRouter, or any other compatible proxy.

Request shape (per the chat completions multimodal spec):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": [
            {"type": "text", "text": "FRAME 1/N @ t=..."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}},
            ...
        ]},
    ]
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CaptionerConfig
from .io_utils import imread_unicode, resize_max_width
from .logging_setup import get_logger
from .selector import Keyframe

log = get_logger("captioner")


SYSTEM_PROMPT = (
    "You are a careful video analyst. Given keyframes from a single video in "
    "time order, write one clear, factual narration of what happens in the "
    "video.\n"
    "Rules:\n"
    "- Output a single coherent paragraph, in English, 200-350 words.\n"
    "- Describe what the camera sees: setting, on-screen text, people, key "
    "actions, transitions, and the final state.\n"
    "- Do NOT guess at unstated facts: do not invent locations, names, brands, "
    "or motivations that are not visible. Say 'unclear' when uncertain.\n"
    "- Do NOT list frames or output JSON or bullet points.\n"
)


USER_HEADER_TEMPLATE = (
    "Below are {n} keyframes selected from a single video of duration ~{dur:.1f}s, "
    "presented in temporal order. Synthesize what happens across the whole video.\n"
)


@dataclass
class CaptionResult:
    text: str
    model: str
    frames_sent: int
    latency_sec: float
    usage: dict | None
    raw: dict | None


def _frame_to_data_url(bgr: np.ndarray, max_width: int, quality: int) -> str:
    img = resize_max_width(bgr, max_width)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_content(keyframes: list[Keyframe], duration_sec: float, cfg: CaptionerConfig) -> list[dict]:
    content: list[dict] = [{
        "type": "text",
        "text": USER_HEADER_TEMPLATE.format(n=len(keyframes), dur=duration_sec),
    }]
    for k in keyframes:
        bgr = imread_unicode(k.bgr_path)
        if bgr is None:
            log.warning("could not read keyframe image: %s", k.bgr_path)
            continue
        url = _frame_to_data_url(bgr, cfg.thumb_max_width, cfg.jpeg_quality)
        content.append({"type": "text",
                        "text": f"FRAME {k.keyframe_id}/{len(keyframes)} @ t={k.timestamp_sec:.2f}s"
                                f"  (segment #{k.segment_id})"})
        content.append({"type": "image_url",
                        "image_url": {"url": url, "detail": cfg.detail}})
    return content


def caption_keyframes(
    keyframes: list[Keyframe],
    duration_sec: float,
    cfg: CaptionerConfig,
) -> CaptionResult:
    """Single LLM call. Caps frames sent to ``cfg.max_keyframes``."""
    if not cfg.enabled:
        return CaptionResult(text="(captioner disabled)", model=cfg.model,
                             frames_sent=0, latency_sec=0.0, usage=None, raw=None)
    if not keyframes:
        return CaptionResult(text="(no keyframes selected; nothing to caption)",
                             model=cfg.model, frames_sent=0, latency_sec=0.0,
                             usage=None, raw=None)

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"missing {cfg.api_key_env}. set it in .env or environment to enable captioning."
        )

    if len(keyframes) > cfg.max_keyframes:
        # uniformly subsample to the cap
        idx = np.linspace(0, len(keyframes) - 1, cfg.max_keyframes).round().astype(int).tolist()
        seen: list[int] = []
        for i in idx:
            if i not in seen:
                seen.append(int(i))
        keyframes = [keyframes[i] for i in seen]
        log.info("capped keyframes to %d for captioning", len(keyframes))

    from openai import OpenAI
    base_url = os.environ.get(cfg.base_url_env, "") or None
    client = OpenAI(base_url=base_url) if base_url else OpenAI()

    content = _build_content(keyframes, duration_sec, cfg)

    models_to_try = [cfg.model, *cfg.fallback_models]
    last_error: Exception | None = None
    response = None
    used_model = cfg.model
    t0 = time.perf_counter()

    for attempt, model_name in enumerate(models_to_try):
        try:
            log.info("captioner: trying %s (%d frame%s) ...",
                     model_name, len(keyframes), "" if len(keyframes) == 1 else "s")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            used_model = model_name
            break
        except Exception as exc:  # noqa: BLE001 - intentionally broad
            last_error = exc
            msg = str(exc)
            looks_regional = any(token in msg for token in (
                "403", "not available in your region", "country",
                "invalid_api_key", "Unauthorized", "401",
            ))
            log.warning("captioner: %s failed (%s)%s",
                        model_name, type(exc).__name__,
                        " -- trying fallback" if looks_regional and attempt < len(models_to_try) - 1 else "")
            if not looks_regional:
                raise
            continue

    if response is None:
        raise RuntimeError(
            f"all captioner models failed. last error: {last_error}"
        )

    latency = time.perf_counter() - t0
    text = response.choices[0].message.content.strip()
    usage = None
    if getattr(response, "usage", None) is not None:
        try:
            usage = json.loads(response.usage.model_dump_json())
        except Exception:
            usage = None
    raw = None
    try:
        raw = json.loads(response.model_dump_json())
    except Exception:
        pass

    if used_model != cfg.model:
        log.info("captioner: succeeded on fallback %s after %s was blocked",
                 used_model, cfg.model)

    log.info("captioner: %d chars in %.2fs (model=%s)", len(text), latency, used_model)
    return CaptionResult(text=text, model=used_model, frames_sent=len(keyframes),
                         latency_sec=latency, usage=usage, raw=raw)
