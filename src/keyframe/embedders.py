"""Embedder plugin registry.

Each embedder turns a BGR frame into a 1-D float32 numpy vector. Vectors live
in cosine-similarity space, which is all downstream stages care about.

Registered embedders:
  yolov8n / yolov8s / yolov8m      YOLOv8 detector backbone features
  yolov8n-seg                       YOLOv8 segmentation backbone
  yolo11n / yolo11s                 YOLO11 detector backbone
  yolo26n                           YOLO26 NMS-free detector backbone
  mobile_sam                        MobileSAM TinyViT image encoder (global-pooled)
  clip-vit-b32 / b16 / l14          OpenAI CLIP image tower (semantic)
  siglip-b16                        SigLIP base/16-256 (beats CLIP on retrieval)
  mobileclip-s1 / s2 / b            MobileCLIP v1 family (Apple)
  mobileclip2-s0                    MobileCLIP v2 small (DFN-DR-2B pretrain)
  phash                             8x8 DCT perceptual hash (cheap baseline)
  hsv                               HSV histogram (color-only signature)

Adding a new model = subclass Embedder, decorate with @register("name").
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import resolve_device
from .logging_setup import get_logger

log = get_logger("embedders")


@dataclass
class EmbedderInfo:
    name: str
    dim: int
    device: str
    backend: str
    """Human-readable description, e.g. 'YOLOv8n backbone (256-d)'."""

    weight_size_mb: float | None = None
    """Disk size of the weight file, when applicable."""


class Embedder(ABC):
    """Frame -> vector. Pure synchronous, batchable interface."""

    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed(self, bgr: np.ndarray) -> np.ndarray:
        """Return a 1-D float32 vector. Caller does not normalise."""

    def embed_batch(self, bgrs: list[np.ndarray]) -> np.ndarray:
        """Embed N frames at once. Default: per-frame loop. Override to batch on GPU.

        Returns
        -------
        np.ndarray, shape (N, D), float32
        """
        if not bgrs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed(b) for b in bgrs], axis=0)

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(name=self.name, dim=self.dim, device=getattr(self, "device", "cpu"),
                            backend=self.__class__.__name__)


_REGISTRY: dict[str, Callable[..., Embedder]] = {}


def register(name: str) -> Callable[[type[Embedder]], type[Embedder]]:
    def decorator(cls: type[Embedder]) -> type[Embedder]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def list_embedders() -> list[str]:
    return sorted(_REGISTRY)


def build_embedder(name: str, device: str = "auto") -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown embedder '{name}'. choices: {list_embedders()}")
    return _REGISTRY[name](device=device)


# --------------------------------------------------------------------------- #
# YOLOv8 backbone embedders                                                   #
# --------------------------------------------------------------------------- #

class _YoloEmbedder(Embedder):
    weights: str = "yolov8n.pt"
    dim = 256

    def __init__(self, device: str = "auto") -> None:
        from ultralytics import YOLO
        self.device = resolve_device(device)
        self._model = YOLO(self.weights)
        log.info("loaded %s on %s", self.weights, self.device)
        self._frames_since_flush = 0

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        result = self._model.embed(bgr, device=self.device, verbose=False)
        vec = result[0] if isinstance(result, list) else result
        if hasattr(vec, "cpu"):
            vec = vec.cpu().numpy()
        arr = np.asarray(vec).flatten().astype(np.float32)
        self._frames_since_flush += 1
        if self.device == "mps" and self._frames_since_flush % 32 == 0:
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
        return arr

    def embed_batch(self, bgrs: list[np.ndarray]) -> np.ndarray:
        """True batched inference. YOLOv8/11/26 all accept a list -> list of features."""
        if not bgrs:
            return np.zeros((0, self.dim), dtype=np.float32)
        results = self._model.embed(bgrs, device=self.device, verbose=False)
        vecs: list[np.ndarray] = []
        for r in results:
            v = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
            vecs.append(v.flatten().astype(np.float32))
        self._frames_since_flush += len(bgrs)
        if self.device == "mps" and self._frames_since_flush >= 32:
            try:
                import torch
                torch.mps.empty_cache()
                self._frames_since_flush = 0
            except Exception:
                pass
        return np.stack(vecs, axis=0)

    def info(self) -> EmbedderInfo:
        size_mb = None
        p = Path(self.weights)
        if p.exists():
            size_mb = round(p.stat().st_size / 1024 / 1024, 1)
        return EmbedderInfo(
            name=self.name, dim=self.dim, device=self.device,
            backend=f"YOLO backbone ({self.weights})", weight_size_mb=size_mb,
        )


@register("yolov8n")
class YoloV8Nano(_YoloEmbedder):
    weights = "yolov8n.pt"
    dim = 256


@register("yolov8s")
class YoloV8Small(_YoloEmbedder):
    weights = "yolov8s.pt"
    dim = 512


@register("yolov8m")
class YoloV8Medium(_YoloEmbedder):
    weights = "yolov8m.pt"
    dim = 576


@register("yolov8n-seg")
class YoloV8NanoSeg(_YoloEmbedder):
    """YOLOv8n trained for instance segmentation. Same 256-d backbone."""
    weights = "yolov8n-seg.pt"
    dim = 256


@register("yolo11n")
class YoloV11Nano(_YoloEmbedder):
    """YOLO11 nano. Architecture revision over YOLOv8 with C3K2 + C2PSA blocks."""
    weights = "yolo11n.pt"
    dim = 256


@register("yolo11s")
class YoloV11Small(_YoloEmbedder):
    weights = "yolo11s.pt"
    dim = 512


@register("yolo26n")
class YoloV26Nano(_YoloEmbedder):
    """YOLO26: NMS-free end-to-end detector. DFL-free regression, MuSGD optimiser."""
    weights = "yolo26n.pt"
    dim = 256


@register("mobile_sam")
class MobileSamEmbedder(Embedder):
    """MobileSAM TinyViT image encoder. Outputs 256x64x64; we global-average-pool to 256-d.

    SAM is normally a prompt-driven segmenter, but its image encoder alone produces a
    dense feature map that works well as a frame embedding. Tiny-ViT runs at ~8ms/image
    on a single GPU, making it the lightest semantic encoder we ship.
    """
    weights = "mobile_sam.pt"
    dim = 256

    def __init__(self, device: str = "auto") -> None:
        from ultralytics import SAM
        import torch
        self.device = resolve_device(device)
        self._sam = SAM(self.weights)
        self._encoder = self._sam.model.image_encoder
        self._encoder.eval()
        try:
            self._encoder = self._encoder.to(self.device)
            self._torch_device = self.device
        except Exception:
            self._torch_device = "cpu"
            log.warning("mobile_sam encoder fell back to CPU")
        self._target_size = 1024
        log.info("loaded mobile_sam encoder on %s", self._torch_device)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        return self._encode_batch([bgr])[0]

    def embed_batch(self, bgrs: list[np.ndarray]) -> np.ndarray:
        if not bgrs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode_batch(bgrs)

    def _encode_batch(self, bgrs: list[np.ndarray]) -> np.ndarray:
        import torch
        tensors = []
        for bgr in bgrs:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self._target_size, self._target_size),
                             interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(self._torch_device)
        with torch.no_grad():
            feat = self._encoder(batch)
        if hasattr(feat, "cpu"):
            feat = feat.cpu().numpy()
        feat = np.asarray(feat)
        if feat.ndim == 4:
            feat = feat.mean(axis=(2, 3))  # (N, C)
        return feat.astype(np.float32)

    def info(self) -> EmbedderInfo:
        size_mb = None
        p = Path(self.weights)
        if p.exists():
            size_mb = round(p.stat().st_size / 1024 / 1024, 1)
        return EmbedderInfo(name=self.name, dim=self.dim, device=getattr(self, "_torch_device", "cpu"),
                            backend="MobileSAM TinyViT encoder (global-pooled)",
                            weight_size_mb=size_mb)


# --------------------------------------------------------------------------- #
# Cheap baselines that need no neural model                                   #
# --------------------------------------------------------------------------- #

@register("phash")
class PerceptualHashEmbedder(Embedder):
    """64-bit perceptual hash unpacked to a 64-d {0,1} vector.

    Useful as a near-zero-cost reference for benchmarks. Catches gross scene
    changes but misses semantic shifts that share the same low-frequency layout.
    """
    dim = 64

    def __init__(self, device: str = "auto") -> None:
        self.device = "cpu"

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(resized)
        low = dct[:8, :8].flatten()
        median = float(np.median(low[1:]))
        bits = (low > median).astype(np.float32)
        return bits


@register("hsv")
class HsvHistogramEmbedder(Embedder):
    """3-channel HSV histogram concatenated, L1-normalised. 96-d."""
    dim = 96

    def __init__(self, device: str = "auto") -> None:
        self.device = "cpu"

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        parts: list[np.ndarray] = []
        for channel, bins in zip(range(3), (32, 32, 32)):
            hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256]).flatten()
            total = hist.sum()
            if total > 0:
                hist = hist / total
            parts.append(hist.astype(np.float32))
        return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Semantic encoders (CLIP family via open_clip)                               #
# --------------------------------------------------------------------------- #

class _OpenClipEmbedder(Embedder):
    """Base class for open_clip image-tower embedders.

    Subclasses set ``model_name`` and ``pretrained`` (open_clip ids); we
    construct the tower once, then push BGR frames through the standard
    open_clip transforms in batches. Output is the projected image
    embedding (i.e. the same vector the text tower compares against).

    Why open_clip and not transformers? open_clip ships the original CLIP
    weights, the LAION re-trains, SigLIP, EVA-CLIP, and MobileCLIP behind
    one interface, and the install is tiny vs. transformers.
    """
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    dim = 512

    def __init__(self, device: str = "auto") -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                f"'{self.name}' requires open_clip_torch.\n"
                "install it with: pip install 'keyframe[clip]'  (or)  "
                "pip install open_clip_torch"
            ) from exc

        self.device = resolve_device(device)
        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained,
        )
        model.eval()
        try:
            model = model.to(self.device)
            self._torch_device = self.device
        except Exception:
            self._torch_device = "cpu"
            log.warning("%s tower fell back to CPU", self.name)
        self._model = model
        self._preprocess = preprocess
        try:
            self.dim = int(model.visual.output_dim)
        except Exception:
            pass
        log.info("loaded %s/%s on %s, dim=%d",
                 self.model_name, self.pretrained, self._torch_device, self.dim)

    def _to_tensor(self, bgr: np.ndarray):
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._preprocess(Image.fromarray(rgb))

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([bgr])[0]

    def embed_batch(self, bgrs: list[np.ndarray]) -> np.ndarray:
        if not bgrs:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        batch = torch.stack([self._to_tensor(b) for b in bgrs], dim=0)
        batch = batch.to(self._torch_device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
        return feats.detach().cpu().numpy().astype(np.float32)

    def info(self) -> EmbedderInfo:
        return EmbedderInfo(
            name=self.name, dim=self.dim,
            device=getattr(self, "_torch_device", "cpu"),
            backend=f"open_clip {self.model_name}/{self.pretrained}",
        )


@register("clip-vit-b32")
class ClipViTB32(_OpenClipEmbedder):
    """OpenAI CLIP ViT-B/32. 512-d. Strong semantic features, ~10 ms / frame on MPS."""
    model_name = "ViT-B-32"
    pretrained = "openai"
    dim = 512


@register("clip-vit-b16")
class ClipViTB16(_OpenClipEmbedder):
    """OpenAI CLIP ViT-B/16. 512-d. Sharper than B/32, ~2x the cost."""
    model_name = "ViT-B-16"
    pretrained = "openai"
    dim = 512


@register("clip-vit-l14")
class ClipViTL14(_OpenClipEmbedder):
    """OpenAI CLIP ViT-L/14. 768-d. Best semantic separation we ship; heavy."""
    model_name = "ViT-L-14"
    pretrained = "openai"
    dim = 768


@register("siglip-b16")
class SigLipB16(_OpenClipEmbedder):
    """SigLIP base/16-256. 768-d. Generally beats CLIP on retrieval benchmarks."""
    model_name = "ViT-B-16-SigLIP"
    pretrained = "webli"
    dim = 768


@register("mobileclip-s1")
class MobileClipS1(_OpenClipEmbedder):
    """MobileCLIP-S1 (Apple). ~5 ms / frame on M-series MPS. Fast semantic encoder."""
    model_name = "MobileCLIP-S1"
    pretrained = "datacompdr"
    dim = 512


@register("mobileclip-s2")
class MobileClipS2(_OpenClipEmbedder):
    """MobileCLIP-S2 (Apple). A notch slower than S1, higher accuracy."""
    model_name = "MobileCLIP-S2"
    pretrained = "datacompdr"
    dim = 512


@register("mobileclip-b")
class MobileClipB(_OpenClipEmbedder):
    """MobileCLIP-B (Apple). Largest of the originals; still ViT-based."""
    model_name = "MobileCLIP-B"
    pretrained = "datacompdr_lt"
    dim = 512


@register("mobileclip2-s0")
class MobileClip2S0(_OpenClipEmbedder):
    """MobileCLIP2-S0 (Apple, v2). DFN-DR-2B pretrain; fastest of the v2 family."""
    model_name = "MobileCLIP2-S0"
    pretrained = "dfndr2b"
    dim = 512


def time_embedder_throughput(embedder: Embedder, sample_bgr: np.ndarray, n_runs: int = 20) -> dict:
    """Quick microbench. Returns p50/p95 latency in ms."""
    # warm-up
    for _ in range(3):
        embedder.embed(sample_bgr)
    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        embedder.embed(sample_bgr)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(timings)
    return {
        "embedder": embedder.name,
        "dim": embedder.dim,
        "device": getattr(embedder, "device", "cpu"),
        "latency_ms_mean": round(float(arr.mean()), 2),
        "latency_ms_p50": round(float(np.percentile(arr, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(arr, 95)), 2),
        "throughput_fps": round(1000.0 / max(arr.mean(), 1e-6), 2),
    }
