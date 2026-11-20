"""Coarse place recognition on the map: "which map tile does this frame show?".

The map is cut into square tiles of the frame footprint size (map px); every tile gets a global descriptor (DINOv2
embedding or a classical thumbnail), cached on disk. A frame resized to the map resolution (and optionally rotated)
is embedded the same way and the most similar tiles are returned. A precise matcher then verifies the top-k tiles
(see `track.TrackLocalizer`, `localize.init: retrieval`), so retrieval only has to put the right tile in the top-k.

Embedders (`retrieval.embedder`):
    dinov2  DINOv2 ViT (HF `facebook/dinov2-small|base`), pooling cls | mean | cls_mean (L2-normalised)
    thumb   grey thumbnail, zero-mean / unit-norm (classical baseline, no weights; used in tests)
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = ROOT / "weights" / "hf"


def center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def query_images(frame: np.ndarray, scale: float, rotations=(0,)) -> list[np.ndarray]:
    """Frame resized to the map resolution, centre square, rotated clockwise by each of `rotations` (multiples of 90)."""
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    sq = center_square(small)
    rots = {0: sq, 90: cv2.rotate(sq, cv2.ROTATE_90_CLOCKWISE), 180: cv2.rotate(sq, cv2.ROTATE_180),
            270: cv2.rotate(sq, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    return [rots[int(r) % 360] for r in rotations]


class ThumbEmbedder:
    name = "thumb"

    def __init__(self, size: int = 16, **_):
        self.size = size

    def embed(self, imgs: list[np.ndarray]) -> np.ndarray:
        out = []
        for im in imgs:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            v = cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            v -= v.mean()
            out.append(v / (np.linalg.norm(v) + 1e-6))
        return np.stack(out)


def load_dinov2(model: str = "facebook/dinov2-small"):
    from transformers import AutoModel

    kw = {"cache_dir": str(HF_CACHE / "hub")} if HF_CACHE.exists() else {}
    return AutoModel.from_pretrained(model, **kw)


def build_dino_head(model: str = "facebook/dinov2-small", dim: int = 256):
    """DINOv2 backbone + projection of [normalised CLS, normalised mean patch token] -> `dim`, L2-normalised.
    Trained by scripts/finetune_dino.py; used by DinoV2Embedder(weights=...)."""
    import torch
    from torch import nn

    class DinoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = load_dinov2(model)
            h = self.backbone.config.hidden_size
            self.proj = nn.Linear(2 * h, dim)

        def forward(self, x):
            tok = self.backbone(pixel_values=x).last_hidden_state
            n = nn.functional.normalize
            f = torch.cat([n(tok[:, 0].float(), dim=1), n(tok[:, 1:].float().mean(1), dim=1)], 1)
            return n(self.proj(f), dim=1)

    return DinoHead()


class DinoV2Embedder:
    def __init__(self, device: str = "cpu", model: str = "facebook/dinov2-small", input_size: int = 224,
                 pooling: str = "cls_mean", gray: bool = False, batch: int = 32, weights: str | None = None, **_):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.fp16 = self.device.type == "cuda"
        self.head = None
        if weights:  # fine-tuned backbone + projection head
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            model, pooling = ck.get("model", model), "head"
            self.head = build_dino_head(model, ck["dim"])
            self.head.load_state_dict(ck["state_dict"])
            self.head = self.head.to(self.device).eval()
            self.model = self.head.backbone
        else:
            self.model = load_dinov2(model).to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()
            if self.head is not None:
                self.head.proj = self.head.proj.float()
        self.size = int(round(input_size / 14)) * 14  # ViT/14
        self.pooling, self.gray, self.batch = pooling, gray, batch
        tag = f",ft={Path(weights).stem}" if weights else ""
        self.name = f"dinov2[{model.split('/')[-1]},{self.size},{pooling}{',gray' if gray else ''}{tag}]"
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def _tensor(self, imgs):
        xs = []
        for im in imgs:
            if self.gray:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA
                            if min(im.shape[:2]) > self.size else cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            xs.append(((rgb - self.mean) / self.std).transpose(2, 0, 1))
        t = self.torch.from_numpy(np.stack(xs)).to(self.device)
        return t.half() if self.fp16 else t

    def embed(self, imgs: list[np.ndarray]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(imgs), self.batch):
                if self.head is not None:
                    x = self._tensor(imgs[i:i + self.batch])
                    tok = self.model(pixel_values=x).last_hidden_state.float()
                    n = self.torch.nn.functional.normalize
                    f = self.torch.cat([n(tok[:, 0], dim=1), n(tok[:, 1:].mean(1), dim=1)], 1)
                    out.append(n(self.head.proj(f), dim=1).cpu().numpy())
                    continue
                tok = self.model(pixel_values=self._tensor(imgs[i:i + self.batch])).last_hidden_state.float()
                cls, patches = tok[:, 0], tok[:, 1:].mean(1)
                n = self.torch.nn.functional.normalize
                if self.pooling == "cls":
                    v = n(cls, dim=1)
                elif self.pooling == "mean":
                    v = n(patches, dim=1)
                else:
                    v = n(self.torch.cat([n(cls, dim=1), n(patches, dim=1)], 1), dim=1)
                out.append(v.cpu().numpy())
        return np.concatenate(out)


def build_embedder(rc: dict, device: str):
    kind = rc.get("embedder", "dinov2")
    params = {k: v for k, v in rc.items() if k not in ("embedder",)}
    if kind == "dinov2":
        return DinoV2Embedder(device=device, **params)
    if kind == "thumb":
        return ThumbEmbedder(**params)
    raise ValueError(f"unknown retrieval.embedder {kind!r}")


class TileIndex:
    """Square map tiles of side `tile` px with stride `stride`, each with a global descriptor."""

    def __init__(self, map_bgr: np.ndarray, tile: int, stride: int, embedder, cache_dir: str | Path | None = None,
                 map_id: str = ""):
        H, W = map_bgr.shape[:2]
        tile = int(min(tile, W, H))
        stride = max(1, int(stride))
        xs = sorted(set(list(range(0, W - tile + 1, stride)) + [W - tile]))
        ys = sorted(set(list(range(0, H - tile + 1, stride)) + [H - tile]))
        self.tile = tile
        self.boxes = np.array([(x, y, x + tile, y + tile) for y in ys for x in xs], np.int32)
        self.centers = (self.boxes[:, :2] + tile / 2).astype(np.float32)
        key = hashlib.sha1(f"{map_id}|{W}x{H}|{tile}|{stride}|{embedder.name}".encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"tiles_{key}.npz" if cache_dir else None
        if path is not None and path.exists():
            self.emb = np.load(path)["emb"]
            log.info("retrieval: %d tiles loaded from %s", len(self.boxes), path)
            return
        crops = [map_bgr[y0:y1, x0:x1] for x0, y0, x1, y1 in self.boxes]
        self.emb = embedder.embed(crops).astype(np.float32)
        log.info("retrieval: %d tiles of %d px (stride %d) embedded with %s", len(self.boxes), tile, stride,
                 embedder.name)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, emb=self.emb)

    def scores(self, q: np.ndarray) -> np.ndarray:
        """Cosine similarity of every tile to each query row, max over the query rows (e.g. rotations)."""
        return (self.emb @ np.atleast_2d(q).T).max(1)

    def query(self, q: np.ndarray, k: int, center=None, radius: float | None = None, nms: float | None = None):
        """Top-k tile indices (best first); optionally only within `radius` of `center`; tiles closer than `nms`
        px to a better one are skipped (neighbouring tiles overlap, so the top-k would otherwise be one place)."""
        s = self.scores(q)
        if center is not None and radius is not None:
            far = np.hypot(*(self.centers - np.asarray(center, np.float32)).T) > radius
            s = np.where(far, -np.inf, s)
        order = np.argsort(-s)
        nms = self.tile / 2 if nms is None else nms
        out = []
        for i in order:
            if not np.isfinite(s[i]):
                break
            if all(np.hypot(*(self.centers[i] - self.centers[j])) >= nms for j in out):
                out.append(int(i))
            if len(out) == k:
                break
        return out, s
