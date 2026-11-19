from .base import Extractor, Features
from .classical import ClassicalExtractor
from .learned import LG_EXTRACTORS

CLASSICAL = {"sift", "rootsift", "orb", "akaze", "brisk", "kaze"}


def build_extractor(cfg: dict, device: str = "cpu") -> Extractor:
    kind = cfg["type"].lower()
    if kind in CLASSICAL:
        return ClassicalExtractor(kind, cfg.get("max_keypoints", 4000), cfg.get("params"))
    if kind in LG_EXTRACTORS:
        from .learned import LightGlueExtractor

        return LightGlueExtractor(kind, cfg.get("max_keypoints", 2048), device, cfg.get("params"))
    raise ValueError(f"Unknown feature type {kind!r}; options: {sorted(CLASSICAL | set(LG_EXTRACTORS))}")


__all__ = ["Extractor", "Features", "build_extractor", "CLASSICAL", "LG_EXTRACTORS"]
