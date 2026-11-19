from ..features import build_extractor
from ..preprocess import Preprocessor
from .base import Correspondences, FeatureMatcher, ImagePairMatcher, PairMatcher
from .descriptor import BFMatcher, FLANNMatcher

DESCRIPTOR_MATCHERS = {"bf", "flann", "lightglue"}
IMAGE_MATCHERS = {"loftr", "xfeat", "xfeat_star", "superglue"}


def build_pair_matcher(cfg: dict, device: str = "cpu") -> PairMatcher:
    """Build the frame-pair matcher described by the full experiment config."""
    m = cfg["matching"]
    kind = m["type"].lower()
    pre = Preprocessor(**cfg["preprocess"])
    params = m.get("params") or {}

    if kind in DESCRIPTOR_MATCHERS:
        extractor = build_extractor(cfg["features"], device)
        if kind == "bf":
            dm = BFMatcher(m.get("ratio"), m.get("cross_check", False))
        elif kind == "flann":
            dm = FLANNMatcher(m.get("ratio"), m.get("cross_check", False), **params)
        else:
            from .learned import LightGlueMatcher

            dm = LightGlueMatcher(cfg["features"]["type"].lower(), device, params)
        return FeatureMatcher(extractor, dm, pre)

    from . import learned

    rl = m.get("resize_long")
    if kind == "loftr":
        return learned.LoFTRMatcher(device, rl, preprocess=pre, **params)
    if kind in ("xfeat", "xfeat_star"):
        return learned.XFeatMatcher(device, rl, star=kind == "xfeat_star", preprocess=pre, **params)
    if kind == "superglue":
        return learned.SuperGlueMatcher(device, rl, preprocess=pre, **params)
    raise ValueError(f"Unknown matcher {kind!r}; options: {sorted(DESCRIPTOR_MATCHERS | IMAGE_MATCHERS)}")


__all__ = ["Correspondences", "PairMatcher", "FeatureMatcher", "ImagePairMatcher", "build_pair_matcher"]
