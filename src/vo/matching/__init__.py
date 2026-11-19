from ..features import build_extractor
from ..preprocess import Preprocessor
from .base import Correspondences, FeatureMatcher, ImagePairMatcher, PairMatcher
from .descriptor import BFMatcher, FLANNMatcher

DESCRIPTOR_MATCHERS = {"bf", "flann", "lightglue", "adalam"}
IMAGE_MATCHERS = {"loftr", "xfeat", "xfeat_star", "superglue", "superglue_hf", "eloftr", "roma_tiny", "roma"}


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
        elif kind == "adalam":
            from .learned import AdaLAMMatcher

            dm = AdaLAMMatcher(device, params)
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
    if kind == "superglue_hf":
        return learned.HFKeypointMatcher("magic-leap-community/superglue_outdoor", device, rl, preprocess=pre, **params)
    if kind == "eloftr":
        return learned.HFKeypointMatcher("zju-community/efficientloftr", device, rl, preprocess=pre, **params)
    if kind in ("roma_tiny", "roma"):
        return learned.RoMaMatcher(device, rl, variant="tiny" if kind == "roma_tiny" else "full", preprocess=pre,
                                   **params)
    raise ValueError(f"Unknown matcher {kind!r}; options: {sorted(DESCRIPTOR_MATCHERS | IMAGE_MATCHERS)}")


__all__ = ["Correspondences", "PairMatcher", "FeatureMatcher", "ImagePairMatcher", "build_pair_matcher"]
