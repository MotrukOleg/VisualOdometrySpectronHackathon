"""Pre-download the weights of every learned method used in the benchmark (torch hub / HF caches).

Run once on a new machine (e.g. a vast.ai instance) so that experiments do not stall on downloads.
    python scripts/download_weights.py            # everything except full RoMa
    python scripts/download_weights.py --roma-full
"""
import argparse
import sys
import time
import traceback


def step(name, fn):
    t = time.time()
    try:
        fn()
        print(f"[ok]   {name} ({time.time() - t:.0f}s)", flush=True)
    except Exception:
        print(f"[FAIL] {name}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roma-full", action="store_true", help="also RoMa outdoor + DINOv2 ViT-L (~1.6 GB)")
    a = ap.parse_args()
    import torch

    def lg():
        import lightglue as L

        L.SuperPoint()
        L.LightGlue(features="superpoint")

    step("SuperPoint + LightGlue", lg)
    step("DISK + LightGlue", lambda: (__import__("lightglue").DISK(), __import__("lightglue").LightGlue(features="disk")))
    step("ALIKED + LightGlue", lambda: (__import__("lightglue").ALIKED(),
                                        __import__("lightglue").LightGlue(features="aliked")))
    step("ALIKED-n16rot", lambda: __import__("lightglue").ALIKED(model_name="aliked-n16rot"))
    step("SIFT + LightGlue", lambda: __import__("lightglue").LightGlue(features="sift"))

    def loftr():
        import kornia.feature as KF

        KF.LoFTR(pretrained="outdoor")

    step("LoFTR outdoor (kornia)", loftr)

    def dedode():
        import kornia.feature as KF

        KF.DeDoDe.from_pretrained("L-C4-v2", "B-upright")

    step("DeDoDe v2 (kornia)", dedode)

    def hf(mid):
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching

        AutoImageProcessor.from_pretrained(mid)
        AutoModelForKeypointMatching.from_pretrained(mid)

    step("Efficient LoFTR (HF)", lambda: hf("zju-community/efficientloftr"))
    step("SuperGlue outdoor (HF)", lambda: hf("magic-leap-community/superglue_outdoor"))
    step("XFeat (torch.hub)", lambda: torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True,
                                                    top_k=4096, trust_repo=True))

    def tiny_roma():
        import romatch

        romatch.tiny_roma_v1_outdoor(device=torch.device("cpu"))

    step("tiny-RoMa", tiny_roma)
    if a.roma_full:
        step("RoMa outdoor", lambda: __import__("romatch").roma_outdoor(device=torch.device("cpu")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
