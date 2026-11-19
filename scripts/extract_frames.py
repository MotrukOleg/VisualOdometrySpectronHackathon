"""Extract frames from a video into a directory (every k-th frame)."""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vo.io import imwrite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ext", default=".jpg")
    a = ap.parse_args()
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video} (non-ASCII paths may fail with cv2.VideoCapture on Windows)")
    k = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % a.every == 0:
            imwrite(Path(a.out) / f"frame_{k:06d}{a.ext}", frame)
            saved += 1
        k += 1
    print(f"saved {saved} of {k} frames to {a.out}")


if __name__ == "__main__":
    main()
