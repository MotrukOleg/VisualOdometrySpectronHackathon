"""Fine-tune DINOv2 for frame -> map tile retrieval (stage 2 of scripts/localize_islands.py).

Training pairs = (UAV frame rotated north-up and resized to the map resolution, centre square) <-> (map crop of the
same ground area around the frame's true centre). Sources with known positions, never the test flight:
    data/areas/<area>/        exact GT (x, y, yaw_deg, scale) - Esri zoom 18/19 frames vs zoom 17 map
    experiments/results/final_*/trajectory.csv   task crops + our videos positioned by the final method (pseudo-GT)
Loss: symmetric InfoNCE with in-batch negatives; every batch comes from ONE map, so the negatives are other places
of the same terrain. Augmentation: colour / gamma / blur / grey on the frame, blur on the map, scale +-18 % and a shift
of up to 1/3 tile on the map crop, the same random 90-degree turn and flip on both. Only the last `--train-blocks`
ViT blocks and the projection head are trained (GTX 1650: fp16 autocast, batch 32).

    python scripts/finetune_dino.py --out weights/retrieval/dinov2s_ft.pt --val-areas gorgany_forest tylihul_estuary
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vo.io import FrameSource, imread  # noqa: E402
from vo.retrieval import TileIndex, build_dino_head, DinoV2Embedder  # noqa: E402
from vo.track import prepare_frame  # noqa: E402

Q, MARGIN = 224, 1.4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(img, c, side, out):
    """Axis-aligned square crop of `side` px around c, resized to out x out; None if it leaves the image."""
    x0, y0 = int(round(c[0] - side / 2)), int(round(c[1] - side / 2))
    s = int(round(side))
    if x0 < 0 or y0 < 0 or x0 + s > img.shape[1] or y0 + s > img.shape[0] or s < 16:
        return None
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (out, out), interpolation=cv2.INTER_AREA if s > out else
                      cv2.INTER_LINEAR)


def pair(frame, map_bgr, x, y, heading, scale):
    img, A, _ = prepare_frame(frame, scale, heading, {})
    h, w = frame.shape[:2]
    c = A @ [w / 2, h / 2, 1]
    t = math.radians(heading)
    side = min(w, h) * scale / (abs(math.cos(t)) + abs(math.sin(t)))  # inscribed square: no black corners
    q = crop(img, c[:2], side, Q)
    m = crop(map_bgr, (x, y), side * MARGIN, int(round(Q * MARGIN)))
    return (q, m) if q is not None and m is not None else None


def sources():
    out = []
    for a in sorted((ROOT / "data" / "areas").glob("*/flight/gt.csv")):
        area = a.parents[1]
        g = pd.read_csv(a)
        out.append((area.name, area / "flight" / "frames", area / "map.jpg",
                    g[["name", "x", "y", "yaw_deg", "scale"]].rename(columns={"yaw_deg": "heading"})))
    for run in sorted((ROOT / "experiments" / "results").glob("final_*")):
        c = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        t = pd.read_csv(run / "trajectory.csv")
        t = t[t.status == "map"]
        out.append((run.name, ROOT / c["data"]["frames"], ROOT / c["map"]["path"],
                    t[["name", "x", "y", "heading_deg", "scale"]].rename(columns={"heading_deg": "heading"})))
    return out


def build_pairs(srcs):
    data = []  # (group, q, m)
    for name, frames, map_path, df in srcs:
        mp = imread(map_path)
        fs = FrameSource(frames, 0, 1, None, resize_long=None)
        lut = {fs.name(k): k for k in range(len(fs))}
        n0 = len(data)
        for r in df.itertuples():
            k = lut.get(str(r.name))
            if k is None:
                continue
            p = pair(fs.load(k), mp, r.x, r.y, r.heading, r.scale)
            if p is not None:
                data.append((name, *p))
        print(f"  {name}: {len(data) - n0} pairs")
    return data


def augment(q, m, rng):
    # map crop: random scale (+-18 %) and shift (<= 1/3 tile) inside the margin, then to Q x Q
    M = m.shape[0]
    s = Q * rng.uniform(0.85, 1.18)
    s = min(s, M - 2)
    sh = (M - s) / 2
    dx, dy = rng.uniform(-1, 1, 2) * min(sh, Q / 3)
    c = (M / 2 + dx, M / 2 + dy)
    x0, y0 = int(np.clip(c[0] - s / 2, 0, M - s)), int(np.clip(c[1] - s / 2, 0, M - s))
    m = cv2.resize(m[y0:y0 + int(s), x0:x0 + int(s)], (Q, Q), interpolation=cv2.INTER_AREA)
    q = q.astype(np.float32) / 255
    # photometric: UAV vs satellite differ in colour, contrast, sharpness
    q = q * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)
    q = (q - q.mean()) * rng.uniform(0.7, 1.4) + q.mean()
    hsv = cv2.cvtColor(np.clip(q, 0, 1), cv2.COLOR_BGR2HSV)
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 360
    q = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0, 1) ** rng.uniform(0.75, 1.33)
    if rng.random() < 0.5:
        q = cv2.GaussianBlur(q, (0, 0), rng.uniform(0.3, 2.0))
    if rng.random() < 0.2:
        q = np.repeat(cv2.cvtColor(q.astype(np.float32), cv2.COLOR_BGR2GRAY)[..., None], 3, 2)
    m = m.astype(np.float32) / 255
    if rng.random() < 0.3:
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.3, 1.2))
    r = rng.integers(4)
    q, m = np.rot90(q, r), np.rot90(m, r)
    if rng.random() < 0.5:
        q, m = q[:, ::-1], m[:, ::-1]
    return q, m


def to_tensor(imgs, torch, device):
    x = np.stack([((cv2.cvtColor(np.ascontiguousarray(i, np.float32), cv2.COLOR_BGR2RGB) - MEAN) / STD)
                  .transpose(2, 0, 1) for i in imgs])
    return torch.from_numpy(x).to(device)


def eval_areas(model, areas, torch, device):
    """recall@1/5 of map-tile retrieval on held-out areas (frames derotated by the GT heading, as with a compass)."""
    emb = DinoV2Embedder.__new__(DinoV2Embedder)
    emb.torch, emb.device, emb.fp16, emb.size, emb.gray, emb.batch = torch, device, True, Q, False, 32
    emb.mean, emb.std, emb.pooling, emb.name = MEAN, STD, "head", f"eval{time.time()}"
    emb.head, emb.model = model, model.backbone
    res = {}
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        for a in areas:
            d = ROOT / "data" / "areas" / a
            mp = imread(d / "map.jpg")
            g = pd.read_csv(d / "flight" / "gt.csv")
            fs = FrameSource(d / "flight" / "frames", 0, 1, None, resize_long=None)
            f0 = fs.load(0)
            side = int(round(min(f0.shape[:2]) * float(g.scale.iloc[0]) / math.sqrt(2)))
            idx = TileIndex(mp, side, max(1, side // 3), emb)
            hits = np.zeros(2)
            for k in range(len(fs)):
                r = g.iloc[k]
                img, A, _ = prepare_frame(fs.load(k), r.scale, r.yaw_deg, {})
                c = A @ [f0.shape[1] / 2, f0.shape[0] / 2, 1]
                q = crop(img, c[:2], side, Q)
                top, _ = idx.query(emb.embed([q]), 5)
                dd = np.hypot(*(idx.centers[top] - [r.x, r.y]).T)
                hits += [dd[0] <= side / 2, (dd <= side / 2).any()]
            res[a] = (hits / len(fs) * 100).round(1).tolist()
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="weights/retrieval/dinov2s_ft.pt")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--train-blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--val-areas", nargs="*", default=["gorgany_forest", "tylihul_estuary"])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srcs = [s for s in sources() if s[0] not in a.val_areas]
    print("building pairs (val areas held out: %s)" % ", ".join(a.val_areas))
    data = build_pairs(srcs)
    groups = {}
    for i, (g, _, _) in enumerate(data):
        groups.setdefault(g, []).append(i)
    groups = {g: v for g, v in groups.items() if len(v) >= 8}
    weights = np.array([len(v) for v in groups.values()], float)
    weights /= weights.sum()
    print(f"{len(data)} pairs in {len(groups)} groups")

    model = build_dino_head(a.model, a.dim).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    blocks = model.backbone.encoder.layer[-a.train_blocks:] if a.train_blocks else []
    for b in blocks:
        for p in b.parameters():
            p.requires_grad = True
    for p in model.backbone.layernorm.parameters():
        p.requires_grad = True
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr}, {"params": model.proj.parameters(), "lr": a.lr_head}],
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) *
                                              0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    log = {"args": vars(a), "eval": {}}
    if a.val_areas:
        log["eval"][0] = eval_areas(model, a.val_areas, torch, device)
        print("step 0 val recall@1/@5:", log["eval"][0], flush=True)
    model.train()
    t0, run = time.time(), 0.0
    gnames = list(groups)
    for step in range(1, a.steps + 1):
        g = gnames[rng.choice(len(gnames), p=weights)]
        ids = rng.choice(groups[g], min(a.batch, len(groups[g])), replace=False)
        qs, ms = zip(*(augment(data[i][1], data[i][2], rng) for i in ids))
        xq, xm = to_tensor(qs, torch, device), to_tensor(ms, torch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            eq, em = model(xq), model(xm)
            logits = (eq @ em.T) / a.tau
            y = torch.arange(len(ids), device=device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, y) +
                          torch.nn.functional.cross_entropy(logits.T, y))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0:
            print(f"step {step}  loss {run:.3f}  {time.time() - t0:.0f} s  "
                  f"vram {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)
        if a.val_areas and (step % a.eval_every == 0 or step == a.steps):
            log["eval"][step] = eval_areas(model, a.val_areas, torch, device)
            print(f"step {step} val recall@1/@5:", log["eval"][step], flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": a.model, "dim": a.dim, "state_dict": model.state_dict(), "args": vars(a)}, out)
    out.with_suffix(".json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
