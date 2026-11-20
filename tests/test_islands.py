"""Island localisation: stage 1 (frame chains) and stage 3 (island placement from noisy candidates)."""
import numpy as np
import pandas as pd

from vo.config import load_config
from vo.io import FrameSource, imwrite
from vo.island_fit import fit_island, umeyama2d
from vo.islands import build_islands
from vo.retrieval import ThumbEmbedder, TileIndex, query_images
from vo.synthetic import make_flight, make_world


def _similarity_residual(src, dst):
    S = umeyama2d(src, dst)
    return np.hypot(*((src @ S[:2, :2].T + S[:2, 2]) - dst).T)


def test_islands_chain_follows_the_flight(tmp_path):
    world = make_world(1600, 1200, seed=3)
    imwrite(tmp_path / "map.png", world)
    # 60 frames per half loop -> overlapping consecutive frames; the first 15 are used
    data = make_flight(tmp_path / "syn", n_frames=60, frame_size=(400, 300), turns=0.5, tilt_std=0.0, seed=3,
                       world_path=tmp_path / "map.png", frame_scale=1.0)
    src = FrameSource(data / "frames", 0, 1, 15, resize_long=None)
    islands, links = build_islands(src, load_config(None, ["islands.geometry.min_inliers=10"]))
    assert len(islands) == 1 and len(islands[0].frames) == 15
    q = islands[0].centers(lambda k: (400, 300))
    gt = pd.read_csv(data / "gt.csv")[["x", "y"]].to_numpy()[:15]
    # island geometry = flight geometry up to one similarity (island px -> map px)
    assert np.median(_similarity_residual(q, gt)) < 1.5


def test_fit_island_recovers_placement_from_mostly_wrong_candidates():
    rng = np.random.default_rng(0)
    q = np.stack([np.arange(12) * 400.0, np.zeros(12)], 1)  # 12 frames on a line, island px
    th, s, t = np.radians(30), 0.25, np.array([1500.0, 800.0])
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    true = q @ R.T + t
    cands, sims = [], []
    for k in range(12):
        c = rng.uniform(0, 4000, (10, 2))  # 10 random candidates over a 4000 px map
        if k % 3 == 0:  # only every 3rd frame has the right tile somewhere in its top-10
            c[rng.integers(10)] = true[k] + rng.normal(0, 10, 2)
        cands.append(c)
        sims.append(np.linspace(1, 0.5, 10))
    for heading in (None, 30.0):
        f = fit_island(q, cands, sims, scale0=0.25, r_inlier=40, heading_deg=heading)
        P = q @ f["S"][:2, :2].T + f["S"][:2, 2]
        assert f["n_inliers"] >= 4
        assert np.median(np.hypot(*(P - true).T)) < 15


def test_tile_index_finds_the_crop():
    world = make_world(1200, 900, seed=1)
    emb = ThumbEmbedder(size=16)
    idx = TileIndex(world, 200, 50, emb)
    for x, y in [(700, 300), (200, 500), (900, 600)]:
        frame = world[y:y + 200, x:x + 200]  # a tile-sized crop at scale 1
        top, _ = idx.query(emb.embed(query_images(frame, 1.0)), 5)
        assert np.hypot(*(idx.centers[top[0]] - [x + 100, y + 100])) <= 50
