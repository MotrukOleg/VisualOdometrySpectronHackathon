"""What the demo offers: localisation methods (map configs) and ready-made datasets (presets)."""
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "data" / "train" / "global_map.png"


@dataclass
class Method:
    id: str
    label: str
    config: str  # relative to the repository root
    note: str
    needs: tuple[str, ...] = ()  # python packages beyond the classical pipeline

    def available(self) -> bool:
        return all(importlib.util.find_spec(p) is not None for p in self.needs)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["available"] = self.available()
        return d


METHODS = [
    Method("baseline", "Бейзлайн: SIFT + FLANN, вся мапа", "configs/map/baseline_sift.yaml",
           "CPU, найшвидший; кожен кадр шукається по всій мапі незалежно"),
    Method("final", "Фінальний: SIFT + LightGlue, трекінг", "configs/map/final.yaml",
           "GPU бажано; стійкий на складній місцевості, ~1 кадр/с на відео", ("torch", "lightglue")),
    Method("track_sift", "SIFT + FLANN, трекінг", "configs/map/methods/track_sift.yaml",
           "CPU; пошук у вікні навколо прогнозу з деротацією"),
    Method("track_akaze", "AKAZE + BF, трекінг", "configs/map/methods/track_akaze.yaml", "CPU"),
    Method("track_orb", "ORB + BF, трекінг", "configs/map/methods/track_orb.yaml", "CPU"),
    Method("track_superpoint_lg", "SuperPoint + LightGlue, трекінг", "configs/map/methods/track_superpoint_lg.yaml",
           "GPU; ваги некомерційні (Magic Leap)", ("torch", "lightglue")),
    Method("track_disk_lg", "DISK + LightGlue, трекінг", "configs/map/methods/track_disk_lg.yaml",
           "GPU, багато відеопам'яті", ("torch", "lightglue")),
    Method("track_loftr", "LoFTR, трекінг", "configs/map/methods/track_loftr.yaml", "GPU", ("torch", "kornia")),
]
METHODS_BY_ID = {m.id: m for m in METHODS}

PRESET_LABELS = {
    "crops": "Задача з ТЗ: 50 кадрів і мапа",
    "syn_easy": "Синтетика над мапою, легка (є GT)",
    "syn_hard": "Синтетика «з екрана», важка (є GT)",
    "video0061": "Відео IMG_0061: петля",
    "video0062": "Відео IMG_0062: вертикальне, складне",
    "video214816": "Третє відео: сильний крен",
    "area_lviv": "Район: Львів, старе місто (є GT)",
    "area_kyiv": "Район: Київ, Дніпро (є GT)",
    "area_carpathians": "Район: Карпати, Буковель (є GT)",
    "area_poltava": "Район: поля Полтавщини (є GT)",
    "area_oleshky": "Район: Олешківські піски (є GT)",
    "area_gorgany": "Район: Горгани, ліс (є GT)",
    "area_kyiv_sea": "Район: Київське море (є GT)",
    "area_askania": "Район: степ Асканії-Нова (є GT)",
    "area_tylihul": "Район: Тилігульський лиман (є GT)",
}


def presets() -> list[dict]:
    """Datasets from configs/datasets.yaml whose frames exist locally (data/ is not versioned)."""
    path = ROOT / "configs" / "datasets.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        sets = yaml.safe_load(f) or {}
    out = []
    for key, d in sets.items():
        if key.endswith("s2"):  # subsampled duplicates used only by the benchmark
            continue
        frames = ROOT / d["frames"]
        map_path = ROOT / d["map"] if d.get("map") else DEFAULT_MAP
        if not frames.is_dir() or not map_path.exists():
            continue
        label = PRESET_LABELS.get(key) or (f"Район {key[5:]} (є GT)" if key.startswith("area_") else key)
        gt = ROOT / d["gt"] if d.get("gt") else None
        kind = "synthetic" if key.startswith(("syn_", "area_")) else "real_nogt"
        out.append({"id": key, "label": label, "frames": str(frames), "map": str(map_path),
                    "gt": str(gt) if gt and gt.exists() else None, "frame_scale": d.get("frame_scale", 0.5),
                    "step": d.get("step", 1), "geo": (map_path.parent / "geo.json").exists(), "kind": kind})
    real = ROOT / "data" / "real" / "visloc_24"  # scripts/import_uavvisloc.py --flight data/uavvisloc/24 --zoom 19
    if (real / "flight" / "frames").is_dir() and (real / "map.jpg").exists():
        out.append({"id": "visloc_24", "label": "Реальний політ дрона UAV-VisLoc #24 (є GT)",
                    "frames": str(real / "flight" / "frames"), "map": str(real / "map.jpg"),
                    "gt": str(real / "flight" / "gt.csv"), "frame_scale": 0.3, "step": 1, "geo": True,
                    "kind": "real"})
    return sorted(out, key=lambda p: p["id"] != "crops")  # the task first
