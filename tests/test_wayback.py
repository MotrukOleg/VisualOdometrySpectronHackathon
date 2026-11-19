import json

import numpy as np

from vo import wayback

# five releases, oldest first; the imagery at our tile changed in r2 and r4, r5 only re-encoded r4's image
RELEASES = {f"r{k}": {"itemTitle": f"World Imagery (Wayback 20{10 + k}-01-01)", "metadataLayerUrl": ""}
            for k in range(1, 6)}
LAST_CHANGE = {"r5": "r5", "r4": "r4", "r3": "r2", "r2": "r2", "r1": "r1"}
IMAGE = {"r5": 4, "r4": 4, "r2": 2, "r1": 1}  # r5 and r4 look identical


def test_versions_at_keeps_only_distinct_imagery(monkeypatch, tmp_path):
    (tmp_path / "wayback").mkdir()
    (tmp_path / "wayback" / "config.json").write_text(json.dumps(RELEASES))
    asked = []

    def fake_download(url, retries=3):
        rel = url.split("/tilemap/")[1].split("/")[0]
        asked.append(rel)
        return json.dumps({"data": [1], "select": [LAST_CHANGE[rel]]}).encode()

    monkeypatch.setattr(wayback, "_download", fake_download)
    monkeypatch.setattr(wayback, "fetch_tile",
                        lambda provider, z, x, y, cache: np.full((4, 4, 3), IMAGE[provider.split(":")[1]], np.uint8))
    vs = wayback.versions_at(49.84, 24.03, 17, tmp_path, with_capture=False)
    assert [v.release.id for v in vs] == ["r5", "r2", "r1"]  # newest first, r4 duplicates r5
    assert "r2" not in asked  # r3's tilemap points at r2, so the walk jumps straight to r1
    assert vs[0].provider == "wayback:r5" and vs[-1].tag == "r2011-01-01"
