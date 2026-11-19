"""Download, extract and verify the project datasets listed in data/manifest.yaml.

Everyone on the team gets byte-identical data, so experiment results are comparable.
Only the standard library + PyYAML are used (no gdown needed).

Examples (PowerShell, from the repository root):
    python scripts/download_data.py                      # all datasets from the manifest
    python scripts/download_data.py train                # one dataset
    python scripts/download_data.py train --from-zip C:/path/Trainee_2025_classic_cv.zip   # already downloaded
    python scripts/download_data.py --verify             # only check checksums of local files
    python scripts/download_data.py --hash data/video/easy.mp4   # print sha256 for the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.yaml"
CACHE = ROOT / "data" / "cache"  # downloaded archives (ignored by git)
CHUNK = 1 << 20


# ----------------------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _progress(done: int, total: int | None) -> None:
    mb = done / 2**20
    msg = f"\r  {mb:7.1f} MB" + (f" / {total / 2**20:.1f} MB ({100 * done / total:5.1f}%)" if total else "")
    sys.stdout.write(msg)
    sys.stdout.flush()


def _stream_to_file(resp, dst: Path) -> None:
    total = resp.headers.get("Content-Length")
    total = int(total) if total and total.isdigit() else None
    tmp = dst.with_suffix(dst.suffix + ".part")
    done = 0
    with open(tmp, "wb") as f:
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            f.write(block)
            done += len(block)
            _progress(done, total)
    print()
    tmp.replace(dst)


def download_url(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (dataset-downloader)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        _stream_to_file(resp, dst)


def download_gdrive(file_id: str, dst: Path) -> None:
    """Download a publicly shared Google Drive file, handling the 'can't scan for viruses' page."""
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (dataset-downloader)")]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    for _ in range(3):
        resp = opener.open(url, timeout=60)
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            _stream_to_file(resp, dst)
            return
        page = resp.read().decode("utf-8", "replace")
        form = re.search(r'<form[^>]+action="([^"]+)"', page)
        if not form:
            raise RuntimeError(
                "Google Drive returned an HTML page instead of the file. Check that the file is shared as "
                "'Anyone with the link' and the id is correct, or download it manually and use --from-zip."
            )
        fields = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
        url = html.unescape(form.group(1)) + "?" + urllib.parse.urlencode(fields)
    raise RuntimeError("Google Drive download did not start after 3 attempts")


def safe_extract(archive: Path, dest: Path, skip_prefixes: list[str]) -> int:
    """Extract a zip, skipping junk entries and refusing paths that escape `dest`."""
    dest = dest.resolve()
    n = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or any(name.startswith(p) for p in skip_prefixes):
                continue
            target = (dest / name).resolve()
            if dest not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            n += 1
    return n


def verify(ds: dict) -> tuple[int, list[str]]:
    """Returns (number of checked files, list of problems)."""
    dest = ROOT / ds["dest"]
    expected = ds.get("files") or {}
    if not expected and ds.get("filename"):
        expected = {ds["filename"]: ds.get("archive_sha256")}
    problems = []
    for rel, digest in expected.items():
        p = dest / rel
        if not p.exists():
            problems.append(f"missing: {p.relative_to(ROOT)}")
        elif digest and sha256(p) != digest:
            problems.append(f"checksum mismatch: {p.relative_to(ROOT)}")
    return len(expected), problems


# ----------------------------------------------------------------------------- main logic
def fetch(name: str, ds: dict, from_zip: Path | None, force: bool) -> None:
    print(f"[{name}] {ds.get('description', '')}")
    n, problems = verify(ds)
    if n and not problems and not force:
        print(f"[{name}] already present and verified ({n} files) -> {ds['dest']}")
        return

    single = ds.get("archive") is None
    fname = ds.get("filename") if single else ds["archive"]
    if from_zip:
        local = Path(from_zip)
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        local = CACHE / fname
        if force or not local.exists() or (ds.get("archive_sha256") and sha256(local) != ds["archive_sha256"]):
            src = ds.get("source") or {}
            print(f"[{name}] downloading {fname} ...")
            if src.get("gdrive_id"):
                download_gdrive(src["gdrive_id"], local)
            elif src.get("url"):
                download_url(src["url"], local)
            else:
                raise RuntimeError(f"[{name}] no source in the manifest")

    if ds.get("archive_sha256"):
        got = sha256(local)
        if got != ds["archive_sha256"]:
            raise RuntimeError(f"[{name}] checksum mismatch for {local}:\n  expected {ds['archive_sha256']}\n"
                               f"  got      {got}\nThe source file has changed or the download is corrupted.")
        print(f"[{name}] archive checksum OK")

    dest = ROOT / ds["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    if single:
        shutil.copyfile(local, dest / fname)
        print(f"[{name}] copied to {dest / fname}")
    else:
        k = safe_extract(local, dest, ds.get("strip_prefixes", ["__MACOSX/"]))
        print(f"[{name}] extracted {k} files -> {ds['dest']}")

    n, problems = verify(ds)
    if problems:
        raise RuntimeError(f"[{name}] verification failed:\n  " + "\n  ".join(problems))
    print(f"[{name}] verified {n} files")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help="dataset names from the manifest (default: all)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--from-zip", help="use an already downloaded archive instead of downloading")
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    ap.add_argument("--verify", action="store_true", help="only verify local files")
    ap.add_argument("--hash", nargs="+", metavar="FILE", help="print sha256 of files (to fill the manifest)")
    a = ap.parse_args()

    if a.hash:
        for f in a.hash:
            print(f"{sha256(Path(f))}  {f}")
        return 0

    with open(a.manifest, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)["datasets"]
    names = a.datasets or list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        print(f"unknown datasets: {unknown}; available: {list(manifest)}")
        return 2
    if a.from_zip and len(names) != 1:
        print("--from-zip needs exactly one dataset name")
        return 2

    failed = False
    for name in names:
        try:
            if a.verify:
                n, problems = verify(manifest[name])
                status = "OK" if not problems else "FAILED\n  " + "\n  ".join(problems)
                print(f"[{name}] {n} files: {status}")
                failed |= bool(problems)
            else:
                fetch(name, manifest[name], Path(a.from_zip) if a.from_zip else None, a.force)
        except Exception as e:  # report and continue with other datasets
            print(f"[{name}] ERROR: {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
