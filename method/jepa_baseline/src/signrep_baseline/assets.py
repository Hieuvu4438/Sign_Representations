"""Retrieve hash-pinned original public weights; no weights are committed to source."""

import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from .cache import sha256

ROOT = Path(__file__).resolve().parents[2]
RELEASED_URL = "https://github.com/ryanwongsa/SignRep/releases/download/v0.0.1/ckpt.pt"
RELEASED_SHA = "f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b"


def download_pinned(url, path, expected, google_drive=False):
    path = Path(path)
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(f"Existing asset hash mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        if google_drive:
            import gdown
            gdown.download(url, str(temporary), fuzzy=True, quiet=False)
        else:
            urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded asset hash mismatch: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def download_assets(kind="all", destination=None):
    output = Path(destination or ROOT / "runtime")
    result = {}
    if kind in {"all", "released"}:
        path = download_pinned(RELEASED_URL, output / "released/ckpt.pt", RELEASED_SHA)
        result["released"] = str(path.resolve())
    if kind in {"all", "kinetics"}:
        pin = json.loads((ROOT / "evidence/kinetics_weights.json").read_text())
        path = download_pinned(pin["url"], output / "kinetics/mae_hiera_base_16x224.pth", pin["sha256"])
        result["kinetics"] = str(path.resolve())
    if kind in {"all", "teacher"}:
        pin = json.loads((ROOT / "evidence/teacher_weights.json").read_text())
        archive = download_pinned(pin["source"], output / "teacher/models.zip", pin["archive_sha256"], google_drive=True)
        with zipfile.ZipFile(archive) as file:
            for name, record in pin["weights"].items():
                path = output / "teacher" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    temporary = path.with_suffix(".download")
                    with file.open(name) as source, temporary.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                    if sha256(temporary) != record["sha256"]:
                        temporary.unlink()
                        raise ValueError("Extracted teacher weight hash mismatch")
                    temporary.replace(path)
                if sha256(path) != record["sha256"]:
                    raise ValueError("Existing teacher weight hash mismatch")
        result["teacher"] = str((output / "teacher/models").resolve())
    return result
