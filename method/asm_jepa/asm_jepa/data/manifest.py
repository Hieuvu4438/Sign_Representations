"""Official ASL-Citizen split CSVs -> canonical manifests, train-only vocabulary and split audit."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from asm_jepa.utils import read_json, read_jsonl

SPLITS = ("train", "val", "test")


class ProtocolError(RuntimeError):
    pass


def sample_id_for(relative_filename: str) -> str:
    return hashlib.sha256(relative_filename.encode("utf-8")).hexdigest()[:24]


def read_official_csv(path: Path, headers: dict, positions: dict) -> tuple[list[dict], dict]:
    """Read with csv.DictReader; fall back to the official positional order when headers differ."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = [r for r in reader if any(cell.strip() for cell in r)]
    if all(headers[k] in header for k in ("signer", "filename", "gloss")):
        idx = {k: header.index(headers[k]) for k in ("signer", "filename", "gloss")}
        mode = "header"
    else:
        idx = {k: int(positions[k]) for k in ("signer", "filename", "gloss")}
        mode = "positional"
    rows = [{k: r[i] for k, i in idx.items()} for r in body]
    return rows, {"mode": mode, "header": header, "columns": {k: header[i] if i < len(header) else None
                                                              for k, i in idx.items()}}


def build_vocab(train_rows: list[dict]) -> dict[str, int]:
    return {g: i for i, g in enumerate(sorted({r["gloss"].strip() for r in train_rows}))}


def build_split_rows(split: str, rows: list[dict], vocab: dict, videos_dir: str) -> list[dict]:
    out, unknown = [], []
    for r in rows:
        gloss = r["gloss"].strip()
        if gloss not in vocab:
            unknown.append(gloss)
            continue
        rel = f"{videos_dir}/{r['filename']}"
        sid = sample_id_for(rel)
        out.append({
            "sample_id": sid,
            "official_split": split,
            "signer_id": r["signer"],
            "video_relpath": rel,
            "source_stem": Path(r["filename"]).stem,
            "gloss": gloss,
            "class_id": vocab[gloss],
            "pose_relpath": f"poses/{sid}.npz",
            "duration_sec": None,
            "fps": None,
            "num_frames": None,
            "width": None,
            "height": None,
            "decode_status": "not_checked",
        })
    if unknown:
        raise ProtocolError(f"{split}: {len(unknown)} glosses absent from the train vocabulary, e.g. {sorted(set(unknown))[:10]}")
    return out


def audit_splits(split_rows: dict[str, list[dict]]) -> dict:
    vids = {s: {r["sample_id"] for r in rows} for s, rows in split_rows.items()}
    signers = {s: {r["signer_id"] for r in rows} for s, rows in split_rows.items()}
    audit = {"counts": {s: len(rows) for s, rows in split_rows.items()},
             "unique_ids": {s: len(v) for s, v in vids.items()},
             "signers": {s: sorted(v) for s, v in signers.items()},
             "num_signers": {s: len(v) for s, v in signers.items()},
             "num_glosses": {s: len({r["class_id"] for r in rows}) for s, rows in split_rows.items()},
             "overlaps": {}}
    names = list(split_rows)
    ok = True
    for i, a in enumerate(names):
        if audit["unique_ids"][a] != audit["counts"][a]:
            ok = False
            audit.setdefault("errors", []).append(f"duplicate sample ids within {a}")
        for b in names[i + 1:]:
            v = len(vids[a] & vids[b])
            s = sorted(signers[a] & signers[b])
            audit["overlaps"][f"{a}_{b}"] = {"videos": v, "signers": s}
            ok &= v == 0 and not s
    audit["disjoint"] = ok
    return audit


def load_split(processed_root: Path, split: str) -> list[dict]:
    path = Path(processed_root) / f"metadata_{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run `python -m asm_jepa.cli.prepare_aslc` first")
    return read_jsonl(path)


def load_vocab(processed_root: Path) -> dict[str, int]:
    return read_json(Path(processed_root) / "vocab.json")["gloss_to_class_id"]
