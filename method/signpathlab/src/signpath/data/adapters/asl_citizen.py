"""Actual ASL Citizen CSV columns; preserve the supplied official split."""
import csv
from pathlib import Path

from ...cache.hashing import file_hash
from ...cli import write_json
from ..manifest import ManifestRecord, write_manifest


def ingest(root, output, *, hash_videos=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    tables, csv_hashes = {}, {}
    for split in ("train", "val", "test"):
        path = root / "splits" / f"{split}.csv"
        csv_hashes[split] = file_hash(path)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"Participant ID", "Video file", "Gloss"} <= set(reader.fieldnames or []):
                raise ValueError(f"Not an ASL Citizen official CSV: {path}")
            tables[split] = list(reader)
    # Train-only vocabulary; held-out labels never determine a fitted taxonomy.
    labels = {label: index for index, label in enumerate(sorted({r["Gloss"] for r in tables["train"]}))}
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"asl_citizen.{split}.jsonl").exists() for split in tables):
        raise FileExistsError(f"Refusing to overwrite manifests in {output}")
    summary = {"dataset": "asl_citizen", "csv_sha256": csv_hashes,
               "source_root": str(root), "video_hashes_computed": hash_videos, "splits": {}}
    for split, table in tables.items():
        records = []
        for row in table:
            relative = Path(row["Video file"])
            path = (root / "videos" / relative).resolve()
            if not path.is_relative_to(root / "videos"):
                raise ValueError(f"Video path escapes dataset root: {relative}")
            if row["Gloss"] not in labels:
                raise ValueError(f"Held-out class absent from train taxonomy: {row['Gloss']}")
            exists = path.is_file()
            records.append(ManifestRecord(
                1, f"asl_citizen:{relative.stem}", "asl_citizen", "ase", split,
                f"asl_citizen:{relative.stem}", str(path),
                source_sha256=file_hash(path) if exists and hash_videos else None,
                signer_id=row["Participant ID"] or None,
                label_id=labels[row["Gloss"]], gloss=row["Gloss"],
                official_split_version=f"csv_sha256:{csv_hashes[split]}",
                availability="available" if exists else "missing",
                metadata={"annotation_sha256": csv_hashes[split], "asl_lex_code": row.get("ASL-LEX Code")}))
        manifest = output / f"asl_citizen.{split}.jsonl"
        write_manifest(manifest, records)
        summary["splits"][split] = {"count": len(records), "available": sum(r.availability == "available" for r in records),
                                    "manifest_sha256": file_hash(manifest)}
    write_json(output / "labels.json", labels)
    write_json(output / "import_report.json", summary)
    return summary
