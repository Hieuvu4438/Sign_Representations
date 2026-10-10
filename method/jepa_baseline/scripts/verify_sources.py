"""Verify every pinned source and every import-only namespace adaptation."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    sources = json.loads((ROOT / "evidence/sources.json").read_text())
    sources.update(json.loads((ROOT / "evidence/translation_sources.json").read_text()))
    sources["planckian_jitter"] = json.loads((ROOT / "evidence/planckian_source.json").read_text())
    sources["asl_citizen_metrics"] = json.loads((ROOT / "evidence/asl_citizen_source.json").read_text())
    checked = 0
    for name, record in sources.items():
        for relative, expected in record.get("files", {}).items():
            path = ROOT / "vendor" / name / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
            checked += 1
        for relative, target in record.get("symlinks", {}).items():
            assert os.readlink(ROOT / "vendor" / name / relative) == target
    for relative in ["planckianTransforms.py", "LICENSE.txt"]:
        assert (ROOT / "vendor/planckian_jitter" / relative).read_bytes() == (ROOT / "src/signrep_baseline/upstream_planckian" / relative).read_bytes()
    for relative, record in json.loads((ROOT / "evidence/namespace_adaptation.json").read_text()).items():
        original = (ROOT / "vendor/signrep" / relative).read_text()
        expected = original.replace("from models.", "from signrep_baseline.upstream_signrep.models.")
        expected = expected.replace("from augmentation.", "from signrep_baseline.upstream_signrep.augmentation.")
        expected = expected.replace('"models.head_models.', '"signrep_baseline.upstream_signrep.models.head_models.')
        adapted = ROOT / "src/signrep_baseline/upstream_signrep" / relative
        assert adapted.read_text() == expected, adapted
        assert hashlib.sha256(adapted.read_bytes()).hexdigest() == record["adapted_sha256"], adapted
    return {"pinned_files_checked": checked, "namespace_adaptation": "import-only", "status": "PASS"}


if __name__ == "__main__":
    report = verify()
    (ROOT / "evidence/source_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
