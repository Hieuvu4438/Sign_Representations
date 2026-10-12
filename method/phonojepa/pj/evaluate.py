"""Score PhonoJEPA features with the unchanged proposal11 evaluation (method/obs_jepa/oj/evaluate.py), so numbers are
directly comparable with SR3D / SRVJ / ASLC002 / JV3D / JH3D / A00 / obs_jepa A & B.

oj.evaluate reads runtime/feats/<name>/<set> and caches per-model results in runtime/eval/models/<name>.{json,npz};
here both roots are redirected to method/phonojepa/runtime. Reference models (baselines) are copied read-only from
method/obs_jepa/runtime/eval/models so that paired signer-cluster statistics can be computed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pj.paths import OBS_JEPA, RUNTIME  # noqa: E402

sys.path.insert(0, str(OBS_JEPA))
import oj.evaluate as E  # noqa: E402

E.RUNTIME = RUNTIME
E.EVAL_DIR = RUNTIME / "eval"


def import_refs(refs):
    src = OBS_JEPA / "runtime" / "eval" / "models"
    dst = E.EVAL_DIR / "models"; dst.mkdir(parents=True, exist_ok=True)
    for r in refs:
        for ext in ("json", "npz"):
            if not (dst / f"{r}.{ext}").exists() and (src / f"{r}.{ext}").exists():
                shutil.copy2(src / f"{r}.{ext}", dst / f"{r}.{ext}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--refs", default="SR3D,JH3D,JV3D,B13", help="baselines from obs_jepa for paired statistics")
    ap.add_argument("--protocol", default="both", choices=["loso", "aslc300", "both"])
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    refs = [r for r in a.refs.split(",") if r]
    import_refs(refs)
    names = [n for n in a.names.split(",") if n]
    res = {"models": {}, "refs": {}, "paired": {}}
    for n in names:
        res["models"][n] = E.evaluate_model(n, a.protocol, not a.no_probes, a.force)
    for r in refs:
        js = E.EVAL_DIR / "models" / f"{r}.json"
        if js.exists():
            res["refs"][r] = json.load(open(js))
    for r in refs:
        for n in names:
            if (E.EVAL_DIR / "models" / f"{r}.npz").exists():
                res["paired"][f"{n}-{r}"] = E.paired(n, r)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    for n in names + refs:
        d = res["models"].get(n) or res["refs"].get(n) or {}
        l, s = d.get("loso", {}), d.get("aslc300", {})
        print(f"{n:28s} LOSO w512 R@1 {l.get('white512', {}).get('R@1', float('nan')):.3f} | ASLC300 retW "
              f"{s.get('retrieval_white512', {}).get('R@1', float('nan')):.3f} LP {s.get('linear_probe', {}).get('macro_top1', float('nan')):.3f}"
              f" | signer {l.get('signer_probe_acc', float('nan'))}")


if __name__ == "__main__":
    main()
