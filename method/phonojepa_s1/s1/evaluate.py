"""Score PhonoJEPA features with the unchanged proposal11 evaluation (method/obs_jepa/oj/evaluate.py), so numbers are
directly comparable with SR3D / SRVJ / ASLC002 / JV3D / JH3D / A00 / obs_jepa A & B.

oj.evaluate reads runtime/feats/<name>/<set> and caches per-model results in runtime/eval/models/<name>.{json,npz};
here both roots are redirected to method/phonojepa_s1/runtime. Reference models (baselines) are copied read-only from
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
from s1.paths import OBS_JEPA, RUNTIME  # noqa: E402

sys.path.insert(0, str(OBS_JEPA))
import oj.evaluate as E  # noqa: E402

E.RUNTIME = RUNTIME
E.EVAL_DIR = RUNTIME / "eval"


def import_refs(refs):
    sources = [OBS_JEPA / "runtime" / "eval" / "models", ROOT.parent / "phonojepa/runtime/eval/models"]
    dst = E.EVAL_DIR / "models"; dst.mkdir(parents=True, exist_ok=True)
    for r in refs:
        for ext in ("json", "npz"):
            for src in sources:
                if not (dst / f"{r}.{ext}").exists() and (src / f"{r}.{ext}").exists():
                    shutil.copy2(src / f"{r}.{ext}", dst / f"{r}.{ext}")


def validate_features(name, protocol):
    """Reject partial/smoke features before producing comparable metric tables."""
    import numpy as np
    from oj.extract import rows_for
    sets = ['val','aslc300'] if protocol == 'both' else ['val' if protocol == 'loso' else 'aslc300']
    identity = None
    for setname in sets:
        d = RUNTIME/'feats'/name/setname
        meta = json.loads((d/'meta.json').read_text())
        if not meta.get('protocol_complete') or meta.get('training_config',{}).get('smoke'):
            raise ValueError('Smoke/partial features cannot be reported as full-protocol evaluation')
        if meta.get('group_projection_required'):
            raise ValueError('Run s1.project for matched group dimension before evaluating')
        signature = {k:meta.get(k) for k in ['checkpoint_sha256','extraction_source_sha256','training_config','weights']}
        if identity is not None and signature!=identity:raise ValueError('Mixed checkpoint/extraction provenance across evaluation sets')
        identity = signature
        actual = [json.loads(l) for l in (d/'clips.jsonl').read_text().splitlines()]
        expected = rows_for(setname)
        if actual != expected: raise ValueError(f'Changed evaluation rows/order: {setname}')
        win = np.load(d/'windows.npy',mmap_mode='r');off = np.load(d/'offsets.npy');spans = np.load(d/'src_span.npy')
        if len(off)!=len(actual)+1 or off[0]!=0 or off[-1]!=len(win) or np.any(np.diff(off)<=0):
            raise ValueError('Malformed window offsets')
        if spans.shape!=(len(win),2) or not np.isfinite(win).all():raise ValueError('Invalid windows/spans')
        if name.endswith('_grp') and win.shape[1]!=768:raise ValueError('Primary group readout must match baseline 768 dimensions')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--refs", default="PJ_PLctx_16k_grp,SR3D,JH3D,JV3D,B13", help="baselines from obs_jepa for paired statistics")
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
        validate_features(n, a.protocol)
        res["models"][n] = E.evaluate_model(n, a.protocol, not a.no_probes, a.force)
    for r in refs:
        js = E.EVAL_DIR / "models" / f"{r}.json"
        if js.exists():
            res["refs"][r] = json.load(open(js))
    for r in refs:
        for n in names:
            if (E.EVAL_DIR / "models" / f"{r}.npz").exists():
                res["paired"][f"{n}-{r}"] = E.paired(n, r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    for n in names + refs:
        d = res["models"].get(n) or res["refs"].get(n) or {}
        l, s = d.get("loso", {}), d.get("aslc300", {})
        print(f"{n:28s} LOSO w512 R@1 {l.get('white512', {}).get('R@1', float('nan')):.3f} | ASLC300 retW "
              f"{s.get('retrieval_white512', {}).get('R@1', float('nan')):.3f} LP {s.get('linear_probe', {}).get('macro_top1', float('nan')):.3f}"
              f" | signer {l.get('signer_probe_acc', float('nan'))}")


if __name__ == "__main__":
    main()
