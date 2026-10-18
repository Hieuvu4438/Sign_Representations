"""Summary table of the 5k-step diagnostic queue (val only)."""
import json
from pathlib import Path

D = Path("runs/diagnostics")
rows = []
for r in sorted(D.glob("*/metrics.jsonl")):
    v = r.parent.name
    m = [json.loads(l) for l in r.read_text().splitlines() if l.strip()]
    val = [x for x in m if x.get("kind") == "frozen_val"]
    tr = [x for x in m if "loss/total" in x]
    ro = D / f"readout_q_{v}.json"
    ro = json.loads(ro.read_text()) if ro.is_file() else {}
    pct = lambda d, k="r1": f"{100 * d[k]:.2f}" if d else "–"
    rows.append((v, " / ".join(f"{100 * x['val/frozen_gloss_max/r1']:.2f}" for x in val),
                 pct(ro.get("concat1024 (current readout)")), pct(ro.get("hands only (lh+rh)")),
                 pct(ro.get("hands only, whitened")), f"{ro.get('hands_val_nn_same_signer', float('nan')):.2f}",
                 f"{tr[-1]['diag/effective_rank']:.1f}" if tr else "–", f"{tr[-1].get('teacher_entropy', float('nan')):.3f}" if tr else "–"))
print("# Diagnostic queue (5k steps, val R@1 %, gallery train → query val)\n")
print("| variant | val R@1 at 1k/2k/3k/4k/5k (frozen concat) | last: concat | last: hands | last: hands whitened | hands NN same signer | eff. rank | teacher entropy |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print("| " + " | ".join(r) + " |")
ref = Path("runs/diagnostics/raw_baseline.json")
if ref.is_file():
    print("\nBaseline không học (pose thô, whitened): " + ", ".join(f"{k}: {100 * v['r1']:.2f}" for k, v in json.loads(ref.read_text()).items() if "whitened" in k))
