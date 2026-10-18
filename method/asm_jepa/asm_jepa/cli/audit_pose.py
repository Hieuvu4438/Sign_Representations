"""Pose cache audit (§6.4): coverage per part, score distributions, NaN/inf, timestamp gaps, bone-length outliers,
tracking flags, missing caches, LH/RH anatomical consistency, optional overlays for visual inspection.

The sample is stratified by signer and duration (not by confidence). Overlays still have to be *looked at*;
this tool only renders them.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from asm_jepa.cli.common import run
from asm_jepa.data.manifest import SPLITS, load_split
from asm_jepa.data.pose_store import read_clip
from asm_jepa.data.schema import FACE, LH, RH
from asm_jepa.utils import now, read_json, resolve, write_json

PARTS = {"shoulders": [5, 6], "elbows": [7, 8], "lh": LH, "rh": RH, "face": FACE}
COLORS = {"body": (0, 255, 255), "lh": (0, 0, 255), "rh": (255, 0, 0), "face": (0, 255, 0)}


def stratified_sample(rows, n, seed):
    rng = np.random.default_rng(seed)
    by_signer = defaultdict(list)
    for r in rows:
        by_signer[r["signer_id"]].append(r)
    per = max(1, int(np.ceil(n / max(len(by_signer), 1))))
    out = []
    for s in sorted(by_signer):
        rs = sorted(by_signer[s], key=lambda r: (r.get("duration_sec") or 0.0, r["sample_id"]))
        pos = np.linspace(0, len(rs) - 1, min(per, len(rs))).round().astype(int)
        jitter = rng.integers(-2, 3, size=len(pos))
        out += [rs[int(np.clip(p + j, 0, len(rs) - 1))] for p, j in zip(pos, jitter)]
    rng.shuffle(out)
    return out[:n]


def render_overlay(video: Path, clip, frame_idx: int, out: Path):
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(clip.source_frame_index[frame_idx]))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return False
    xy, ob = clip.xy[frame_idx], clip.observed[frame_idx]
    groups = {"body": [5, 6, 7, 8, 9, 10], "lh": LH, "rh": RH, "face": FACE}
    for g, idx in groups.items():
        for j in idx:
            if ob[j]:
                cv2.circle(img, (int(xy[j, 0]), int(xy[j, 1])), 2 if g == "face" else 3, COLORS[g], -1)
    for j, label in ((91, "LH"), (112, "RH"), (9, "Lw"), (10, "Rw")):
        if ob[j]:
            cv2.putText(img, label, (int(xy[j, 0]) + 4, int(xy[j, 1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
    cv2.putText(img, f"frame {frame_idx}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out), img))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", default="data/processed/aslc")
    ap.add_argument("--split", default="train", choices=SPLITS)
    ap.add_argument("--sample-count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-overlays", action="store_true")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    root = resolve(args.processed_root)
    rows = load_split(root, args.split)
    out_dir = resolve(args.output_dir) if args.output_dir else root / "audit" / args.split
    raw_root = Path(read_json(root / "dataset_version.json")["raw_root"])
    missing_all = [r["sample_id"] for r in rows if not (root / r["pose_relpath"]).is_file()]
    sample = stratified_sample([r for r in rows if (root / r["pose_relpath"]).is_file()], args.sample_count, args.seed)
    cov = defaultdict(list)
    scores = defaultdict(list)
    issues = defaultdict(list)
    lr_ok, lr_n = 0, 0
    forearm_ratio, hand_ratio = [], []
    alias_frames, both_hand_frames = 0, 0
    overlays = []
    for r in sample:
        c = read_clip(root / r["pose_relpath"])
        fv = c.frame_valid
        if (~np.isfinite(c.xy[c.observed])).any():
            issues["nan_in_observed"].append(r["sample_id"])
        for part, idx in PARTS.items():
            cov[part].append(float(c.observed[fv][:, idx].mean()) if fv.any() else 0.0)
            s = c.score[fv][:, idx]
            if s.size:
                scores[part].append(np.percentile(s, [5, 50, 95]))
        dt = np.diff(c.timestamps_sec)
        if len(dt) and (dt.max() > 2.5 * np.median(dt)):
            issues["timestamp_gap"].append(r["sample_id"])
        if c.meta.get("frame_count_mismatch"):
            issues["frame_count_mismatch"].append(r["sample_id"])
        if (~fv).mean() > 0.2:
            issues["many_invalid_frames"].append(r["sample_id"])
        both = c.observed[:, 91] & c.observed[:, 9] & c.observed[:, 10]
        if both.any():
            d_same = np.linalg.norm(c.xy[both, 91] - c.xy[both, 9], axis=-1)
            d_other = np.linalg.norm(c.xy[both, 91] - c.xy[both, 10], axis=-1)
            lr_ok += int((d_same < d_other).sum())
            lr_n += int(both.sum())
        sh = c.observed[:, 5] & c.observed[:, 6]
        if sh.any():
            sw = np.median(np.linalg.norm(c.xy[sh, 5] - c.xy[sh, 6], axis=-1))
            for e, w in ((7, 9), (8, 10)):
                ok = c.observed[:, e] & c.observed[:, w]
                if ok.any():
                    forearm_ratio += list(np.linalg.norm(c.xy[ok, e] - c.xy[ok, w], axis=-1) / sw)
            both_h = c.observed[:, LH].all(1) & c.observed[:, RH].all(1)
            if both_h.any():
                d = np.linalg.norm(c.xy[both_h][:, LH] - c.xy[both_h][:, RH], axis=-1).mean(1) / sw
                alias_frames += int((d < 0.15).sum())
                both_hand_frames += int(both_h.sum())
            for base in (91, 112):
                ok = c.observed[:, base] & c.observed[:, base + 9]
                if ok.any():
                    hand_ratio += list(np.linalg.norm(c.xy[ok, base] - c.xy[ok, base + 9], axis=-1) / sw)
        if args.render_overlays:
            T = c.num_frames
            for q in (0.25, 0.5, 0.75):
                f = int(q * (T - 1))
                path = out_dir / "overlays" / f"{r['sample_id']}_{f:04d}.png"
                if render_overlay(raw_root / r["video_relpath"], c, f, path):
                    overlays.append(str(path))

    def outliers(x):
        x = np.asarray(x)
        if len(x) < 10:
            return {}
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        return {"median": float(med), "p01": float(np.percentile(x, 1)), "p99": float(np.percentile(x, 99)),
                "outlier_fraction_robust_z_gt_5": float(np.mean(np.abs(x - med) / (1.4826 * mad) > 5))}

    report = {"time": now(), "split": args.split, "n_rows": len(rows), "n_missing_cache": len(missing_all),
              "missing_cache_ids": missing_all[:200], "sample_count": len(sample),
              "sample_signers": sorted({r["signer_id"] for r in sample}),
              "coverage_mean": {k: float(np.mean(v)) for k, v in cov.items()},
              "coverage_p10": {k: float(np.percentile(v, 10)) for k, v in cov.items()},
              "score_percentiles_p5_p50_p95_mean": {k: np.mean(v, 0).round(3).tolist() for k, v in scores.items()},
              "lh_closer_to_body_left_wrist_fraction": lr_ok / lr_n if lr_n else None,
              "hand_alias_fraction": alias_frames / both_hand_frames if both_hand_frames else None,
              "hand_alias_rule": "both hands observed and mean LH-RH keypoint distance < 0.15 shoulder widths",
              "forearm_over_shoulder": outliers(forearm_ratio), "wrist_mcp_over_shoulder": outliers(hand_ratio),
              "issues": {k: {"count": len(v), "ids": v[:50]} for k, v in issues.items()},
              "overlays": overlays, "overlays_reviewed": False,
              "note": "overlays must be inspected by a person/agent; LH/RH are checked against COCO body wrists"}
    write_json(out_dir / "audit_report.json", report)
    print({k: report[k] for k in ("n_rows", "n_missing_cache", "coverage_mean", "lh_closer_to_body_left_wrist_fraction", "hand_alias_fraction")})
    print(f"report: {out_dir / 'audit_report.json'}  overlays: {len(overlays)}")
    return 0


if __name__ == "__main__":
    run(main)
