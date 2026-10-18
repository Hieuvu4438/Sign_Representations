"""Markdown summary of the 1-seed screening (decision rule of docs/proposal11/BAO_CAO_CHIEN_LUOC_SO_SANH_SIGNREP.md §4.3,
screening version: one seed, so 'passes screening' never means 'better than SignRep' yet)."""
from __future__ import annotations

import argparse
import json

ROLE = {"SR3D": "SignRep-repro (Hiera-B, Kinetics-MAE)", "SRVJ": "SignRep recipe on V-JEPA 2.1", "ASLC002": "SignRep-repro (Codex, batch 4)",
        "JV3D": "JEPA + SignRep priors, V-JEPA init", "JH3D": "JEPA + SignRep priors, Hiera init", "A00": "frozen V-JEPA 2.1 (no training)"}


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


def ci(d):
    if not d:
        return "—"
    return f"{100 * d['diff']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}] ({d.get('pos', '')})"


def get(m, *ks):
    for k in ks:
        if not isinstance(m, dict) or k not in m:
            return None
        m = m[k]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True); ap.add_argument("--state", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = json.load(open(a.final)); S = json.load(open(a.state)); M, P = F["models"], F["paired"]
    L = ["# Kết quả screening 1 seed: proposal11 so với SignRep tái dựng", "",
         "Dữ liệu 3.000 video ASLC002, 4.000 bước × 32 view, chỉ validation. **1 seed: chỉ là sàng lọc.**", "",
         "Quyết định tự động của lịch chạy: " + json.dumps(S.get("decisions", {}), ensure_ascii=False), "",
         "## Chỉ số chính", "",
         "| Model | Vai trò | ASLC300 linear probe macro Top-1 | LOSO-val R@1 (whitened) | ASLC300 retrieval R@1 | signer probe | mean cos |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, m in M.items():
        L.append(f"| {n} | {ROLE.get(n, '')} | {pct(get(m, 'aslc300', 'linear_probe', 'macro_top1'))} | {pct(get(m, 'loso', 'white512', 'R@1'))} | "
                 f"{pct(get(m, 'aslc300', 'retrieval_white512', 'R@1'))} | {pct(get(m, 'loso', 'signer_probe_acc'))} | "
                 f"{get(m, 'loso', 'geometry', 'mean_cos') or 0:.3f} |")
    L += ["", "## Probe theo bộ phận cơ thể (R², LOSO-val)", "", "| Model | handshape | body | location/relations | non-manual | movement |", "| --- | --- | --- | --- | --- | --- |"]
    for n, m in M.items():
        p = get(m, "loso", "probe_R2") or {}
        L.append(f"| {n} | " + " | ".join(f"{p.get(k, float('nan')):.3f}" for k in ("handshape", "body", "location_relations", "non_manual", "movement")) + " |")
    L += ["", "## So sánh ghép cặp (bootstrap theo cụm người ký; điểm %)", "",
          "| So sánh | Linear probe macro | LOSO R@1 | Retrieval ASLC300 R@1 | Qua sàng lọc? |", "| --- | --- | --- | --- | --- |"]
    verdict = {}
    for k, d in P.items():
        lp, lo = d.get("aslc300_linear_probe_macro"), d.get("loso_R@1_white512")
        ok = bool(lp and lo and lp["diff"] >= 0.01 and lo["diff"] >= 0.01 and lp["ci95"][0] > 0 and lo["ci95"][0] > 0)
        verdict[k] = ok
        L.append(f"| {k} | {ci(lp)} | {ci(lo)} | {ci(d.get('aslc300_retrieval_R@1'))} | {'có' if ok else 'không'} |")
    win = sorted({k.split('-')[0] for k, v in verdict.items() if v and k.endswith('-SR3D') and verdict.get(k.replace('-SR3D', '-SRVJ'))})
    L += ["", "## Kết luận sàng lọc", "",
          "Qua sàng lọc so với **cả SR3D và SR-VJ** (≥ +1 điểm ở cả hai chỉ số chính, CI > 0): " + (", ".join(win) if win else "**không có model nào**"), "",
          "Một model qua sàng lọc ở đây mới chỉ đủ điều kiện chạy thêm 2 seed và kiểm tra trên WLASL (mục 4.3 của báo cáo chiến lược); "
          "chưa được tuyên bố là tốt hơn SignRep."]
    open(a.out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
