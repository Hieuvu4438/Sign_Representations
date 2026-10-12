"""Collect measured frozen retrieval and clearly separate published references."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "runtime/eval/PJ_PLctx_FULL_160ep_dictionary"
PAPER = "https://arxiv.org/html/2503.08529v1#S7.SS4"


def main():
    records = []
    for dataset, title, n_gallery, n_query in (("aslc", "ASL Citizen", 40154, 32941),
                                               ("wlasl", "WLASL2000", 14296, 2879)):
        result = json.loads((OUT / dataset / "results.json").read_text())
        for readout, metrics in result["readouts"].items():
            records.append(dict(dataset=title, method="PLctx_FULL_160ep" + readout,
                source="measured", pooling="avg", primary=readout == "_grp",
                pretraining="ASL Citizen train, 40154 clips, 160 epochs, 1 seed",
                gallery=n_gallery, queries=n_query, window=64, stride=2,
                DCG=100 * metrics["DCG"], Rec1=100 * metrics["Rec1"], Rec5=100 * metrics["Rec5"]))
        if dataset == "wlasl":
            folder = REPO / "method/obs_jepa/runtime/wlasl2000_frozen/results"
            for name in ("SignRep_public", "JH3D", "JV3D", "SR3D"):
                d = json.loads((folder / f"{name}.json").read_text())
                for pooling, metrics in d["variants"].items():
                    records.append(dict(dataset=title, method=name, source="existing local measurement",
                        pooling=pooling, primary=False,
                        pretraining="YouTube-SL-25" if name == "SignRep_public" else "ASL Citizen 3000-clip corpus",
                        gallery=n_gallery, queries=n_query, window=16, stride=2,
                        DCG=100 * metrics["DCG"], Rec1=100 * metrics["Rec1"], Rec5=100 * metrics["Rec5"]))
        else:
            path = REPO / "method/p4_signssl/runtime/eval/EX05_aslc_R0/metrics.json"
            metrics = json.loads(path.read_text())["results"]["features:activity"]
            assert metrics["n"] == n_query
            records.append(dict(dataset=title, method="SignRep_public", source="existing local measurement",
                pooling="weighted", primary=False, pretraining="YouTube-SL-25",
                gallery=n_gallery, queries=n_query, window=16, stride=2,
                DCG=100 * metrics["DCG"], Rec1=100 * metrics["R@1"], Rec5=100 * metrics["R@5"]))
        paper = {"aslc": {"avg": (61.40, 37.47, 68.77), "weighted": (71.21, 49.95, 80.09)},
                 "wlasl": {"avg": (53.57, 26.16, 60.98), "weighted": (57.93, 29.92, 67.41)}}
        for pooling, (dcg, rec1, rec5) in paper[dataset].items():
            records.append(dict(dataset=title, method="SignRep (paper Table 4)", source="published reference",
                pooling=pooling, primary=False, pretraining="YouTube-SL-25",
                gallery=n_gallery, queries=n_query, window=16, stride=2, DCG=dcg, Rec1=rec1, Rec5=rec5))
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    primary = [r for r in records if r["primary"]]
    lines = ["# PLctx full160: frozen dictionary retrieval", "", "## Material Passport", "",
        "Experiment: PLctx-full160-frozen-SignRep-retrieval. Frozen final checkpoint, official train galleries",
        "and test queries. No classifier, PCA, fine-tuning or WLASL pretraining. No external data upload.", "",
        "Primary readout `_grp` was fixed before test scoring. Secondary `_cls`/`_mean` results are retained",
        "in the CSV and JSON; they are not used to select a test-optimal configuration.", "",
        "| Dataset | Gallery train | Query test | DCG ×100 | Rec@1 % | Rec@5 % |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in primary:
        lines.append(f"| {r['dataset']} | {r['gallery']} | {r['queries']} | {r['DCG']:.2f} | {r['Rec1']:.2f} | {r['Rec5']:.2f} |")
    lines += ["", "## Protocol and scope", "",
        f"[SignRep §6, §7.2 and Table 4]({PAPER}): raw window mean -> video L2 -> cosine; class score is",
        "maximum gallery cosine; rank counts distinct classes with pessimistic ties. DCG is 1/log2(1+rank).", "",
        "PhonoJEPA uses its trained 64-frame skeleton encoder at min(fps,25), stride 2, native short-clip validity",
        "masks and uniform pooling. SignRep uses 16 RGB frames and can additionally use its trained activity head.",
        "These are declared model/input differences; this is the same frozen dictionary ranking protocol,",
        "not a claim of identical feature extraction. WLASL videos are the local pretrimmed release.", "",
        "PLctx pretraining uses all 40,154 ASL Citizen train clips, 160 epochs, one seed. This data budget differs",
        "from both the earlier 3,000-clip pilot and the SignRep public checkpoint's YouTube-SL-25 pretraining.", "",
        "Published Table 4 values and existing local SignRep/JH3D/JV3D/SR3D measurements are separate CSV rows.",
        "ASL Citizen test has 32,941 queries; earlier LOSO validation and ASLC300 are different evaluations.", "",
        "## Earlier pilot-compatible validation", ""]
    validation = json.loads((ROOT / "runtime/eval/PJ_PLctx_FULL_160ep.json").read_text())
    d = validation["models"]["PJ_PLctx_FULL_160ep_grp"]
    lines += [f"LOSO whitened Rec@1: {100*d['loso']['white512']['R@1']:.2f}%; "
              f"ASLC300 whitened Rec@1: {100*d['aslc300']['retrieval_white512']['R@1']:.2f}%; "
              f"linear probe macro Top-1: {100*d['aslc300']['linear_probe']['macro_top1']:.2f}%.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[12:17]))
    print("Report:", OUT / "REPORT.md")
    print("CSV:", OUT / "comparison.csv")


if __name__ == "__main__":
    main()
