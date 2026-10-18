"""Compare completed EX16/proposal11 models and public SignRep on ASLC300.

CPU only; reads existing feature caches without changing scheduler artifacts.
Pooling: arithmetic mean of RAW window features, then one clip-level L2.
Public SignRep stride-2 windows are selected at starts divisible by 8.
No whitening, classifier training, activity weighting, or test data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNTIME = REPO / "method/obs_jepa/runtime"
MODELS = ["SR3D", "SRVJ", "ASLC002", "JV3D", "JH3D", "A00",
          "A10_AR", "A10_AD", "A11_AD", "A12_AD", "A13_AD", "A10c_AD",
          "B10", "B11", "B12", "B13"]


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite clip embedding")
    # Preserve zero embeddings rather than silently dropping gallery/query clips.
    # Cosines are zero; ties at the true-class score count against the query.
    return x / np.maximum(norms, 1e-12)


def pool_raw(windows, offsets, indices=None, starts=None):
    indices = range(len(offsets) - 1) if indices is None else indices
    out, counts = [], []
    for i in indices:
        a, b = int(offsets[i]), int(offsets[i + 1])
        x = windows[a:b]
        if starts is not None:
            x = x[starts[a:b] % 8 == 0]
        if len(x) == 0:
            raise ValueError(f"No feature windows for clip {i}")
        out.append(x.astype(np.float32).mean(axis=0))
        counts.append(len(x))
    return normalize(np.stack(out)), np.asarray(counts)


def rank_and_metrics(features, rows):
    gallery = np.array([r["role"] == "gallery" for r in rows])
    query = ~gallery
    labels = np.array([r["label"] for r in rows], dtype=int)
    original_gallery, original_query = labels[gallery], labels[query]
    sim = features[query] @ features[gallery].T
    classes = np.unique(original_gallery)
    assert len(classes) == 300 and set(original_query) <= set(classes)
    yg, yq = np.searchsorted(classes, original_gallery), np.searchsorted(classes, original_query)
    scores = np.stack([sim[:, yg == c].max(axis=1) for c in range(len(classes))], axis=1)
    true = scores[np.arange(len(yq)), yq]
    ranks = (scores >= true[:, None]).sum(axis=1)
    # Independent rank computation by explicitly sorting the 300 class scores.
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_ranks = np.argmax(order == yq[:, None], axis=1) + 1
    ties = (scores == true[:, None]).sum(axis=1) > 1
    assert np.array_equal(ranks[~ties], sorted_ranks[~ties])
    nearest_correct = yg[np.argmax(sim, axis=1)] == yq
    assert np.array_equal((ranks <= 1)[~ties], nearest_correct[~ties])
    metrics = {"DCG": float(np.mean(1 / np.log2(1 + ranks))),
               "Rec@1": float(np.mean(ranks <= 1)),
               "Rec@5": float(np.mean(ranks <= 5)),
               "queries": len(yq), "true_class_tied_queries": int(ties.sum())}
    return metrics, ranks


def training_info(name):
    if name == "SignRep_public":
        return "YouTube-SL-25 (tác giả); không train thêm trên ASLC002", "Hiera-B MAE-Kinetics", "RGB"
    if name == "A00":
        return "Không train thêm trên ASLC002; dùng V-JEPA 2.1 public", "V-JEPA 2.1 ViT-B distilled từ ViT-G", "RGB"
    init = "Ngẫu nhiên" if name.startswith("B") else (
        "Hiera-B MAE-Kinetics" if name in {"SR3D", "JH3D", "ASLC002"}
        else "V-JEPA 2.1 ViT-B distilled từ ViT-G")
    return "ASLC002: 3.000 video ASL Citizen train, không nhãn gloss", init, "Landmark 2D" if name.startswith("B") else "RGB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RUNTIME / "eval/comparison_avg_aslc300")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    canonical_path = RUNTIME / "feats/JH3D/aslc300/clips.jsonl"
    rows = read_rows(canonical_path)
    gallery = [r for r in rows if r["role"] == "gallery"]
    query = [r for r in rows if r["role"] == "query"]
    assert len(gallery) == 3000 and len(query) == 1500
    assert rows == gallery + query
    assert len({r["gloss"] for r in rows}) == 300
    assert not ({r["sample_id"] for r in gallery} & {r["sample_id"] for r in query})
    assert not ({r["signer"] for r in gallery} & {r["signer"] for r in query})
    output = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "protocol": {"dataset": "ASLC300 subset of official ASL Citizen splits",
                           "gallery": "3,000 train videos; 300 glosses x 10 videos",
                           "query": "1,500 validation videos; 300 glosses x 5 videos; 5 signers",
                           "gallery_signers": len({r["signer"] for r in gallery}),
                           "query_signers": len({r["signer"] for r in query}),
                           "signer_disjoint": True, "test_used": False,
                           "pooling": "mean of raw window features, then clip-level L2",
                           "similarity": "cosine; class score = max over gallery videos of that class",
                           "rank": "one plus number of other classes with score >= true-class score",
                           "DCG": "mean(1/log2(1+true_class_rank))",
                           "Rec@K": "mean(true_class_rank <= K)",
                           "whitening": False, "activity_weighting": False,
                           "manifest_sha256": digest(canonical_path)}, "models": {}}
    ranks_out = {"query_ids": np.array([r["sample_id"] for r in query]),
                 "query_labels": np.array([r["label"] for r in query])}
    for name in MODELS:
        root = RUNTIME / "feats" / name / "aslc300"
        assert read_rows(root / "clips.jsonl") == rows, name
        features, counts = pool_raw(np.load(root / "windows.npy", mmap_mode="r"), np.load(root / "offsets.npy"))
        metrics, ranks = rank_and_metrics(features, rows)
        train, init, modality = training_info(name)
        output["models"][name] = {"metrics": metrics, "training_dataset": train,
                                   "initialization": init, "input": modality,
                                   "source": str(root.relative_to(REPO)),
                                   "feature_meta": json.loads((root / "meta.json").read_text()),
                                   "window_count": int(counts.sum()),
                                   "zero_feature_clips": [{"sample_id": r["sample_id"], "role": r["role"]}
                                                          for r, f in zip(rows, features) if not f.any()]}
        ranks_out[name] = ranks
        print(name, json.dumps(metrics), flush=True)
    public_parts, public_counts, source_meta = [], [], {}
    for split, role in (("train", "gallery"), ("val", "query")):
        root = REPO / "method/p4_signssl/runtime/feats" / f"aslc_{split}_s2"
        cached_rows = read_rows(root / "clips.jsonl")
        lookup = {r["sample_id"]: i for i, r in enumerate(cached_rows)}
        assert len(lookup) == len(cached_rows)
        chosen = [r for r in rows if r["role"] == role]
        indices = []
        for r in chosen:
            i = lookup[r["sample_id"]]
            assert r["gloss"] == cached_rows[i]["gloss"]
            assert Path(r["video"]).resolve() == Path(cached_rows[i]["path"]).resolve()
            assert r["split"] == cached_rows[i]["split"]
            indices.append(i)
        feats, counts = pool_raw(np.load(root / "windows_features.npy", mmap_mode="r"),
                                 np.load(root / "offsets.npy"), indices,
                                 np.load(root / "window_start.npy", mmap_mode="r"))
        public_parts.append(feats); public_counts.append(counts)
        source_meta[split] = json.loads((root / "meta.json").read_text())
    assert source_meta["train"]["sha256"] == source_meta["val"]["sha256"]
    features = np.concatenate(public_parts)
    metrics, ranks = rank_and_metrics(features, rows)
    train, init, modality = training_info("SignRep_public")
    output["models"]["SignRep_public"] = {"metrics": metrics, "training_dataset": train,
                                               "initialization": init, "input": modality,
                                               "feature_meta": source_meta,
                                               "selected_stride": 8,
                                               "window_count": int(sum(c.sum() for c in public_counts)),
                                               "zero_feature_clips": [{"sample_id": r["sample_id"], "role": r["role"]}
                                                                      for r, f in zip(rows, features) if not f.any()]}
    ranks_out["SignRep_public"] = ranks
    print("SignRep_public", json.dumps(metrics), flush=True)
    np.savez(a.out / "per_query_ranks.npz", **ranks_out)
    (a.out / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    sorted_models = sorted(output["models"], key=lambda n: output["models"][n]["metrics"]["Rec@1"], reverse=True)
    with (a.out / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "DCG_x100", "Rec@1_pct", "Rec@5_pct", "training_dataset", "initialization", "input"])
        for n in sorted_models:
            m = output["models"][n]; v = m["metrics"]
            writer.writerow([n, *[100 * v[k] for k in ["DCG", "Rec@1", "Rec@5"]], m["training_dataset"], m["initialization"], m["input"]])
    lines = ["# So sánh dictionary retrieval ASLC300: SignRep(avg)", "",
             f"Thời điểm: {output['generated_at']}", "",
             "Gallery: 3.000 video official train, 300 gloss × 10. Query: 1.500 video official validation, 300 gloss × 5, 5 người ký không có trong gallery. Không dùng test.", "",
             "Pooling: trung bình feature window gốc → L2 ở cấp clip → cosine; không PCA whitening, không activity weighting. Class score là cosine cao nhất trong các video gallery của lớp. DCG = mean(1/log2(1+r)); Rec@K = mean(r≤K). Mọi cột dưới đây nhân 100.", "",
             "RGB: window 16 frame, stride 8; các nhánh A/A00 dùng letterbox và effective fps ≤25 theo preprocessing của chúng, Hiera/JV3D/SR-VJ/public dùng native-frame squash 224. B: landmark 2D, window 64 frame, stride 32, effective fps ≤25. Cache public stride 2 được chọn lại tại start % 8 = 0.", "",
             "Lưu ý: retrieval_mean cũ chuẩn hóa L2 từng window trước khi pooling. Bảng này tính lại từ feature gốc, chỉ L2 sau pooling; vì vậy số liệu có thể khác các báo cáo trước.", "",
             "| Method | Input | DCG | Rec@1 | Rec@5 | Dữ liệu train ký hiệu | Khởi tạo |",
             "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for n in sorted_models:
        m = output["models"][n]; v = m["metrics"]
        lines.append(f"| {n} | {m['input']} | {100*v['DCG']:.2f} | {100*v['Rec@1']:.2f} | {100*v['Rec@5']:.2f} | {m['training_dataset']} | {m['initialization']} |")
    lines += ["", "Tất cả nhánh tự train dùng checkpoint cuối, một seed. ASLC002 là lượt 5.000 bước, batch 4 view; EX16/SR-VJ là 4.000 bước, 32 view; A/B là 4.000 bước, batch 32. A00 không train thêm. Public SignRep là checkpoint tác giả, SHA256 " + source_meta["train"]["sha256"] + ".", "",
              "Kiểm tra: toàn bộ 16 manifest tự train/foundation trùng thứ tự; public phủ đủ sample ID, gloss, path và split; gallery/query không trùng video hoặc người ký; xếp hạng đối chiếu bằng sort độc lập và nearest-video Top-1. Tie true-class được xử lý bất lợi cho query và ghi số lượng trong JSON.", "",
              "P3 bị bỏ qua do B gate không đạt; P4 chưa có kết quả ở thời điểm tạo bảng. Không bổ sung các thí nghiệm cũ ngoài EX16/proposal11 vào bảng này.", "",
              "Nguồn phương pháp: SignRep §6, §7.1–7.4, https://arxiv.org/html/2503.08529v1 ; public checkpoint https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1 .", ""]
    (a.out / "comparison.md").write_text("\n".join(lines))
    print("SAVED", a.out, flush=True)


if __name__ == "__main__":
    main()
