"""Frozen dictionary retrieval: max exemplar cosine per gloss, pessimistic ties."""
from pathlib import Path
import numpy as np
import torch

from ..cache.hashing import file_hash
from ..cli import write_json
from ..inference.extract import temporal_embedding


def dictionary_metrics(queries, gallery, query_labels, gallery_labels, query_sources, gallery_sources, *, chunk_size=128):
    q, g = np.array(queries, dtype=np.float64, copy=True), np.array(gallery, dtype=np.float64, copy=True)
    ql, gl = np.asarray(query_labels), np.asarray(gallery_labels)
    if q.ndim != 2 or g.ndim != 2 or not len(q) or not len(g) or q.shape[1] != g.shape[1] or chunk_size < 1:
        raise ValueError("Dictionary retrieval needs nonempty compatible embedding matrices")
    if ql.shape != (len(q),) or gl.shape != (len(g),) or len(query_sources) != len(q) or len(gallery_sources) != len(g):
        raise ValueError("Dictionary retrieval annotations do not align")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Dictionary retrieval embeddings must be finite")
    if set(query_sources) & set(gallery_sources):
        raise ValueError("Dictionary query/gallery sources must be disjoint")
    if len(set(query_sources)) != len(q) or len(set(gallery_sources)) != len(g):
        raise ValueError("Dictionary retrieval requires one clip per unique source")
    classes = np.unique(gl)
    if not set(ql.tolist()) <= set(classes.tolist()):
        raise ValueError("Query class absent from training dictionary; no silent query exclusion")
    qn, gn = np.linalg.norm(q, axis=1, keepdims=True), np.linalg.norm(g, axis=1, keepdims=True)
    q /= np.maximum(qn, 1e-12)
    g /= np.maximum(gn, 1e-12)
    order = np.argsort(gl, kind="stable")
    sorted_labels = gl[order]
    starts = np.r_[0, np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1]
    ranks = []
    for start in range(0, len(q), chunk_size):
        stop = min(len(q), start + chunk_size)
        cosine = q[start:stop] @ g.T
        scores = np.maximum.reduceat(cosine[:, order], starts, axis=1)
        true = scores[np.arange(stop - start), np.searchsorted(classes, ql[start:stop])]
        ranks.extend((scores >= true[:, None]).sum(axis=1).tolist())
    rank = np.asarray(ranks)
    return {"DCG": float(np.mean(1 / np.log2(1 + rank))), "Rec@1": float(np.mean(rank <= 1)),
        "Rec@5": float(np.mean(rank <= 5)), "MRR": float(np.mean(1 / rank)), "ranks": ranks,
        "query_count": len(q), "gallery_count": len(g), "gloss_count": len(classes),
        "zero_query_embeddings": int((qn == 0).sum()), "zero_gallery_embeddings": int((gn == 0).sum()),
        "ranking": "gloss_max_exemplar_cosine_float64", "tie_policy": "pessimistic_other_gloss_score_greater_or_equal",
        "pooling": "timestamp-aligned tiled dense mean, distinct source frames, L2 once per clip",
        "metric_scale": "0..1; multiply by100 only for presentation"}


@torch.no_grad()
def evaluate_dictionary(model, train, queries, config, directory, *, device="cpu", split="val"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    matrices = []
    try:
        for role, corpus in (("gallery", train), ("query", queries)):
            values = []
            for index, name in enumerate(corpus.available):
                from ..training.interruptions import raise_if_requested
                raise_if_requested()
                output = temporal_embedding(model, corpus.load(name), config,
                    window_samples=config["downstream"]["window_samples"], overlap_samples=config["downstream"]["overlap_samples"], device=device)
                values.append(output.clip.cpu().numpy())
                if (index + 1) % 100 == 0:
                    print(f"Retrieval {split} {role}: {index + 1}/{len(corpus.available)}", flush=True)
            matrix = np.stack(values)
            np.savez_compressed(directory / f"{role}_embeddings.npz", embeddings=matrix, sample_ids=np.array(corpus.available))
            matrices.append(matrix)
    finally:
        model.train(was_training)
    records = [[corpus.records[n] for n in corpus.available] for corpus in (train, queries)]
    gallery_records, query_records = records
    metrics = dictionary_metrics(matrices[1], matrices[0], [r.label_id for r in query_records], [r.label_id for r in gallery_records],
        [r.source_group_id for r in query_records], [r.source_group_id for r in gallery_records])
    report = {"status": "completed", "split": split, "metrics": metrics, "classifier_trained": False,
        "train_manifest_sha256": file_hash(train.manifest), "query_manifest_sha256": file_hash(queries.manifest),
        "query_signers": len({r.signer_id for r in query_records}), "gallery_signers": len({r.signer_id for r in gallery_records}),
        "activity_weighting": False, "paper_recipe_note": "SignRep-inspired dictionary metrics; SignPathLab pooling/inputs differ; not exact SignRep reproduction"}
    write_json(directory / "metrics.json", report)
    with (directory / "predictions.jsonl").open("w") as stream:
        import json
        for record, rank in zip(query_records, metrics["ranks"]):
            stream.write(json.dumps({"sample_id": record.sample_id, "source_group_id": record.source_group_id,
                "reference_label_id": record.label_id, "split": split, "rank": rank, "DCG": 1 / np.log2(1 + rank),
                "Rec@1": int(rank <= 1), "Rec@5": int(rank <= 5)}) + "\n")
    return report
