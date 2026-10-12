from ..cache.hashing import file_hash
from ..cache.index import CachedCorpus
from ..data.dataset import fitting_geometry, geometry_kwargs
from .kmeans import KMeansTargets
from .projection import ProjectionObservation, TargetProjectors


def target_paths(resolved):
    p = resolved.values["geometry"]["projection"]
    return resolved.path(p["statistics_path"]), resolved.path(p["matrices_path"])


def cluster_path(resolved):
    # Codebooks belong to the common train corpus, not to an experiment's temporal seed.
    return resolved.path(resolved.values["cache"]["root"]).parent / "targets" / "kmeans"


def fit_targets(resolved, kind, split="train"):
    if split != "train":
        raise ValueError("Target fitting is restricted to the train split")
    c = resolved.values
    manifest = resolved.path(c["data"]["train_manifest"])
    corpus = CachedCorpus(manifest, resolved.path(c["cache"]["root"]), split="train", config=c)
    if kind == "kmeans":
        model = KMeansTargets.fit(((corpus.records[name], corpus.load(name)) for name in corpus.available),
            manifest_sha256=file_hash(manifest), extractor_hash=corpus.identity["extractor_hash"], seed=17)
        destination = cluster_path(resolved)
        model.save(destination)
        return {"status": "completed", "kind": kind, "output": str(destination), "counts": model.metadata["sample_counts"]}
    if kind != "geometry":
        raise ValueError(f"Unsupported target kind: {kind}")
    observations = []
    for name in corpus.available:
        view = corpus.records[name].pretrain_view()
        observations.extend(ProjectionObservation(view.source_group_id, "train", f"{g.group}/{g.scale}", g.descriptor)
                            for g in fitting_geometry(corpus.load(name), c) if g.valid)
    p = c["geometry"]["projection"]
    model = TargetProjectors.fit(observations, allowed_train_sources=set(corpus.pretrain_sources()),
        manifest_sha256=file_hash(manifest), extractor_hash=corpus.identity["extractor_hash"], descriptor_config=geometry_kwargs(c),
        output_dim=p["output_dim"], seed=p["seed"])
    stats, matrices = target_paths(resolved)
    model.save_files(stats, matrices)
    return {"status": "completed", "kind": kind, "statistics": str(stats), "matrices": str(matrices),
            "transform_hash": model.fingerprint(), "counts": model.metadata["counts"]}


def load_targets(resolved, corpus):
    c, geo, cluster = resolved.values, None, None
    identity = {"manifest_sha256": file_hash(corpus.manifest), "extractor_hash": corpus.identity["extractor_hash"]}
    if c["loss"]["geometry_weight"]:
        stats, matrices = target_paths(resolved)
        geo = TargetProjectors.load_files(stats, matrices, expected_manifest_sha256=identity["manifest_sha256"],
            expected_extractor_hash=identity["extractor_hash"], expected_descriptor_config=geometry_kwargs(c))
    if c["loss"]["cluster_weight"]:
        cluster = KMeansTargets.load(cluster_path(resolved), **identity)
    return geo, cluster
