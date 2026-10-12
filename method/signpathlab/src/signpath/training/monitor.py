import torch


@torch.no_grad()
def representation_diagnostics(clip_embeddings):
    x = clip_embeddings.float()
    if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
        return {"status": "insufficient_independent_clips", "count": len(x)}
    centered = x - x.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    power = singular.square()
    probability = power / power.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    normalized = torch.nn.functional.normalize(x, dim=-1)
    similarity = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
    return {"status": "measured", "clip_count": len(x), "feature_std_mean": float(x.std(dim=0).mean()),
            "effective_rank": float(entropy.exp()) if power.sum() > 0 else 0.0,
            "off_diagonal_cosine_mean": float(similarity[off_diagonal].mean()),
            "embedding_norm_mean": float(x.norm(dim=-1).mean()), "singular_values": singular.cpu().tolist()}


@torch.no_grad()
def training_diagnostics(model, corpus, config, directory, step, device, *, max_sources=64):
    """One embedding/source on a fixed train set; no adjacent-token pseudo-replication."""
    import numpy as np
    from pathlib import Path
    from ..cli import write_json
    from ..data.sampling import slice_features
    from ..evaluation.records import quality_summary
    from ..models.pooling import pool_raw_tokens
    from .runner import full_clip_batch
    was_training = model.training
    model.eval()
    sources = sorted(corpus.pretrain_sources().items())[:max_sources]
    online, teacher, identities, qualities = [], [], [], []
    typed = [[] for _ in range(7)]
    try:
        for source, names in sources:
            bundle = corpus.load(sorted(names)[0])
            qualities.append(quality_summary(bundle))
            segment = slice_features(bundle, 0, min(len(bundle.timestamps), config["data"]["segment_samples"]))
            batch = full_clip_batch([segment], config).to(device)
            output = model.encode_full(batch)
            online.append(pool_raw_tokens(output, batch).clip[0])
            if config["loss"]["semantic_weight"]:
                teacher.append(pool_raw_tokens(model.teacher(batch, batch.valid), batch).clip[0])
            raw = output.tokens.reshape(1, len(segment.timestamps), 7, -1)
            for stream in range(7):
                keep = batch.valid[0, :, stream]
                if keep.any():
                    typed[stream].append(raw[0, keep, stream].mean(0))
            identities.append({"source_group_id": source, "sample_id": sorted(names)[0]})
    finally:
        model.train(was_training)
    values = torch.stack(online)
    spectrum = torch.linalg.svdvals(values.float() - values.float().mean(0)).cpu().numpy()
    path = Path(directory) / "diagnostics" / f"step{step:09d}"
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "feature_spectrum.npz", singular_values=spectrum, clip_embeddings=values.cpu().numpy())
    report = {"successful_step": step, "sample_unit": "one_fixed_prefix_clip_per_training_source", "sources": identities,
        "online": representation_diagnostics(values), "teacher": representation_diagnostics(torch.stack(teacher)) if teacher else {"status": "not_used_by_objective"},
        "per_stream": {str(i): representation_diagnostics(torch.stack(x)) if x else {"status": "no_observations"} for i, x in enumerate(typed)},
        "quality_availability_buckets": qualities, "projector_gradient_norms": {
            str(i): float(torch.sqrt(sum((p.grad.float().square().sum() for p in projector.parameters() if p.grad is not None), values.new_zeros(()))))
            for i, projector in enumerate(model.online.projectors)}, "collapse_claim": "diagnostics only; no label/probe guarantee"}
    write_json(path / "report.json", report)
    import shutil
    shutil.copyfile(path / "feature_spectrum.npz", Path(directory) / "feature_spectrum.npz")
    return report
