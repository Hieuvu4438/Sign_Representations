import time
import torch
from ..cache.hashing import file_hash
from ..cache.index import CachedCorpus
from ..cli import write_json
from ..data.dataset import BatchProvider
from ..models.teacher_student import ResearchModel
from ..targets.fitting import load_targets
from .engine import Trainer
from .runner import seed_runtime


def profile(resolved, steps, output, device="cpu"):
    if steps <= 0:
        raise ValueError("Profile needs positive update count")
    c = resolved.values
    seed_runtime(c["experiment"]["seed"])
    torch.set_num_threads(1)
    corpus = CachedCorpus(resolved.path(c["data"]["train_manifest"]), resolved.path(c["cache"]["root"]), split="train", config=c)
    geo, cluster = load_targets(resolved, corpus)
    provider = BatchProvider(c, corpus.pretrain_loader(), corpus.pretrain_sources(), geo, cluster_targets=cluster)
    trainer = Trainer(ResearchModel(c), c, device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    durations, loading, counts = [], [], []
    for _ in range(min(steps, c["training"]["max_successful_steps"])):
        start = time.monotonic()
        batches = [provider.next_batch() for _ in range(c["training"]["gradient_accumulation"])]
        loading.append(time.monotonic() - start)
        if cuda:
            torch.cuda.synchronize(device)
        start = time.monotonic()
        result = trainer.update(batches)
        if cuda:
            torch.cuda.synchronize(device)
        durations.append(time.monotonic() - start)
        counts.append({"success": result.succeeded, "targets": sum(sum(log["semantic"]["counts"]) for log in result.logs),
                       "geometry_targets": sum(sum(log["geometry"]["counts"]) for log in result.logs)})
    report = {"status": "completed_profile_only", "steps_attempted": len(durations), "successful_steps": trainer.successful_step,
        "loader_seconds": loading, "update_seconds": durations, "teacher_included": bool(c["loss"]["semantic_weight"]),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if cuda else None, "counts": counts,
        "config_hash": resolved.hash, "train_manifest_sha256": file_hash(corpus.manifest),
        "scope": "cached loading + student/teacher/predictor/backward/optimizer/EMA; excludes decode/detector/DINO costs"}
    write_json(output, report)
    return report
