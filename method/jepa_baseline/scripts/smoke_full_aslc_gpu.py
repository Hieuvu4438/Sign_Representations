"""Technical batch20/Kinetics/BF16 check and real interrupted-update replay."""

import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from signrep_baseline.augmentation import VideoTransform, AugmentationConfig
from signrep_baseline.cli import load_config
from signrep_baseline.data import PairedVideoDataset, StatefulPairSampler
from signrep_baseline.job_support import OrderedPairLoader, atomic_json
from signrep_baseline.model import SignRep
from signrep_baseline.schema import FidelityChoices
from signrep_baseline.training import Pretrainer, seed_everything


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 unavailable")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    config = load_config(ROOT / "configs/pretrain_full_aslc_kinetics.yaml")
    dataset = PairedVideoDataset(ROOT / "runtime/ASLC002/corpus/pretrain.jsonl",
              VideoTransform(True, AugmentationConfig(**config["augmentation"])),
              FidelityChoices(**config["choices"]))
    seed_everything(config["seed"])
    model = SignRep()
    heads = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    model.initialize_backbone(config["checkpoint"])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in heads.items())
    del heads
    trainer = Pretrainer(model, device="cuda", total_steps=100000, warmup_steps=10000,
                          max_lr=1e-4, amp=True, amp_dtype="bfloat16")
    sampler = StatefulPairSampler(len(dataset), 10, config["seed"])
    loader = OrderedPairLoader(dataset, sampler, config["seed"], workers=4, prefetch=1, pin_memory=True)
    contract = {"scope": "technical_gpu_replay_only; pilot targets; not production checkpoint"}
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gpu_resume_", dir=ROOT / "runtime") as temp:
        path = Path(temp) / "last.pt"
        try:
            first = loader.next_batch()
            first_metrics = trainer.update(first)
            loader.commit(1)
            trainer.save(path, sampler, contract)
            second = loader.next_batch()
            expected_metrics = trainer.update(second)
            loader.commit(2)
            expected_hashes = {k: state_hash(getattr(trainer, k)) for k in ("model", "discriminator", "ema")}
            peak = torch.cuda.max_memory_reserved() / 2**20
            loader.close()
            del trainer, model, first
            gc.collect()
            torch.cuda.empty_cache()
            restored = Pretrainer(SignRep(), device="cuda", total_steps=100000, warmup_steps=10000,
                                  max_lr=1e-4, amp=True, amp_dtype="bfloat16")
            restored_sampler = StatefulPairSampler(len(dataset), 10, config["seed"])
            restored.resume(path, restored_sampler, contract)
            restored_loader = OrderedPairLoader(dataset, restored_sampler, config["seed"], completed_steps=1,
                                               workers=4, prefetch=1, pin_memory=True)
            try:
                replay_batch = restored_loader.next_batch()
                for key in ("images", "frame_indices"):
                    assert torch.equal(second[key], replay_batch[key])
                assert second["video_ids"] == replay_batch["video_ids"]
                replay_metrics = restored.update(replay_batch)
                restored_loader.commit(2)
                actual_hashes = {k: state_hash(getattr(restored, k)) for k in expected_hashes}
                assert replay_metrics == expected_metrics
                assert actual_hashes == expected_hashes
            finally:
                restored_loader.close()
        finally:
            loader.close()
    report = {"status": "PASS", "scope": contract["scope"], "initialization": "kinetics_mae",
              "signrep_heads_started_fresh": True, "batch_views": 20, "dtype": "bfloat16",
              "data_workers": 4, "GPU": torch.cuda.get_device_name(),
              "peak_reserved_MiB": peak, "GPU_memory_fraction": .22,
              "encoder_discriminator_ema_after_resume_bitwise_equal": True,
              "next_batch_bitwise_equal": True, "metrics_equal": True,
              "first_metrics": first_metrics, "second_metrics": expected_metrics,
              "wall_seconds": time.monotonic() - start}
    output = ROOT / "evidence/full_aslc_gpu_resume.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
