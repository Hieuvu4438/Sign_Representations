"""Real paired-video CPU updates and bitwise interrupted/resumed replay.

Technical verification only: small ASL-Citizen clips and released SignRep init,
not the paper's YT-SL pretraining run or a benchmark experiment.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from signrep_baseline.augmentation import VideoTransform
from signrep_baseline.cache import sha256
from signrep_baseline.data import PairedVideoDataset, StatefulPairSampler, collate_pairs
from signrep_baseline.model import SignRep
from signrep_baseline.training import Pretrainer, seed_everything


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    seed_everything(31)
    folder = ROOT / "runtime/smoke"
    rows = []
    for suffix in ["", "2"]:
        video, cache = folder / f"real_pose_clip{suffix}.mp4", folder / f"standardized{suffix}.npz"
        with np.load(cache) as file:
            metadata = json.loads(str(file["metadata_json"]))
        import cv2
        cap = cv2.VideoCapture(str(video))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rows.append({"video_id": f"technical_clip{suffix or '1'}", "youtube_id": f"technical_clip{suffix or '1'}",
                     "split": "pretrain", "video": str(video.resolve()), "pose_cache": str(cache.resolve()),
                     "pose_cache_sha256": sha256(cache),
                     "video_sha256": sha256(video), "bone_statistics_sha256": metadata["bone_statistics_sha256"],
                     "signer_bbox_xyxy": [0, 0, w, h], "scope": "technical_only; full frame smoke crop"})
    manifest = folder / "pretrain_smoke.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = PairedVideoDataset(manifest, VideoTransform(True))
    sampler = StatefulPairSampler(len(dataset), 2, seed=31)
    model = SignRep()
    model.load_released(ROOT.parents[1] / "checkpoints/signrep/ckpt.pt")
    trainer = Pretrainer(model, total_steps=4, warmup_steps=1)
    contract = {"manifest_sha256": dataset.manifest_sha256, "scope": "technical_only", "seed": 31}
    first_batch = collate_pairs([dataset[i] for i in sampler.next_indices()])
    before = tensor_hash(trainer.model.state_dict())
    first_metrics = trainer.update(first_batch)
    assert before != tensor_hash(trainer.model.state_dict())
    checkpoint = folder / "pretrain_step1.pt"
    trainer.save(checkpoint, sampler, contract)
    second_batch = collate_pairs([dataset[i] for i in sampler.next_indices()])
    second_metrics = trainer.update(second_batch)
    expected_model = tensor_hash(trainer.model.state_dict())
    expected_discriminator = tensor_hash(trainer.discriminator.state_dict())
    expected_ema = tensor_hash(trainer.ema.state_dict())
    try:
        trainer.select_validation({"split": "test", "task": "dictionary_retrieval", "per_instance_top1": 99},
                                   folder / "should_not_exist.pt", sampler, contract, interval=2)
        raise AssertionError("Test selection should fail")
    except ValueError:
        pass
    del model, trainer
    gc.collect()
    restored = Pretrainer(SignRep(), total_steps=4, warmup_steps=1)
    restored_sampler = StatefulPairSampler(len(dataset), 2, seed=900)
    restored.resume(checkpoint, restored_sampler, contract)
    restored_batch = collate_pairs([dataset[i] for i in restored_sampler.next_indices()])
    assert restored_batch["video_ids"] == second_batch["video_ids"]
    assert torch.equal(restored_batch["frame_indices"], second_batch["frame_indices"])
    assert torch.equal(restored_batch["images"], second_batch["images"])
    replay_metrics = restored.update(restored_batch)
    assert tensor_hash(restored.model.state_dict()) == expected_model
    assert tensor_hash(restored.discriminator.state_dict()) == expected_discriminator
    assert tensor_hash(restored.ema.state_dict()) == expected_ema
    assert replay_metrics == second_metrics
    assert restored.select_validation({"split": "val", "task": "dictionary_retrieval", "per_instance_top1": 12.5},
                                      folder / "best_smoke.pt", restored_sampler, contract, interval=2)
    report = {"result": "PASS", "scope": "technical only; not benchmark reproduction",
              "batch_videos": 2, "batch_views": 4, "first_update": first_metrics,
              "second_update": second_metrics, "resume_batch_bitwise_equal": True,
              "resume_encoder_discriminator_ema_bitwise_equal": True,
              "resume_metrics_equal": True, "test_checkpoint_selection_rejected": True,
              "validation_checkpoint_selection": True, "next_model_sha256": expected_model}
    (ROOT / "evidence/S4_training_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
