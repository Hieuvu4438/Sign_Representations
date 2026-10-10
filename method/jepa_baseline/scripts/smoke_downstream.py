"""Actual released-checkpoint RGB window extraction and 64-frame training update."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from signrep_baseline.cache import sha256
from signrep_baseline.model import SignRep, SignRecognizer
from signrep_baseline.retrieval import extract_windows, pool_windows
from signrep_baseline.recognition import RecognitionDataset, class_distribution
from signrep_baseline.recognition_training import RecognitionTrainer
from signrep_baseline.data import StatefulPairSampler
from signrep_baseline.training import seed_everything


def main():
    torch.set_num_threads(2)
    seed_everything(19)
    folder = ROOT / "runtime/smoke"
    checkpoint = ROOT.parents[1] / "checkpoints/signrep/ckpt.pt"
    model = SignRep().eval()
    model.load_released(checkpoint)
    rng_before = torch.get_rng_state().clone()
    outputs = extract_windows(model, folder / "real_pose_clip.mp4", batch_size=2)
    assert outputs["features"].shape == (11, 768)
    assert outputs["window_starts"].tolist() == list(range(0, 21, 2))
    assert torch.equal(torch.get_rng_state(), rng_before)
    metadata = json.loads(str(outputs["metadata_json"]))
    metadata["encoder_checkpoint_sha256"] = sha256(checkpoint)
    outputs["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(folder / "features.npz", **outputs)
    pooled = pool_windows(outputs["features"], outputs["activity"])
    assert torch.isfinite(pooled).all() and abs(float(pooled.norm()) - 1) < 1e-6
    table, _, _ = class_distribution(torch.tensor([[1., .01], [1., -.01], [1., 0.]]), [0, 1, 2], 3)
    recognition = SignRecognizer(model.inflate_for_recognition(), 3)
    assert not any('cls_head' in name or 'active_head' in name for name, _ in recognition.named_parameters())
    manifest = folder / "recognition_smoke.jsonl"
    manifest.write_text(json.dumps({"video_id": "real_clip1", "video": str((folder / 'real_pose_clip.mp4').resolve()),
                                    "label": 0, "split": "train"}) + '\n')
    dataset = RecognitionDataset(manifest, True)
    images, label = dataset[0]
    trainer = RecognitionTrainer(recognition, table, total_steps=3, warmup_steps=1)
    before = trainer.model.classifier.weight.detach().clone()
    metrics = trainer.update(images[None], label[None])
    assert not torch.equal(before, trainer.model.classifier.weight)
    sampler = StatefulPairSampler(2, 2)
    saved = folder / "recognition_step1.pt"
    trainer.save(saved, sampler, {"scope": "technical"})
    expected = trainer.model.classifier.weight.detach().clone()
    with torch.no_grad():
        trainer.model.classifier.weight.zero_()
    trainer.resume(saved, sampler, {"scope": "technical"})
    assert torch.equal(expected, trainer.model.classifier.weight)
    report = {"result": "PASS", "scope": "technical only; synthetic 3-class table, real RGB clip",
              "frozen_windows": 11, "stride": 2, "inference_rng_preserved": True,
              "pooled_feature_l2_norm": float(pooled.norm()), "recognition_frames": 64,
              "prior_decoders_removed": True, "recognition_update": metrics,
              "recognition_checkpoint_restore": True}
    (ROOT / 'evidence/S5_rgb_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
