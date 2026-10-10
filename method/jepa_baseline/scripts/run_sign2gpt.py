"""Run the original Sign2GPT stage2 Trainer using frozen SignRep feature manifests."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integrations"))
sys.path.insert(0, str(ROOT / "vendor/sign2gpt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="JSON with dataset/train_manifest/val_manifest/test_manifest/output")
    parser.add_argument("--inspect", action="store_true", help="Validate schema and import original trainer without training")
    args = parser.parse_args()
    os.environ["SIGNREP_TRANSLATION_CONFIG"] = str(args.config.resolve())
    os.environ["HF_HOME"] = str(ROOT / "runtime/huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "runtime/huggingface/transformers")
    from sign2gpt_config import get_config
    from signrep_baseline.cli import cached_feature_rows, held_out_contract
    from signrep_baseline.data import read_manifest
    cfg = get_config()
    pins = []
    for split in ["train", "val", "test"]:
        user = json.loads(args.config.read_text())
        _, pin = cached_feature_rows(user[f"{split}_manifest"], split)
        pins.append(pin)
    if len(set(pins)) != 1:
        raise ValueError("Translation partitions have different frozen encoder checkpoints")
    held_out_contract(user["train_manifest"], user["val_manifest"], "val")
    held_out_contract(user["train_manifest"], user["test_manifest"], "test")
    held_out_contract_pair = {r["video_id"] for r in read_manifest(user["val_manifest"], "val")}
    test_ids = {r["video_id"] for r in read_manifest(user["test_manifest"], "test")}
    if held_out_contract_pair & test_ids:
        raise ValueError("Translation validation and test overlap")
    from trainer.complete_translation_trainer import Trainer
    if args.inspect:
        print(cfg.to_json_best_effort())
        return
    import torch
    import ignite.distributed as idist
    if not torch.cuda.is_available():
        raise RuntimeError("Original Sign2GPT trainer/xformers recipe requires CUDA")
    output = Path(user["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(cfg.to_json_best_effort())
    with idist.Parallel("nccl", nproc_per_node=1) as parallel:
        parallel.run(Trainer, cfg)


if __name__ == "__main__":
    main()
