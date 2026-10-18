"""Supervised readouts.

linear probe : --config configs/eval/linear.yaml --train-embeddings E_train.npz --val-embeddings E_val.npz
fine-tune    : --config configs/eval/finetune.yaml --init <ssl checkpoint>|scratch
overfit check: --config configs/eval/finetune.yaml --init scratch --overfit-samples 16 --max-steps 500
Selection uses val Top-1; the test split is never read here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from asm_jepa.cli.common import run
from asm_jepa.config import load_config
from asm_jepa.utils import resolve


def config_mode(path) -> str | None:
    """`mode` of a classifier config, following relative `inherits` (child files may omit it)."""
    path = Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    if "mode" in d:
        return d["mode"]
    for parent in d.get("inherits") or []:
        m = config_mode(path.parent / parent)
        if m:
            return m
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-embeddings")
    ap.add_argument("--val-embeddings")
    ap.add_argument("--init", help="scratch or an SSL checkpoint (teacher tokenizer/encoder weights)")
    ap.add_argument("--overfit-samples", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--limit-train-samples", type=int, help="smoke runs only")
    ap.add_argument("--limit-eval-samples", type=int, help="smoke runs only: first N val clips")
    ap.add_argument("--few-shot-k", type=int, help="linear probe: k labelled train samples per class (1/2/4)")
    ap.add_argument("--few-shot-seed", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    mode = config_mode(args.config)
    run_dir = resolve(args.run_dir)
    if mode == "linear_probe":
        from asm_jepa.training.classifier import train_linear_probe

        cfg = load_config(args.config, "linear", args.set)
        if not (args.train_embeddings and args.val_embeddings):
            raise ValueError("linear probe needs --train-embeddings and --val-embeddings")
        train_linear_probe(cfg, resolve(args.train_embeddings), resolve(args.val_embeddings), run_dir, args.device,
                           args.few_shot_k, args.few_shot_seed)
    elif mode == "finetune_islr":
        from asm_jepa.training.classifier import FinetuneTrainer

        cfg = load_config(args.config, "finetune", args.set)
        if not args.init:
            raise ValueError("fine-tuning needs --init scratch or --init <checkpoint>")
        init = "scratch" if args.init == "scratch" else str(resolve(args.init))
        FinetuneTrainer(cfg, run_dir, init, args.device, args.overfit_samples, args.max_steps, args.resume,
                        args.limit_train_samples,
                        args.limit_eval_samples).train()
    else:
        raise ValueError(f"unknown classifier mode {mode!r}")
    return 0


if __name__ == "__main__":
    run(main)
