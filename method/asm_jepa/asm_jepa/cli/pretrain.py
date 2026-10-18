"""ASM-JEPA SSL pretraining on the official ASL-Citizen train split (label-free batches).

Examples:
  python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --synthetic --run-dir runs/smoke_synthetic \
      --set train.max_steps=20 --set train.warmup_steps=2 --set train.batch_size_per_gpu=4 --set train.accum_steps=1 \
      --set data.num_workers=0 --set eval.frozen_every_steps=0 --set train.checkpoint_every_steps=10
  python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/aslc_asm_state_s0
  python -m asm_jepa.cli.pretrain --config configs/pretrain/state.yaml --run-dir runs/aslc_asm_state_s0 \
      --resume runs/aslc_asm_state_s0/checkpoints/last.pt
"""
from __future__ import annotations

import argparse

from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.utils import resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap, required=True)
    ap.add_argument("--run-dir", help="default: run_dir from the config (relative to method/asm_jepa)")
    ap.add_argument("--synthetic", action="store_true", help="in-memory synthetic poses + fixture stats (smoke only)")
    ap.add_argument("--limit-train-samples", type=int, help="deterministic train subset (smoke runs)")
    ap.add_argument("--resume", help="checkpoint to resume from (same config/manifest/stats required)")
    ap.add_argument("--limit-eval-samples", type=int, help="smoke only: first N gallery and N val clips for validation")
    ap.add_argument("--profile-steps", type=int, help="throughput benchmark for N optimizer steps")
    ap.add_argument("--device")
    ap.add_argument("--dry-run", action="store_true", help="resolve config, build model, print sizes, exit")
    args = ap.parse_args()
    cfg = config_from_args(args, "pretrain")
    run_dir = resolve(args.run_dir or cfg["run_dir"])
    cfg["run_dir"] = str(run_dir)
    if args.dry_run:
        import yaml

        from asm_jepa.features.tokens import layout_from_config
        from asm_jepa.models.asm_jepa import ASMJEPA

        from asm_jepa.config import strip_meta

        layout = layout_from_config(cfg)
        model = ASMJEPA(cfg)
        print(yaml.safe_dump(strip_meta(cfg), sort_keys=False))
        print(f"tokens={layout.n_tokens} (state {layout.n_state}, motion {layout.n_motion}) "
              f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
              f"encoder={sum(p.numel() for p in model.online.parameters()) / 1e6:.2f}M "
              f"anchor={model.anchor_enabled} global={model.global_enabled} objective={model.objective}")
        return 0
    from asm_jepa.training.engine import PretrainEngine

    if args.synthetic:
        print("SYNTHETIC run: metrics are code checks only, never ASL-Citizen results", flush=True)
    engine = PretrainEngine(cfg, run_dir, synthetic=args.synthetic, limit_train=args.limit_train_samples,
                            limit_eval=args.limit_eval_samples,
                            resume=args.resume, profile_steps=args.profile_steps, device=args.device)
    engine.train()
    return 0


if __name__ == "__main__":
    run(main)
