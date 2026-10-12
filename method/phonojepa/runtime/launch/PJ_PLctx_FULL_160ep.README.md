# PLctx full ASL Citizen train, 160 epochs

Authorized by the user on 2026-10-12. Training only; no test-set evaluation.

- Name: `PJ_PLctx_FULL_160ep`
- Data: all 40,154 unique clips in the official ASL Citizen train split. Cache
  construction checks that train IDs do not overlap validation or test IDs.
- Initialization: scratch, seed 0.
- Method: PL masks, contextual teacher targets, centered CE, EMA, unchanged
  `pj/train.py` recipe (256-dimensional encoder, depth 6, bf16, learning rate
  0.0003, weight decay 0.05, batch 32, six data workers).
- Budget: 160 DataLoader epochs, `drop_last=True`, 1,254 steps per epoch,
  **200,640 optimizer steps**. Each shuffled epoch drops 26 clips; 6,420,480
  total clip draws. The approximately 6.2-hour estimate is not a time limit.
- Full dataset training differs from the earlier 3,000-clip ASLC002 protocol.
  Future comparisons should identify the different pretraining data budget.
- Training cache: `runtime/cache/aslc_train_full.npz`; its adjacent manifest
  records the split and cache SHA-256. Labels are absent from cache arrays.
- The existing train-only ASLC002 target statistics are retained. Contextual
  training does not use the intrinsic regression target normalization.
- Checkpoints: `runtime/runs/PJ_PLctx_FULL_160ep/ckpt.pt` every 500 steps;
  `final.pt` after the complete budget. Existing runs are not overwritten.

The detached tmux session is `plctx_full160`. The launcher and stage logs,
configuration, source hashes and status are in this directory. The launcher
waits for existing user train/extract processes and sufficient free GPU memory.
It performs no automatic retries.

Monitor:

```bash
tail -f runtime/launch/PJ_PLctx_FULL_160ep.train.log
tail -f runtime/runs/PJ_PLctx_FULL_160ep/train_log.jsonl
cat runtime/launch/PJ_PLctx_FULL_160ep.status.json
```

If interrupted, the latest checkpoint supports `pj/train.py --resume` using
the same arguments as the `train` command recorded in the status JSON. Resume
continues the step budget; the original trainer does not guarantee bitwise
reproduction of the uninterrupted data order.
