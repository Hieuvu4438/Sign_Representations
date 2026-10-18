#!/usr/bin/env python3
"""End-to-end step throughput and closure run card. Refuse a busy GPU."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp.config import Config
from pp.data import PoseSource,Windows,collate
from pp.signature import RobustStats
from pp.train import Trainer,atomic_json
from schedule import gpu_processes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=20);ap.add_argument('--warmup',type=int,default=3);a=ap.parse_args()
    cfg=Config.load(a.config)
    if cfg.device.startswith('cuda') and len(gpu_processes())>1:raise SystemExit('GPU busy (>1 other compute process); retry after a slot is free')
    torch.set_num_threads(4);src=PoseSource(cfg.cache);ds=Windows(src,cfg)
    if len(src)<cfg.batch:raise ValueError('Source too small')
    batch=collate([ds[i] for i in range(cfg.batch)]);stats=RobustStats.load(cfg.stats,cfg.asinh) if cfg.paths else None
    tr=Trainer(cfg,stats)
    for step in range(a.warmup):tr.step(batch,step,a.steps+a.warmup)
    if cfg.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    for step in range(a.steps):loss,gn,terms=tr.step(batch,step+a.warmup,a.steps+a.warmup)
    if cfg.device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    result=dict(config=cfg.to_dict(),steps=a.steps,seconds=elapsed,steps_per_second=a.steps/elapsed,loss=loss,
                grad_norm=gn,closure=terms,peak_gpu_bytes=torch.cuda.max_memory_allocated() if cfg.device.startswith('cuda') else None,
                target_steps_per_second=5.,meets_target=a.steps/elapsed>=5 if cfg.device.startswith('cuda') else None)
    atomic_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
