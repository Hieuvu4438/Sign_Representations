"""Environment check: GPU/CUDA/BF16, library versions, free disk/RAM, config paths. Never prints secrets."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys

from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.utils import resolve


def _ram_gb():
    try:
        with open("/proc/meminfo") as f:
            info = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return info["MemTotal"] / 2**20, info.get("MemAvailable", 0) / 2**20
    except OSError:
        return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, default="configs/pretrain/state.yaml")
    p.add_argument("--schema", default="pretrain")
    args = p.parse_args()
    import numpy
    import torch
    import yaml

    print(f"python      {sys.version.split()[0]} ({platform.platform()})")
    print(f"torch       {torch.__version__}  cuda build {torch.version.cuda}")
    print(f"numpy       {numpy.__version__}   pyyaml {yaml.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"gpu[{i}]      {prop.name}  {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free, "
                  f"cc {prop.major}.{prop.minor}")
        print(f"bf16        {torch.cuda.is_bf16_supported()}")
    else:
        print("gpu         none (CPU only)")
    total, avail = _ram_gb()
    if total:
        print(f"ram         {total:.1f} GB total, {avail:.1f} GB available")
    print(f"cpus        {os.cpu_count()}")
    cfg = config_from_args(args, args.schema)
    root = resolve(cfg.get("data", {}).get("processed_root", "."))
    du = shutil.disk_usage(root if root.exists() else resolve("."))
    print(f"disk        {du.free / 2**30:.1f} GB free at {root if root.exists() else resolve('.')}")
    print(f"config      {cfg['_meta']['source']}")
    if args.schema == "pretrain":
        d = cfg["data"]
        checks = {"processed_root": root, "ssl_manifest": root / d["ssl_manifest"], "stats": root / d["stats_path"],
                  "metadata_train": root / "metadata_train.jsonl", "metadata_val": root / "metadata_val.jsonl",
                  "poses_dir": root / "poses"}
        for k, v in checks.items():
            print(f"  {k:15s} {'ok     ' if v.exists() else 'MISSING'} {v}")
    for mod in ("cv2", "rtmlib", "onnxruntime"):
        try:
            m = __import__(mod)
            extra = ""
            if mod == "onnxruntime":
                extra = f" providers={m.get_available_providers()}"
            print(f"optional    {mod} {getattr(m, '__version__', '?')}{extra}")
        except ImportError:
            print(f"optional    {mod} not installed (only needed for pose extraction / overlays)")
    return 0


if __name__ == "__main__":
    run(main)
