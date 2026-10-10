import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path
import torch
import yaml
from signrel.data.manifest import sha256


def command_output(command,cwd=None):
    result = subprocess.run(command,cwd=cwd,capture_output=True,text=True)
    return result.stdout if result.returncode==0 else result.stderr


def source_hash(root):
    import hashlib
    h = hashlib.sha256()
    paths = []
    for folder in ("signrel","scripts"):
        paths += [p for p in (root/folder).rglob("*") if p.suffix in {".py",".yaml"}]
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def record_run(run_dir,config,manifest,upstream_init,root):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    values = {
        "config.resolved.yaml":yaml.safe_dump(config,sort_keys=True),
        "command.txt":shlex.join([sys.executable,*sys.argv])+"\n",
        "git_commit.txt":command_output(["git","rev-parse","HEAD"],root),
        "git_diff.patch":command_output(["git","diff","--",str(root)],root),
        "environment.txt":json.dumps({"python":sys.version,"platform":platform.platform(),
                                      "torch":torch.__version__,"cuda":torch.version.cuda},indent=2),
        "pip_freeze.txt":command_output([sys.executable,"-m","pip","freeze"]),
        "cuda_info.txt":command_output(["nvidia-smi"]),
        "dataset_manifest.sha256":sha256(manifest)+"\n",
        "upstream_checkpoints.json":json.dumps(upstream_init,indent=2),
        "source.sha256":source_hash(root)+"\n",
    }
    for name,value in values.items():
        (run_dir/name).write_text(value)
    # Capture local untracked source as well as git tracked diffs.
    import tarfile
    with tarfile.open(run_dir/"source.tar.gz","w:gz") as archive:
        for folder in ("signrel","scripts","configs"):
            for p in sorted((root/folder).rglob("*")):
                if p.suffix in {".py",".yaml"}:
                    archive.add(p,arcname=str(p.relative_to(root)))
