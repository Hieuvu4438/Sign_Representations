"""Fetch pinned upstream source/weights inside this isolated project only."""
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from signcoord.models.generic_hiera_adapter import OFFICIAL_WEIGHT_SHA256, UPSTREAM_REVISION, import_upstream
from signcoord.provenance import sha256_file

root = PROJECT / "vendor/hiera"
if not root.exists():
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/hiera.git", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", UPSTREAM_REVISION], check=True)
revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if revision != UPSTREAM_REVISION:
    raise ValueError("existing checkout differs; preserving it instead of resetting")
import_upstream(root)
weights = PROJECT / "runs/assets/mae_hiera_base_16x224.pth"
url = "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth"
if not weights.exists():
    weights.parent.mkdir(parents=True, exist_ok=True)
    temporary = weights.with_suffix(".download")
    urllib.request.urlretrieve(url, temporary)
    if sha256_file(temporary) != OFFICIAL_WEIGHT_SHA256:
        raise ValueError("download checksum mismatch; temporary file retained for inspection")
    temporary.replace(weights)
if sha256_file(weights) != OFFICIAL_WEIGHT_SHA256:
    raise ValueError("existing weights checksum mismatch")
print(f"verified source {revision}; weights {OFFICIAL_WEIGHT_SHA256}")
