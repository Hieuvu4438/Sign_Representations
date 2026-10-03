import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def relocated_local_path(value):
    """Resolve archived absolute paths without rewriting hash-locked manifests."""
    path = Path(value)
    if path.exists() or not path.is_absolute():
        return path
    repository = ROOT.parents[1]
    try:
        relative = path.relative_to(repository)
    except ValueError:
        return path
    candidate = ROOT / relative
    return candidate if candidate.exists() else path
