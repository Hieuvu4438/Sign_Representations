"""Filesystem locations. Inputs are read-only; every output goes under method/phonojepa/runtime."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUNTIME = ROOT / "runtime"

ASLC_ROOT = Path("/home/dongvk/datasets/ASL_Citizen")
ASLC_KP = ASLC_ROOT / "keypoints"          # COCO-WholeBody 133 kpt, x/W and y/H per frame, unnormalised scores
ASLC_SPLITS = ASLC_ROOT / "splits"

# proposal11 / EX16 comparison protocol: the SignRep-reproduction corpus (3,000 unlabelled ASL Citizen train videos)
ASLC002_RUN = REPO / "method" / "jepa_baseline" / "runtime" / "ASLC002"
ASLC002_CORPUS = ASLC002_RUN / "corpus" / "pretrain.jsonl"
OBS_JEPA = REPO / "method" / "obs_jepa"
OBS_META = OBS_JEPA / "runtime" / "meta_aslc002.json"     # (W, H, fps, n_frames) per corpus video, read-only
