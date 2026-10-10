"""Filesystem locations (read-only inputs) and output roots for obs_jepa."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUNTIME = ROOT / "runtime"
VENDOR_VJEPA = ROOT / "vendor" / "vjepa2_src"
VJEPA_CKPT = REPO / "method" / "signrel_jepa" / "assets" / "vjepa2_1_vitb_dist_vitG_384.pt"

ASLC_ROOT = Path("/home/dongvk/datasets/ASL_Citizen")
ASLC_KP = ASLC_ROOT / "keypoints"            # COCO-WholeBody 133-kpt, normalised xy per frame
ASLC_SPLITS = ASLC_ROOT / "splits"
# SignRep-reproduction pretraining corpus (Codex, ASLC002): 3,000 unlabelled ASL Citizen train videos
ASLC002_CORPUS = REPO / "method" / "jepa_baseline" / "runtime" / "ASLC002" / "corpus" / "pretrain.jsonl"
ASLC002_RUN = REPO / "method" / "jepa_baseline" / "runtime" / "ASLC002"
