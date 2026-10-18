import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
mt5_path = os.environ.get("LA_SIGN_MT5_PATH", str(ROOT / "pretrained_weight/mt5-base"))

# label paths
train_label_paths = {
                    "WLASL2000": "./data/WLASL/labels-2000.train",
                    "WLASL300": "./data/WLASL/labels-300.train",
                    "MSASL1000": "./data/MSASL/labels-1000.train",
                    "MSASL200": "./data/MSASL/labels-200.train",
                    "MSASL100": "./data/MSASL/labels-100.train",
                    }

dev_label_paths = {
                    "WLASL2000": "./data/WLASL/labels-2000.dev",
                    "WLASL300": "./data/WLASL/labels-300.val",
                    "MSASL1000": "./data/MSASL/labels-1000.val",
                    "MSASL200": "./data/MSASL/labels-200.val",
                    "MSASL100": "./data/MSASL/labels-100.val"
                    }

test_label_paths = {
                    "WLASL2000": "./data/WLASL/labels-2000.test",
                    "WLASL300": "./data/WLASL/labels-300.test",
                    "MSASL1000": "./data/MSASL/labels-1000.test",
                    "MSASL200": "./data/MSASL/labels-200.test",
                    "MSASL100": "./data/MSASL/labels-100.test",
                    }

# pose paths
pose_dirs = {
            "WLASL": "./dataset/WLASL/pose_format",
            "MSASL": "./dataset/MSASL/pose_format",
            }

rgb_dirs = {
            "WLASL": "./dataset/WLASL/rgb_format",
            "MSASL": "./dataset/MSASL/rgb_format",
            }

# Resolve defaults independently of the shell's working directory.
for paths in (train_label_paths, dev_label_paths, test_label_paths, pose_dirs, rgb_dirs):
    for name, path in paths.items():
        paths[name] = str(ROOT / path)
if os.environ.get("LA_SIGN_DATA_ROOT"):
    for name in pose_dirs:
        pose_dirs[name] = str(Path(os.environ["LA_SIGN_DATA_ROOT"]) / name / "pose_format")
if os.environ.get('LA_SIGN_LABEL_ROOT'):
    for paths in (train_label_paths, dev_label_paths, test_label_paths):
        for name, path in paths.items():
            paths[name] = str(Path(os.environ['LA_SIGN_LABEL_ROOT']) / Path(path).parent.name / Path(path).name)
