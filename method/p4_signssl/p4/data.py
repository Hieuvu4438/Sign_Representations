"""Dataset manifests and video decoding for p4_signssl."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ASLC_ROOT = Path("/home/dongvk/datasets/ASL_Citizen")


def aslc_rows(split):
    """Official ASL Citizen split rows, in file order. Labels are gloss strings."""
    rows = []
    with open(ASLC_ROOT / "splits" / f"{split}.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "sample_id": Path(r["Video file"]).stem,
                    "path": str(ASLC_ROOT / "videos" / r["Video file"]),
                    "gloss": r["Gloss"],
                    "signer": r["Participant ID"],
                    "asllex": r["ASL-LEX Code"],
                    "split": split,
                }
            )
    return rows


WLASL_ROOT = Path("/home/dongvk/datasets/WLASL2000")


def wlasl_rows(split):
    """WLASL2000 local copy (pre-trimmed 256x256 clips); split from preprocess/nslt_2000.json.
    Signer IDs are not in this file -> 'signer' = 'NA' (WLASL splits are not signer-disjoint)."""
    meta = json.load(open(WLASL_ROOT / "preprocess" / "nslt_2000.json"))
    names = [l.split("\t")[1].strip() for l in open(WLASL_ROOT / "preprocess" / "wlasl_class_list.txt")]
    rows = []
    for vid in sorted(meta):
        m = meta[vid]
        if m["subset"] != split:
            continue
        rows.append({"sample_id": vid, "path": str(WLASL_ROOT / "WLASL2000" / f"{vid}.mp4"),
                     "gloss": names[m["action"][0]], "signer": "NA", "asllex": "NA", "split": split})
    return rows


MSASL_ROOT = Path("/home/dongvk/datasets/MS-ASL")


def msasl_rows(split):
    """MS-ASL local cut clips (`cut_processed_video/<id>.mp4`); split + class index from
    `metadata_msasl_like_wsasl.json`, class names from `MSASL_classes.json`. Signer IDs are not
    mapped for the cut clips -> 'NA'. NOTE: MS-ASL test overlaps YouTube-ASL (contamination risk for
    YouTube-pretrained encoders such as SignRep)."""
    meta = json.load(open(MSASL_ROOT / "metadata_msasl_like_wsasl.json"))
    names = json.load(open(MSASL_ROOT / "MSASL_classes.json"))
    rows = []
    for vid in sorted(meta):
        m = meta[vid]
        if m["subset"] != split:
            continue
        rows.append({"sample_id": vid, "path": str(MSASL_ROOT / "cut_processed_video" / f"{vid}.mp4"),
                     "gloss": names[m["action"][0]], "signer": "NA", "asllex": "NA", "split": split})
    return rows


def manifest_hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()


def decode_resized(path, size=224, max_frames=1000):
    """cv2 decode -> RGB -> resize (INTER_LINEAR, same as albumentations A.Resize default)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR))
    cap.release()
    arr = np.stack(frames) if frames else np.zeros((0, size, size, 3), np.uint8)
    return arr, fps


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, rows, size=224):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr, fps = decode_resized(self.rows[i]["path"], self.size)
        return i, torch.from_numpy(arr), float(fps)


def collate_single(batch):
    return batch[0]
