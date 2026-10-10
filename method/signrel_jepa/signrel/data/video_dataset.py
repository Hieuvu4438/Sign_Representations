import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .manifest import read_manifest, ssl_record, assert_no_labels
from .transforms import transform_clip


def decode_video(path, num_frames, stride=2, start_sec=None, end_sec=None, random_clip=False):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f"Corrupt/empty video: {path}")
        lo = max(0, int((start_sec or 0)*fps))
        hi = min(count, int(end_sec*fps) if end_sec is not None else count)
        if hi <= lo:
            raise ValueError(f"Empty segment in {path}: {lo}:{hi}")
        span = (num_frames-1)*stride+1
        if hi-lo >= span:
            max_offset = hi-lo-span
            offset = int(torch.randint(max_offset+1, ()).item()) if random_clip else max_offset//2
            indices = lo+offset+np.arange(num_frames)*stride
        else:
            indices = np.linspace(lo, hi-1, num_frames).round().astype(int)
        # Sequential decode within the span; avoids inaccurate GOP seeking per frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        frames, target = [], 0
        for frame_idx in range(int(indices[0]), int(indices[-1])+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Decode failed {path} frame {frame_idx}")
            while target < num_frames and indices[target] == frame_idx:
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                target += 1
        video = torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float()/255
        return video, indices, fps, count
    finally:
        cap.release()


class SSLVideoDataset(Dataset):
    def __init__(self, manifest, num_frames=16, sampling_stride=2, crop_size=224,
                 augment=True, require_landmarks=True):
        self.rows = [ssl_record(r) for r in read_manifest(manifest)]
        self.num_frames, self.stride, self.size = num_frames, sampling_stride, crop_size
        self.augment, self.require_landmarks = augment, require_landmarks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        video, idx, _, count = decode_video(r["video_path"], self.num_frames, self.stride,
                                           r.get("start_sec"), r.get("end_sec"), self.augment)
        if r.get("landmark_path"):
            with np.load(r["landmark_path"], allow_pickle=False) as cache:
                if int(cache["num_source_frames"]) != count:
                    raise AssertionError(f"Cache/video frame mismatch: {r['sample_id']}")
                source_indices = cache["frame_indices"]
                positions = np.searchsorted(source_indices, idx)
                if (positions >= len(source_indices)).any() or not np.array_equal(source_indices[positions], idx):
                    raise AssertionError("Landmark cache must cover every sampled frame; no silent time interpolation")
                boxes = torch.from_numpy(cache["boxes"][positions].copy()).float()
                confidence = torch.from_numpy(cache["box_confidence"][positions].copy()).float()
                motion = torch.from_numpy(cache["motion"][positions].copy()).float()
        else:
            if self.require_landmarks:
                raise ValueError(f"Missing landmark_path: {r['sample_id']}")
            boxes = torch.zeros(self.num_frames,4,4)
            confidence = torch.zeros(self.num_frames,4)
            motion = torch.zeros(self.num_frames,4)
        video, meta, record = transform_clip(video, boxes, confidence, size=self.size,
                                             augment=self.augment, crop=r.get("signer_crop") or (0,0,1,1))
        if record.flip:
            motion = motion[:, [1,0,2,3]]
        meta["motion"] = motion
        out = {"video": video, "articulator_meta": meta, "sample_id": r["sample_id"]}
        assert_no_labels(out)
        return out


def ssl_collate(samples):
    assert_no_labels(samples)
    from torch.utils.data import default_collate
    batch = default_collate(samples)
    assert_no_labels(batch)
    return batch
