"""Conservative temporal perceptual duplicate candidates, not semantic similarity."""
import numpy as np


def video_fingerprint(record, samples=8):
    import cv2
    from .decode import decode_video
    video = decode_video(record.video_path, record.start_s, record.end_s)
    indices = np.rint(np.linspace(0, len(video.frames) - 1, samples)).astype(int)
    bits = []
    for frame in video.frames[indices]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(small)[:8, :8].flatten()[1:]
        bits.extend((low > np.median(low)).tolist())
    return np.asarray(bits, dtype=np.uint8)


def cross_split_candidates(records, *, max_hamming_fraction=.05):
    from ..cache.hashing import canonical_hash
    fingerprints = {r.sample_id: video_fingerprint(r) for r in records if r.availability == "available"}
    candidates = []
    available = [r for r in records if r.sample_id in fingerprints]
    for i, first in enumerate(available):
        for second in available[i + 1:]:
            if first.split == second.split:
                continue
            distance = float(np.mean(fingerprints[first.sample_id] != fingerprints[second.sample_id]))
            if distance <= max_hamming_fraction:
                candidates.append({"first": first.sample_id, "second": second.sample_id, "hamming_fraction": distance})
    return {"recipe": "8_uniform_frames_gray32_dct8_drop_dc_median_bits", "threshold": max_hamming_fraction,
            "fingerprints_hash": canonical_hash({k: v.tolist() for k, v in sorted(fingerprints.items())}),
            "candidates": candidates, "status": "review_required" if candidates else "passed_no_candidates"}
