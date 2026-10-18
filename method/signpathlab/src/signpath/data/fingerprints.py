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


def cross_split_candidates(records, *, max_hamming_fraction=.05, fingerprints=None):
    from ..cache.hashing import canonical_hash
    fingerprints = ({r.sample_id: video_fingerprint(r) for r in records if r.availability == "available"}
                    if fingerprints is None else fingerprints)
    candidates = []
    available = [r for r in records if r.sample_id in fingerprints]
    for first_split in sorted({r.split for r in available}):
        first_records = [r for r in available if r.split == first_split]
        second_records = [r for r in available if r.split > first_split]
        if not second_records:
            continue
        gallery = np.packbits(np.stack([fingerprints[r.sample_id] for r in second_records]), axis=1)
        lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
        bit_count = len(fingerprints[first_records[0].sample_id])
        for start in range(0, len(first_records), 16):
            rows = first_records[start:start + 16]
            query = np.packbits(np.stack([fingerprints[r.sample_id] for r in rows]), axis=1)
            for offset in range(0, len(second_records), 4096):
                distance = lookup[np.bitwise_xor(query[:, None], gallery[None, offset:offset + 4096])].sum(axis=2) / bit_count
                for i, j in np.argwhere(distance <= max_hamming_fraction):
                    candidates.append({"first": rows[i].sample_id, "second": second_records[offset + j].sample_id,
                                       "hamming_fraction": float(distance[i, j])})
    return {"recipe": "8_uniform_frames_gray32_dct8_drop_dc_median_bits", "threshold": max_hamming_fraction,
            "fingerprints_hash": canonical_hash({k: v.tolist() for k, v in sorted(fingerprints.items())}),
            "candidates": candidates, "status": "review_required" if candidates else "passed_no_candidates"}
