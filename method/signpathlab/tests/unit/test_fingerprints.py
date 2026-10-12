from dataclasses import replace
import numpy as np
import pytest

from signpath.data.fingerprints import cross_split_candidates
from signpath.data.manifest import ManifestRecord


def test_perceptual_candidates_require_cross_split_and_do_not_claim_semantics(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("cv2")
    path = tmp_path / "pattern.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        for index in range(8):
            pixels = np.zeros((24, 32, 3), dtype=np.uint8)
            pixels[2:16, index:index + 8] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    first = ManifestRecord(1, "first", "fixture", "ase", "train", "a", str(path), availability="available")
    second = replace(first, sample_id="second", source_group_id="b", split="val")
    candidates = cross_split_candidates([first, second])
    assert candidates["status"] == "review_required"
    assert candidates["candidates"][0]["hamming_fraction"] == 0
    assert cross_split_candidates([first, replace(second, split="train")])["status"] == "passed_no_candidates"
