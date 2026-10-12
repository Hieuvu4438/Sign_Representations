from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from signpath.backbones.dino import DINOAdapter
from signpath.pose.extract import match_hands
from signpath.pose.roi import landmark_box, letterbox


def test_wrist_matching_crossed_hands_and_unrelated_missing_wrist():
    body = np.zeros((49, 2))
    body[1], body[2] = [.3, .4], [.7, .4]
    body[5], body[6] = [.8, .5], [.2, .5]  # left anatomical hand is on image right
    hands = np.zeros((2, 21, 2))
    hands[0] = [.2, .5]
    hands[1] = [.8, .5]
    quality = np.ones(49)
    assert match_hands(body, quality, hands, [640, 480]).tolist() == [1, 0]
    # A missing unrelated wrist must not discard an unambiguous observed right hand.
    quality[5], body[5] = 0, np.nan
    assert match_hands(body, quality, hands[:1], [640, 480]).tolist() == [-1, 0]
    # A candidate matching a wrist that is itself unreliable is rejected.
    quality[6] = 0
    assert match_hands(body, quality, hands[:1], [640, 480]).tolist() == [-1, -1]


def test_ambiguous_hand_assignment_is_invalid():
    body = np.zeros((49, 2))
    body[1], body[2] = [.3, .4], [.7, .4]
    body[5], body[6] = [.5, .5], [.5, .5]
    hands = np.ones((2, 21, 2)) * .5
    assert match_hands(body, np.ones(49), hands, [640, 480]).tolist() == [-1, -1]


def test_roi_retains_extent_and_affine():
    pytest.importorskip("cv2")
    image = np.full((40, 80, 3), [255, 0, 0], dtype=np.uint8)
    crop = letterbox(image)
    assert crop.image.shape == (224, 224, 3)
    assert crop.padding_fraction == pytest.approx(.5)
    np.testing.assert_allclose(crop.affine @ [40, 20, 1], [112, 112])
    assert crop.image[112, 112].tolist() == [255, 0, 0]
    border = letterbox(image, [-20, -20, 100, 60])
    assert border.padding_fraction > .5
    assert landmark_box([[.1, .1], [.2, .2]]) is None
    with pytest.raises(ValueError):
        letterbox(image, [0, 0, 0, 1])


def test_dino_rejects_unverified_assets_before_loading(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not a model")
    with patch("torch.hub.load") as hub:
        with pytest.raises(ValueError, match="verified SHA"):
            DINOAdapter().load({"repo_dir": str(tmp_path), "local_path": str(weights), "sha256": None}, "cpu")
        hub.assert_not_called()
