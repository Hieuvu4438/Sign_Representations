import pickle
import numpy as np
from signrepr.data.fixture import fixture_pose
from signrepr.data.person_tracking import PersonTracker
from signrepr.data.import_unisign import import_unisign_pose
from signrepr.data.schema import load_pose
import pytest


def person(center, size=.2):
    xy = fixture_pose(1)['xy'][0]
    xy = (xy-.5)*size+np.array(center)
    return xy.astype(np.float32)


def test_tracker_crossing_reordered_candidates_and_disappearance():
    tracker = PersonTracker()
    selected = []
    for i in range(9):
        a, b = person((.3+.04*i, .5), .6), person((.7-.04*i, .5), .4)
        # Detector order switches, IDs are not frame array positions.
        points = np.stack([a, b] if i%2 == 0 else [b, a])
        index, _ = tracker.step(points, np.full((2, 133), .9, np.float32))
        assert index == (0 if i%2 == 0 else 1)
        selected.append(index)
    index, continuity = tracker.step(np.empty((0, 133, 2)), np.empty((0, 133)))
    assert index is None and not continuity
    # Keep missing rather than jump to a far-away bystander.
    index, continuity = tracker.step(person((.1, .1))[None], np.full((1, 133), .9))
    assert index is None and not continuity


def test_import_preserves_missing_frame_unknown_fps_and_person_selection(tmp_path):
    path = tmp_path/'source.pkl'
    keypoints = [np.stack([person((.8, .8), .2), person((.4, .5), .6)]), np.empty((0, 133, 2)), person((.4, .5), .6)[None]]
    scores = [np.full((len(k), 133), .9, np.float32) for k in keypoints]
    with open(path, 'wb') as f:
        pickle.dump({'keypoints': keypoints, 'scores': scores}, f)
    row, report = import_unisign_pose(path, {'joint_layout': 'coco_wholebody_133', 'coordinate_system': 'viewport_normalized'}, tmp_path/'pose.npz')
    pose = load_pose(tmp_path/'pose.npz')
    assert row['num_frames'] == 3
    assert not bool(pose['timing_known'])
    np.testing.assert_array_equal(pose['xy'][0], keypoints[0][1])
    assert not pose['observed'][1].any()
    assert not pose['track_continuity'].any()
    assert report['multiple_person_frames'] == 1


def test_explicit_score_adaptation_preserves_raw_and_default_rejects(tmp_path):
    path = tmp_path/'source.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'keypoints': [person((.4, .5), .6)[None]], 'scores': [np.full((1, 133), 9., np.float32)]}, f)
    metadata = {'joint_layout': 'coco_wholebody_133', 'coordinate_system': 'viewport_normalized'}
    with pytest.raises(ValueError, match='confidence'):
        import_unisign_pose(path, metadata, tmp_path/'rejected.npz')
    row, _ = import_unisign_pose(path, {**metadata, 'score_profile': 'simcc_positive_rational_v1'}, tmp_path/'converted.npz')
    np.testing.assert_allclose(load_pose(tmp_path/'converted.npz')['confidence'], .9)
    with np.load(row['raw_detector_scores_path']) as raw:
        assert np.all(raw['detector_scores'] == 9)
    assert row['raw_score_range'] == [9., 9.]
