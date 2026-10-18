from copy import deepcopy
import numpy as np
import pytest
import torch
from signrepr.data.schema import PART_INDICES, ROOT_LOCAL_INDEX, validate_pose
from signrepr.data.fixture import fixture_pose, make_fixture
from signrepr.data.normalization import canonicalize_sample, select_frames
from signrepr.data.pose_dataset import PoseDataset


def test_mapping_root_and_confidence_not_depth():
    assert PART_INDICES['body'] == [0, 3, 4, 5, 6, 7, 8, 9, 10]
    assert PART_INDICES['face'] == [23, 25, 27, 29, 31, 33, 35, 37, 39, 83, 84, 85, 86, 87, 88, 89, 90, 53]
    p = fixture_pose()
    p['xy'][:] = np.arange(133)[None, :, None]
    p['xy'][:, 5], p['xy'][:, 6] = [4, 0], [6, 0]
    a = canonicalize_sample(p)
    q = deepcopy(p); q['confidence'][:] = .8
    b = canonicalize_sample(q)
    for part, ids in PART_INDICES.items():
        torch.testing.assert_close(a['parts'][part]['xy'], b['parts'][part]['xy'])
        assert a['parts'][part]['xy'].shape[-2:] == (len(ids), 2)
        if part in ROOT_LOCAL_INDEX:
            assert (a['parts'][part]['xy'][:, ROOT_LOCAL_INDEX[part]] == 0).all()
        expected = (torch.tensor(p['xy'][:, ids])-torch.tensor([5, 0]))/2
        torch.testing.assert_close(a['parts'][part]['global_xy'], expected)


def test_fixed_reference_translation_scale():
    p = fixture_pose()
    q = deepcopy(p); q['xy'] = q['xy']*3+np.array([2, 4], np.float32)
    a, b = canonicalize_sample(p), canonicalize_sample(q)
    for part in PART_INDICES:
        torch.testing.assert_close(a['parts'][part]['xy'], b['parts'][part]['xy'], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a['parts'][part]['anchor'], b['parts'][part]['anchor'], atol=1e-6, rtol=1e-5)
    window = select_frames(a, torch.arange(4, 8))
    assert window['reference'] == a['reference']
    torch.testing.assert_close(window['position'], a['position'][4:8])


def test_natural_missing_unknown_timing():
    for variant in ('all_missing', 'one_part_missing', 'unknown_timing'):
        a = canonicalize_sample(fixture_pose(7, variant=variant))
        assert torch.isfinite(a['parts']['left_hand']['xy']).all()
        if variant == 'all_missing':
            assert a['status'] == 'normalization_failed'
        if variant == 'unknown_timing':
            assert not a['timing_known']
            torch.testing.assert_close(a['position'], torch.arange(7).float())
    p = fixture_pose(); p['timestamps_sec'][2] = p['timestamps_sec'][1]
    with pytest.raises(ValueError, match='timestamps'):
        validate_pose(p)


def test_ssl_dataset_does_not_return_labels(tmp_path, config):
    path = make_fixture(tmp_path)
    ds = PoseDataset(path, config['data'])
    original = ds[3]
    ds.rows[3]['gloss_id'] = 'changed'
    ds.rows[3]['translation'] = 'changed'
    changed = ds[3]
    assert not {'label', 'gloss_id', 'gloss_sequence', 'translation'} & original.keys()
    for part in PART_INDICES:
        torch.testing.assert_close(original['parts'][part]['xy'], changed['parts'][part]['xy'])
    from signrepr.data.collate import collate_pose
    from signrepr.training.pretrain import SSLTrainer
    first, second = SSLTrainer(config), SSLTrainer(config)
    a = first.update([collate_pose([original, ds[6]])])
    b = second.update([collate_pose([changed, ds[6]])])
    a.pop('update_duration_sec'); b.pop('update_duration_sec')
    assert a == b
    for key, value in first.system.state_dict().items():
        torch.testing.assert_close(value, second.system.state_dict()[key], atol=0, rtol=0)
