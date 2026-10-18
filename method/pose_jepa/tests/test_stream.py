import torch
from signrepr.data.fixture import make_fixture
from signrepr.data.pose_dataset import PoseDataset
from signrepr.training.stream import SampleStream


def test_prefetched_worker_cursor_resume_and_model_rng_independence(config, tmp_path):
    path = make_fixture(tmp_path, 16)
    cfg = config['data'].copy(); cfg['max_frames'] = 4
    ds = PoseDataset(path, cfg, sampling='continuous')
    stream = SampleStream(ds, 2, 43, num_workers=2)
    cpu_rng = torch.get_rng_state().clone()
    stream.next(); state = stream.state_dict()
    expected = [stream.next() for _ in range(3)]
    assert torch.equal(cpu_rng, torch.get_rng_state())
    restored = SampleStream(ds, 2, 43, num_workers=0)
    restored.load_state_dict(state)
    for reference in expected:
        actual = restored.next()
        assert reference['sample_ids'] == actual['sample_ids']
        torch.testing.assert_close(reference['frame_indices'], actual['frame_indices'], atol=0, rtol=0)
        for part in reference['parts']:
            torch.testing.assert_close(reference['parts'][part]['xy'], actual['parts'][part]['xy'], atol=0, rtol=0)
    stream.close(); restored.close()


def test_rank_shards_do_not_drift_between_epochs(config, tmp_path):
    path = make_fixture(tmp_path, 17)
    ds = PoseDataset(path, config['data'])
    a, b = SampleStream(ds, 2, 43, rank=0, world_size=2), SampleStream(ds, 2, 43, rank=1, world_size=2)
    for _ in range(12):
        x, y = a.next(), b.next()
        assert not set(x['sample_ids']) & set(y['sample_ids'])
        assert a.state_dict()['epoch'] == b.state_dict()['epoch']
    a.close(); b.close()
