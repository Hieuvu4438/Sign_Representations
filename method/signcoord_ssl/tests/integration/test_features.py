from pathlib import Path
import subprocess
import sys

import numpy as np

from signcoord.config import load_config
from signcoord.engine.checkpoint import atomic_checkpoint
from signcoord.engine.train import make_model
from signcoord.eval.features import extract_features
from signcoord.provenance import sha256_file


def test_full_interval_frozen_features_without_pose_cache(tmp_path):
    project=Path(__file__).resolve().parents[2]
    fixture=tmp_path/'fixture'
    subprocess.run([sys.executable,str(project/'tools/generate_synthetic_fixture.py'),'--output',str(fixture)],check=True,capture_output=True)
    c=load_config(fixture/'fixture_smoke.yaml'); model=make_model(c).eval()
    checkpoint=tmp_path/'checkpoint.pt'
    atomic_checkpoint(checkpoint,{'schema_version':'signcoord_checkpoint_v1','microstep_boundary':0,
                                 'model_state':model.state_dict(),'resolved_config':c})
    before=sha256_file(checkpoint)
    # The data fixture keeps labeled eval metadata outside the encoder input.
    (fixture/'pose').rename(fixture/'pose_unavailable')
    report=extract_features(checkpoint,fixture/'manifests/ssl_dev.jsonl',tmp_path/'features.npz')
    assert sha256_file(checkpoint)==before and not report['labels_read_by_encoder']
    with np.load(tmp_path/'features.npz',allow_pickle=False) as cache:
        assert cache['z'].shape==(8,32)
        assert np.allclose(np.linalg.norm(cache['z'],axis=1),1,atol=1e-6)
        # 40 source frames -> 20 native timestamps across both windows, instead
        # of a single center window dropping onset/offset.
        assert cache['offsets'][1]-cache['offsets'][0]==20
        assert np.all(np.diff(cache['sequence_times'][:20])>.039)
