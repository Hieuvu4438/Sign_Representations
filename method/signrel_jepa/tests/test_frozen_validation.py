import json
import random
import numpy as np
import pytest
import torch

from signrel.eval import frozen as f

def test_frozen_eval_rejects_test_selection(tmp_path):
    manifest = tmp_path/'test.jsonl'
    manifest.write_text(json.dumps({'sample_id':'x','split':'test'})+'\n')
    with pytest.raises(ValueError,match='validation'):
        f.validate_evaluation({'query_manifest':str(manifest),'gallery_manifest':'g','config':'c'})
    manifest.write_text(json.dumps({'sample_id':'x','split':'val'})+'\n')
    assert f.validate_evaluation({'query_manifest':str(manifest),'gallery_manifest':'g','config':'c'}) == ['g',str(manifest)]


def test_periodic_eval_preserves_all_random_streams_on_error():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    expected = (random.random(),np.random.random(),torch.rand(3))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    with pytest.raises(RuntimeError):
        with f.preserved_rng():
            random.random(); np.random.random(); torch.rand(3)
            raise RuntimeError('failed evaluation')
    got = (random.random(),np.random.random(),torch.rand(3))
    assert expected[:2] == got[:2]
    assert torch.equal(expected[2],got[2])


def test_encoder_hash_depends_on_weights_and_selection_is_finite():
    model = torch.nn.Linear(3,2)
    before = f.tensor_hash(model)
    with torch.no_grad():
        model.weight[0,0] += 1
    assert before != f.tensor_hash(model)
    assert f.selection_value({'retrieval':{'Recall@1':.4}}) == .4
    with pytest.raises(ValueError,match='finite'):
        f.selection_value({'retrieval':{'Recall@1':float('nan')}})
