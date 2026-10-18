"""Run in the isolated oracle venv; core runtime has no pySigLib dependency."""
import numpy as np
import torch
import pytest
from pp.signature import logsig2,chen

pysiglib=pytest.importorskip('pysiglib')

@pytest.mark.parametrize('d',[2,3,4,8,9])
def test_logsig2_pysiglib_lyndon(d):
    rng=np.random.default_rng(17+d)
    x=rng.normal(size=(7,32,d)).cumsum(1).astype(np.float64)
    pysiglib.prepare_log_sig(d,2,method=1)
    oracle=pysiglib.log_sig(x,2,method=1,n_jobs=1)
    actual=logsig2(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual,oracle,atol=2e-12,rtol=2e-12)


def test_gap_skipping_pysiglib_oracle():
    rng=np.random.default_rng(13);x=rng.normal(size=(25,4)).cumsum(0)
    pieces=[x[0:5],x[12:17],x[20:25]]
    logs=torch.stack([logsig2(torch.from_numpy(p)) for p in pieces])
    actual=chen(logs,torch.ones(3,dtype=torch.bool),4).numpy()
    # Oracle path translates each piece to the preceding visible endpoint;
    # it adds no increment from a hidden gap.
    joined=[pieces[0]]
    for p in pieces[1:]:joined.append(p[1:]-p[0]+joined[-1][-1])
    pysiglib.prepare_log_sig(4,2,method=1)
    expected=pysiglib.log_sig(np.concatenate(joined),2,method=1,n_jobs=1)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
