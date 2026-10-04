"""Retain temporal oracles plus an independent noncontiguous RGB-byte oracle."""
import hashlib
import numpy as np
from test_msasl_clock_audit import run_tests as temporal_tests
from rgb_payload_hash import rgb_sha256


def run_tests():
    out=temporal_tests()
    backing=np.arange(24,dtype=np.uint8).reshape(2,4,3)
    sliced=backing[:,:2,:]
    assert not sliced.flags.c_contiguous
    expected=bytes([0,1,2,3,4,5,12,13,14,15,16,17])
    digest=hashlib.sha256(expected).hexdigest()
    assert rgb_sha256(sliced)==digest
    assert rgb_sha256(sliced.copy())==digest
    for bad in (np.zeros((2,2,4),dtype=np.uint8),np.zeros((2,2,3),dtype=np.float32)):
        try:rgb_sha256(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid RGB payload accepted')
    out['groups'].append('noncontiguous_RGB_logical_byte_oracle_and_dtype_shape_guards')
    return out
