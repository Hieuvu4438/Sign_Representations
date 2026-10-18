import json
from pathlib import Path
import torch
import pytest
from pp.recovery import save_checkpoint,recover_checkpoint,quarantine_corrupt_latest


def state(step):
    return dict(config={'arm':'A1'},model={},opt={},step=step,epoch=0,cursor=step,centres={},stats={},source_hash='source',total=20,rng=torch.get_rng_state())


def test_rotation_and_corruption_fallback(tmp_path):
    save_checkpoint(tmp_path/'initial.pt',state(0))
    save_checkpoint(tmp_path/'latest.pt',state(3));save_checkpoint(tmp_path/'latest.pt',state(5))
    assert torch.load(tmp_path/'previous.pt',weights_only=False)['step']==3
    (tmp_path/'latest.pt').write_bytes(b'corrupt save')
    path,ck,errors=recover_checkpoint(tmp_path,{'arm':'A1'},'source',20)
    assert ck['step']==3 and path.name=='previous.pt' and len(errors)==1
    quarantine_corrupt_latest(tmp_path,errors);save_checkpoint(tmp_path/'latest.pt',state(4))
    assert torch.load(tmp_path/'previous.pt',weights_only=False)['step']==3
    assert json.loads((tmp_path/'latest.pt.json').read_text())['step']==4


def test_identity_mismatch_never_silently_resets(tmp_path):
    save_checkpoint(tmp_path/'latest.pt',state(3));save_checkpoint(tmp_path/'initial.pt',state(0))
    with pytest.raises(ValueError,match='mismatch'):recover_checkpoint(tmp_path,{'arm':'A2'},'source',20)
    with pytest.raises(RuntimeError,match='No valid'):recover_checkpoint(tmp_path/'empty',{'arm':'A1'},'source',20)


def test_failed_serialization_preserves_latest(tmp_path,monkeypatch):
    save_checkpoint(tmp_path/'latest.pt',state(3))
    def failure(*a,**k):raise OSError('simulated disk failure')
    monkeypatch.setattr(torch,'save',failure)
    with pytest.raises(OSError):save_checkpoint(tmp_path/'latest.pt',state(4))
    assert torch.load(tmp_path/'latest.pt',weights_only=False)['step']==3
