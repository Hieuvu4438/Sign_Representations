from signrel.phase_state import PhaseState


def test_receipts_invalidate_changed_dependency_and_artifact(tmp_path):
    artifact=tmp_path/'evidence';artifact.write_text('actual')
    state=PhaseState(tmp_path/'phases.json')
    state.record(0,{'scope':'pilot'},[artifact],'passed','environment inspected')
    state.record(1,{'code':'one'},[artifact],'passed','tests passed')
    assert state.reusable(1,{'code':'one'})
    artifact.write_text('changed')
    assert not state.reusable(1,{'code':'one'})
    state.record(0,{'scope':'new'},[artifact],'passed','new environment')
    assert state.state['phases']['1']['status']=='incomplete'
