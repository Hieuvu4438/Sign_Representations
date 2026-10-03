import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from audit_ncslgr_branch_controls import ready,validate_selection


class CueControlAuditTests(unittest.TestCase):
    def test_first_validation_tie_and_test_selection_guards(self):
        candidates=[dict(lr=lr,validation=dict(macro_recall=score)) for lr,score in [(0.01,.7),(.03,.7),(.1,.4)]]
        selected=dict(test_used_for_selection=False,candidates=candidates,selected_lr=.01,validation=candidates[0]['validation'])
        policy=dict(lr_grid=[.01,.03,.1]);validate_selection(selected,policy)
        with self.assertRaisesRegex(ValueError,'First validation winner'):
            validate_selection({**selected,'selected_lr':.03},policy)
        with self.assertRaisesRegex(ValueError,'Test used'):
            validate_selection({**selected,'test_used_for_selection':True},policy)
        with self.assertRaisesRegex(ValueError,'grid differs'):
            validate_selection({**selected,'candidates':candidates[:1]},policy)

    def test_failed_or_unheld_pipeline_is_not_treated_as_live_wait(self):
        with patch('audit_ncslgr_branch_controls.read',return_value=dict(status='FAIL')):
            with self.assertRaisesRegex(ValueError,'cannot retry'):
                ready()
        with tempfile.TemporaryDirectory(prefix='cue_audit_unit_',dir=ROOT/'features') as folder, \
             patch('audit_ncslgr_branch_controls.PIPELINE',Path(folder)), \
             patch('audit_ncslgr_branch_controls.read',return_value=dict(status='RUNNING_CPU_INTERVENTION_EXTRACTION')):
            (Path(folder)/'.pipeline.lock').touch()
            with self.assertRaisesRegex(ValueError,'no held worker lock'):
                ready()


if __name__=='__main__':unittest.main()
