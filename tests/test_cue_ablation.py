import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from signrepr.cue_ablation import CONDITIONS, STREAMS, intervene, pool_conditions
from run_ncslgr_branch_ablation import prepare
import continue_ncslgr_branch_ablation as continuation
from signrepr.io import sha256


class CueAblationTests(unittest.TestCase):
    def test_intervention_preserves_input_and_separates_hand_pair(self):
        streams={name:np.ones((3,2),dtype=np.float32) for name in STREAMS}
        value=intervene(streams,'without_hands')
        np.testing.assert_array_equal(value['face'],streams['face'])
        self.assertFalse(value['left_hand'].any());self.assertFalse(value['right_hand'].any())
        self.assertTrue(all(v.all() for v in streams.values()))
        value['face'][:]=0
        self.assertTrue(streams['face'].all())
        self.assertFalse(any(v.any() for v in intervene(streams,'zero_streams').values()))

    def test_all_conditions_pool_identical_source_nodes_and_keep_gaps(self):
        x=np.ones((len(CONDITIONS),4,2),dtype=np.float32)
        x[:,:,1]=np.array([0,100,2,300])
        observed=np.ones((4,4),dtype=bool);observed[1,1]=False
        original=observed.copy()
        result,keep=pool_conditions(x,observed,np.array([0,1,10,11.]),0,11)
        self.assertEqual(keep.tolist(),[True,False,True,False])
        np.testing.assert_allclose(result,np.tile([1/np.sqrt(2),1/np.sqrt(2)],(len(CONDITIONS),1)))
        np.testing.assert_array_equal(observed,original)
        with self.assertRaisesRegex(ValueError,'population lost'):
            pool_conditions(x,np.zeros_like(observed),np.arange(4),0,4)
        with self.assertRaisesRegex(ValueError,'observation mask'):
            pool_conditions(x,observed.astype(int),np.arange(4),0,4)

    def test_partial_feature_attempt_cannot_fit_or_evaluate(self):
        with patch('run_ncslgr_branch_ablation.identities',return_value={}), \
             patch('pathlib.Path.read_text',side_effect=['{"status":"PARTIAL_TRAIN_SMOKE"}', '{"status":"PARTIAL_TRAIN_SMOKE"}']):
            with self.assertRaisesRegex(ValueError,'Full intervention extraction'):
                prepare({'identity_sha256':{}})

    def test_partial_child_cannot_trigger_readouts_or_automatic_retry(self):
        with tempfile.TemporaryDirectory(prefix='cue_pipeline_unit_',dir=ROOT/'features') as folder:
            base=Path(folder);features=base/'features';features.mkdir();(base/'reports').mkdir();(base/'logs').mkdir()
            config=base/'config.json';config.write_text(json.dumps({'identity_sha256':{},'eligible_ids':['synthetic_train']}))
            report=features/'extraction_report.json';report.write_text(json.dumps({'status':'PARTIAL_TRAIN_SMOKE','success':2}))
            (features/'status.json').write_text(json.dumps({'status':'PARTIAL_TRAIN_SMOKE','success':2}))
            (base/'reports/ncslgr_branch_ablation_smoke_v3_audit.json').write_text(json.dumps({
                'status':'PASS_TWO_TRAIN_CASES_WITHIN_RUN_CUE_ABLATION','config_sha256':sha256(config),'report_sha256':sha256(report)}))
            with patch.object(continuation,'ROOT',base),patch.object(continuation,'CONFIG',config), \
                 patch.object(continuation,'FEATURES',features),patch.object(continuation,'OUTPUT',base/'output'), \
                 patch.object(continuation,'identities',return_value={}), \
                 patch.object(continuation.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as child:
                with self.assertRaisesRegex(ValueError,'Incomplete extraction'):
                    continuation.main()
                self.assertEqual(child.call_count,1)
            state=json.loads((base/'state/ncslgr_branch_ablation_v3/status.json').read_text())
            self.assertEqual(state['status'],'FAIL')
            self.assertFalse((base/'output').exists())


if __name__=='__main__':unittest.main()
