import json
from pathlib import Path
import tempfile
import unittest

from cache import file_sha
from confirmatory import METHODS, SEEDS
from replication_test import check_test_freeze, validate_test_config


def config():
    return dict(protocol='P004_v1', H_id='H002', gallery_budget=1, methods=list(METHODS),
                support_seeds=list(SEEDS), dcg_loss_margin=.02, min_window_reduction=2.,
                cost_ratio_floor=.8, new_policy_trials=0, project_gpu_ceiling_seconds=14400,
                hypothesis_gpu_ceiling_seconds=7200)


class ReplicationTestFreezeTests(unittest.TestCase):
    def test_frozen_spec_cannot_promote_random_or_change_budget(self):
        validate_test_config(config())
        for field, value in [('gallery_budget', 8), ('dcg_loss_margin', .03),
                             ('hypothesis_gpu_ceiling_seconds', 10800), ('new_policy_trials', 1),
                             ('methods', ['uniform8']), ('support_seeds', [1704])]:
            with self.assertRaises(ValueError):
                validate_test_config(config() | {field: value})

    def test_prospective_gate_binds_config_source_and_positive_pilot(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            conf = repo / 'config.json'; conf.write_text(json.dumps(config()))
            source = repo / 'source.py'; source.write_text('frozen source\n')
            gate = dict(verdict='FROZEN_REPLICATION_BEFORE_SELECTED_TEST_ACCESS', protocol='P004_v1',
                        fixed_dev_pilot_passed=True, config_hashes={'config.json':file_sha(conf)},
                        file_hashes={'source.py':file_sha(source)})
            path = repo / 'gate.json'; path.write_text(json.dumps(gate))
            check_test_freeze(path, conf, repo)
            path.write_text(json.dumps(gate | {'fixed_dev_pilot_passed': False}))
            with self.assertRaises(ValueError):check_test_freeze(path, conf, repo)
            path.write_text(json.dumps(gate)); source.write_text('modified\n')
            with self.assertRaises(ValueError):check_test_freeze(path, conf, repo)
            source.write_text('frozen source\n'); conf.write_text(json.dumps(config() | {'gallery_budget':8}))
            with self.assertRaises(ValueError):check_test_freeze(path, conf, repo)

if __name__ == '__main__':
    unittest.main()
