import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('study_driver',ROOT/'scripts/run_controlled_pilot.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_fresh_studies_cannot_share_derived_feature_destinations():
    original = driver.feature_root_for('runs/controlled_pilot','reports/controlled_pilot')
    assert original == Path('features/controlled_pilot')
    a = driver.feature_root_for('runs/new_a','reports/new_a')
    b = driver.feature_root_for('runs/new_b','reports/new_b')
    same_basename = driver.feature_root_for('runs/another/new_a','reports/another/new_a')
    assert len({original,a,b,same_basename}) == 4
    assert driver.feature_root_for('runs/new_a','reports/new_a') == a
    assert driver.feature_root_for('runs/new_a','reports/new_a','features/explicit') == Path('features/explicit')
