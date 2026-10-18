import importlib.util
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

path=Path(__file__).resolve().parents[1]/'scripts/test_retrieval_job.py'
spec=importlib.util.spec_from_file_location('test_retrieval_ops',path);J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)


def test_test_projection_reuses_gallery_pca_exactly():
    rng=np.random.default_rng(4);gallery=rng.normal(size=(40,8)).astype(np.float32);test=rng.normal(size=(5,8)).astype(np.float32)
    pca=PCA(4,svd_solver='full').fit(gallery)
    expected=pca.transform(test)
    actual=J.transform_frozen(test,pca.mean_,pca.components_)
    np.testing.assert_array_equal(actual,expected)
    modified=J.transform_frozen(test+10,pca.mean_,pca.components_)
    assert not np.array_equal(actual,modified)
    np.testing.assert_array_equal(J.transform_frozen(gallery,pca.mean_,pca.components_),pca.transform(gallery))


def test_test_waits_for_validation_and_fixed_transforms(tmp_path):
    J.B.atomic_json(tmp_path/'lock.json',dict(dependency_results=str(tmp_path/'validation.json'),checkpoint_sha256='c',pca_folder=str(tmp_path/'pca')))
    assert J.prerequisite(tmp_path) is False
    J.B.atomic_json(tmp_path/'validation.json',dict(status='complete',checkpoint_sha256='c',gallery_n=40154,query_n=10304))
    (tmp_path/'pca').mkdir()
    for name in ('grpP','mw','ord2'):(tmp_path/'pca'/f'{name}_pca.npz').write_bytes(b'fixture')
    assert J.prerequisite(tmp_path) is True
