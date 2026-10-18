from copy import deepcopy
import json
from signrepr.data.fixture import make_fixture
from signrepr.data.schema import read_manifest, write_manifest
from signrepr.data.splits import build_protocol, validate_manifest


def test_protocol_protects_target_corpus_and_duplicates(tmp_path):
    manifest = make_fixture(tmp_path/'source', 8)
    rows = read_manifest(manifest)
    target = deepcopy(rows[0]); target.update(corpus='target', split='test', sample_id='target/test/1')
    rows.append(target)
    rows[1]['source_url'] = 'https://example.invalid/shared_video'
    target['source_url'] = 'https://example.invalid/shared_video'
    write_manifest(manifest, rows)
    result = build_protocol({'protocol': 'cross_corpus_transfer', 'manifest': str(manifest), 'target_corpora': ['target'],
                             'model_selection_scope': 'source_validation'}, tmp_path/'protocol')
    selected = read_manifest(tmp_path/'protocol/ssl_train.jsonl')
    assert len(selected) == 6
    assert not {rows[0]['sample_id'], rows[1]['sample_id']} & {r['sample_id'] for r in selected}
    assert result['model_selection_scope'] == 'source_validation'
    report = validate_manifest(manifest)
    assert not report['valid']
    assert any(e['reason'] == 'exact_pose_duplicate' for e in report['errors'])


def test_malformed_manifest_generates_audit_instead_of_skipping_report(tmp_path):
    path = tmp_path/'bad.jsonl'
    path.write_text('{invalid json}\n[]\n{"sample_id":"missing_fields"}\n')
    report = validate_manifest(path)
    assert not report['valid'] and len(report['errors']) == 3
    assert 'Missing manifest fields' in report['errors'][-1]['reason']
