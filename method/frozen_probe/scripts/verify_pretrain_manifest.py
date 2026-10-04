"""Independent saved-source recount; no network, encoder or task scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache import file_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    for option in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        c = run.config
        for path, digest in c['input_hashes'].items():
            assert file_sha(path) == digest, path
        summaries = []
        for directory in c['parents']:
            root = Path(directory)
            status = json.loads((root / 'status.json').read_text())
            assert status['status'] in ('completed', 'failed')
            assert status['GPU_hours'] == 0
            assert file_sha(root / 'config.json') == status['config_sha256']
            config = json.loads((root / 'config.json').read_text())
            for path, digest in config['input_hashes'].items():
                assert file_sha(path) == digest, path
            environment = json.loads((root / 'environment.json').read_text())
            for relative, digest in environment['code_hashes'].items():
                assert file_sha(root / 'source_snapshot' / relative) == digest
            for relative, digest in status.get('output_hashes', {}).items():
                assert file_sha(root / relative) == digest
            actual_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            registration = json.loads(Path(c['registrations'][status['run_id']]).read_text())
            assert actual_bytes <= registration['durable_bytes_ceiling']
            summaries.append(dict(run_id=status['run_id'], status=status['status'],
                                  wall_seconds=status['wall_seconds'], durable_bytes=actual_bytes,
                                  ceiling_bytes=registration['durable_bytes_ceiling']))
        first, failed, successful = map(Path, c['parents'])
        assert summaries[1]['status'] == 'failed'
        assert not (failed / 'manifest_audit.json').exists()
        assert 'Unrecognized source-ID field' in (failed / 'error.txt').read_text()
        raw = (successful / 'youtube-sl-25-metadata.csv').read_bytes()
        assert raw == (failed / 'youtube-sl-25-metadata.csv').read_bytes()
        # Independent line splitting is safe only for this verified unquoted ASCII schema.
        assert b'"' not in raw and raw.isascii()
        lines = raw.decode('ascii').splitlines()
        pairs = [line.split(',') for line in lines]
        assert all(len(p) == 2 for p in pairs)
        ids = [p[0] for p in pairs]
        assert all(re.fullmatch(r'[A-Za-z0-9_-]{11}', v) for v in ids)
        local = set(Path(c['local_ids']).read_text().splitlines())
        report = json.loads((successful / 'manifest_audit.json').read_text())
        assert ids[0] == 'Bdj5MUf_3Hc'
        assert len(ids) == len(set(ids)) == report['row_count'] == report['unique_source_ids'] == 39197
        assert len(local) == report['local_YouTube_ASL_unique_ids'] == 11096
        assert len(local & set(ids)) == report['local_ids_present_in_public_pool'] == 11096
        assert len(local - set(ids)) == report['local_ids_absent_from_public_pool'] == 0
        assert len(set(ids) - local) == report['public_ids_absent_from_local_ASL_list'] == 28101
        assert dict(Counter(p[1] for p in pairs)) == report['other_field_literal_counts']['raw_second_column']
        assert hashlib.sha256(raw).hexdigest() == report['source_sha256']
        lineage = json.loads((first / 'lineage_audit.json').read_text())
        assert lineage['released_asset_match_verified'] and lineage['selected_ASL_train_val_rows'] == 2352
        assert not lineage['accepted_ASL_to_YouTube_source_join']
        assert not report['checkpoint_specific_filtered_training_list_available']
        assert report['ASL_Citizen_pretraining_disjointness'] == 'unknown'
        result = dict(passed=True, parent_runs=summaries, first_record_retained=ids[0],
                      source_rows=len(ids), source_unique_ids=len(set(ids)), local_ids=len(local),
                      local_ids_in_pool=len(local & set(ids)), failed_and_retry_source_bytes_equal=True,
                      all_bound_inputs_snapshots_configs_outputs_verified=True,
                      parent_instrumented_CPU_seconds=sum(s['wall_seconds'] for s in summaries),
                      GPU_hours=0, new_encoder_calls=0, scope='same-agent independent line recount and provenance verification only; no training-membership or task-quality evidence')
        run.save('verification.json', result)
        run.status['output_hashes'] = {'verification.json': file_sha(run.output / 'verification.json')}


if __name__ == '__main__':
    main()
