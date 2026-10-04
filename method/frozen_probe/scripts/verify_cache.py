"""CPU integrity gate for a completed, registered real extraction."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from cache import CacheReader, file_sha, object_sha
from run_context import RunContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--extraction-run', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume:
        raise ValueError('Register a new integrity run; completed runs are immutable')
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        directory = args.extraction_run
        status = json.loads((directory / 'status.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        environment = json.loads((directory / 'environment.json').read_text())
        if (status['status'] != 'completed' or status['exit_code'] != 0
                or status['run_id'] != config['extraction_run_id']
                or status['protocol'] != config['protocol'] or not status['GPU_hours'] > 0
                or report['run_id'] != status['run_id'] or not report['passed']
                or status['output_sha256'] != file_sha(directory / 'report.json')
                or status['config_sha256'] != config['extraction_config_sha256']
                or file_sha(directory / 'config.json') != status['config_sha256']
                or report['code_hashes'] != environment['code_hashes']):
            raise ValueError('Extraction provenance/status mismatch')
        for name, digest in environment['code_hashes'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe snapshot path')
            if file_sha(directory / 'source_snapshot' / relative) != digest:
                raise ValueError('Extraction source snapshot changed')
        identity = config['cache_identity']
        if (report['packages'] != identity['packages']
                or report['checkpoint_sha256'] != identity['checkpoint_sha256']
                or report['code_hashes']['signrep_adapter.py'] != identity['preprocessing_code_sha256']
                or report['code_hashes']['temporal.py'] != identity['temporal_code_sha256']):
            raise ValueError('Frozen feature recipe mismatch')
        cohort_path = REPO / config['cohort_manifest']
        audit_path = REPO / config['video_audit']
        if (file_sha(cohort_path) != identity['cohort_sha256']
                or file_sha(audit_path) != identity['audit_sha256']):
            raise ValueError('Locked input manifests changed')
        cohort = json.loads(cohort_path.read_text())
        audit = json.loads(audit_path.read_text())
        manifest_path = Path(report['cache_manifest'])
        if file_sha(manifest_path) != report['cache_manifest_sha256']:
            raise ValueError('Completed cache manifest changed')
        reader = CacheReader(manifest_path, identity, cohort, audit)
        durable = json.loads((directory / 'cache_manifest.json').read_text())
        if durable != reader.manifest | {'storage_path': str(manifest_path.parent)}:
            raise ValueError('Durable and RAM manifests disagree')
        windows = 0
        for key in sorted(reader.entries):
            features, activity, centers = reader.load(key)
            windows += len(centers)
        if (len(reader.entries) != 1691 or windows != 61339
                or report['source_count'] != 1691 or report['windows'] != windows
                or report['payload_bytes'] != reader.manifest['payload_bytes']
                or report['test_access'] != 'none'):
            raise ValueError('Incomplete registered cohort')
        run.save('report.json', dict(run_id=args.run_id, passed=True,
                 scope='real cache integrity only; no task score', source_count=1691,
                 windows=windows, payload_bytes=reader.manifest['payload_bytes'],
                 cache_manifest=str(manifest_path), cache_manifest_sha256=file_sha(manifest_path),
                 cache_identity_sha256=object_sha(identity),
                 extraction_report_sha256=file_sha(directory / 'report.json'),
                 extraction_status_sha256=file_sha(directory / 'status.json'),
                 all_shard_hashes_arrays_and_native_centers_checked=True, test_access='none'))
        run.status['output_sha256'] = file_sha(run.output / 'report.json')


if __name__ == '__main__':
    main()
