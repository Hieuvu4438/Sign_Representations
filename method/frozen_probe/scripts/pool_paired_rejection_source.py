"""Pool unchanged train roster and freeze paired dictionary IDs before fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from cache import CacheReader, file_sha
from pooling import activity_mean
from probes import normalize_rows
from paired_rejection_episodes import paired_plan, validate_paired_plan
from resources import scratch_preflight


def read(path): return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'run-id', 'output-dir'): p.add_argument('--' + name, required=True)
    p.add_argument('--seed', type=int, required=True); a = p.parse_args()
    with BoundedCPUContext(a.output_dir, a.run_id, a.config, a.seed) as run:
        c = run.config; resource = scratch_preflight('/dev/shm', 512 << 20)
        root = Path(c['metadata_runtime']); tasks = read(root / 'candidate_partitions.json')['ASLCitizen_base']
        byid = {r['sample_id']: r for r in read(root / 'metadata_rows.json')['ASLCitizen_base']}
        rows = [byid[key] for key in tasks['gallery_pool_ids']]
        assert len(rows) == 1027 and len({r['class_id'] for r in rows}) == 100 and len({r['signer'] for r in rows}) == 24
        for others in (tasks['calibration_ids'], tasks['dev_ids']):
            for field in ('sample_id', 'signer', 'source_group'):
                assert not {r[field] for r in rows} & {byid[key][field] for key in others}, field
        plan = paired_plan(rows, c['training_steps'], 16, 50, 1701)
        assert validate_paired_plan(rows, plan, 50)
        query_counts = Counter(rows[int(i)]['signer'] for i in plan[:, :, 0].reshape(-1))
        assert set(query_counts) == {r['signer'] for r in rows}
        prior = read(c['base_inputs']['identity_config']); report = read(c['base_inputs']['report'])
        assert report['passed'] and file_sha(report['cache_manifest']) == report['cache_manifest_sha256']
        reader = CacheReader(report['cache_manifest'], prior['cache_identity'], read(prior['cohort_manifest']), read(prior['video_audit']))
        values = []; windows = 0
        for row in rows:
            x, activity, _ = reader.load(row['sample_id']); assert len(x) == row['windows']
            values.append(normalize_rows(activity_mean(x, activity)[0][None])[0]); windows += len(x)
        vectors = np.stack(values)
        assert vectors.shape == (1027, 768) and vectors.dtype == np.float64 and np.isfinite(vectors).all()
        run.save_array('fit_vectors.npy', vectors, c['array_specs']['fit_vectors.npy'])
        run.save_array('training_plan.npy', plan, c['array_specs']['training_plan.npy'])
        run.save('source_roster.json', rows)
        run.save('pool_summary.json', dict(passed=True, clips=1027, classes=100, official_signer_ids=24,
            cached_windows=windows, training_episode_count=1000, paired_query_count=16000,
            query_counts_by_signer=dict(sorted(query_counts.items())), source_calibration_dev_disjoint=True,
            vector_bytes_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),
            episode_bytes_sha256=hashlib.sha256(plan.tobytes()).hexdigest(), resource=resource,
            GPU_seconds=0, new_encoder_calls=0, scope='All source fit clips/IDs retained; paired ID plan only, no learned head/heldout scores.'))


if __name__ == '__main__': main()
