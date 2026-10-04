"""Bounded metadata-fixed native eligibility and cost; no model or task quality."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from msasl_native_scan_v2 import scan_interval, ffprobe_interval, file_digest, EligibilityIssue
from resources import scratch_preflight


def main():
    parser = argparse.ArgumentParser()
    for key in ('config', 'run-id', 'output-dir'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    with BoundedCPUContext(args.output_dir, args.run_id, args.config, args.seed) as run:
        c = run.config
        cv2.setNumThreads(1)
        if c.get('software_only'):
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(
                unittest.defaultTestLoader.discover('method/frozen_probe/tests', pattern='test_msasl_native_scan_v2.py'))
            run.save('native_scan_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
                                                   errors=len(result.errors), failures=len(result.failures)))
            if not result.wasSuccessful():
                raise ValueError('Streaming native oracles failed')
            return
        resource = scratch_preflight('/dev/shm', 256 << 20)
        inventory = json.loads(Path(c['candidate_records_path']).read_text())['candidates']
        inventory += c.get('reference_control_samples', [])
        ids = c.get('selected_sample_ids')
        rows = inventory if ids is None else [next(x for x in inventory if x['sample_id']==key) for key in ids]
        assert len({x['sample_id'] for x in rows}) == len(rows)
        annotations = {s:json.loads(Path(c['dataset_root'], f'MSASL_{s}.json').read_text()) for s in ('train','val')}
        parents, records = {}, []
        for index, item in enumerate(rows):
            started = time.monotonic()
            key, path = item['sample_id'], Path(item['raw_path'])
            out = dict(sample_id=key, split=item['split'], class_id=item['class_id'], signer_id=item['signer_id'],
                       parent_id=item['parent_id'], annotation_index=item['annotation_index'])
            try:
                if item['parent_id'] not in parents:
                    parents[item['parent_id']] = file_digest(path, c['source_byte_limit'])
                identity = parents[item['parent_id']]
                stat = path.stat()
                if stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                    raise EligibilityIssue('source_changed_after_hash')
                if identity['bytes'] != item['raw_candidate_bytes']:
                    raise EligibilityIssue('source_size_changed_since_metadata')
                row = annotations[item['split']][item['annotation_index']]
                native, ticks = scan_interval(path, row, c['max_selected_frames'], c['max_decoded_frames'])
                ref = ffprobe_interval(path, row, native, ticks, c['reference_timeout_seconds'], c['reference_byte_limit'])
                stat = path.stat()
                if stat.st_size != identity['bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                    raise EligibilityIssue('source_changed_during_scan')
                out.update(status='eligible' if native['native16_eligible'] else 'native_short',
                           reason=None if native['native16_eligible'] else 'fewer_than_16_native_frames',
                           source_sha256=identity['sha256'], native=native, independent_reference=ref)
            except EligibilityIssue as e:
                out.update(status='incomplete', reason=e.reason)
            except TimeoutError:
                # The context deadline must stop the job, never become a row exclusion.
                raise
            except (av.FFmpegError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, KeyError, ValueError) as e:
                out.update(status='incomplete', reason=type(e).__name__, error=str(e)[:512])
            out['wall_seconds'] = time.monotonic()-started
            records.append(out)
            if (index+1) % c['progress_every_rows'] == 0 or index+1 == len(rows):
                print(json.dumps(dict(progress_rows=index+1, expected_rows=len(rows),
                                      statuses=dict(Counter(x['status'] for x in records)))), flush=True)
                run.status.update(progress_rows=index+1, expected_rows=len(rows),
                                  row_status_counts=dict(Counter(x['status'] for x in records)))
                run.store.write_json('status.json', run.status)
        status_counts = Counter(x['status'] for x in records)
        classes = {split:Counter(x['class_id'] for x in records if x['split']==split and x['status']=='eligible')
                   for split in ('train','val')}
        old = c.get('prior_native_runtime')
        if old:
            prior = json.loads(Path(old, 'native_audit.json').read_text())
            for oldrow in prior['samples']:
                match = next(x for x in records if x['sample_id']==c['prior_id_mapping'][oldrow['sample_id']])
                if match['status'] != 'eligible':
                    continue
                arr = np.load(Path(old, oldrow['sample_id']+'_first_native_window.npy'), allow_pickle=False)
                oldpts = np.load(Path(old, oldrow['sample_id']+'_native_pts.npy'), allow_pickle=False)
                same = (hashlib.sha256(arr.tobytes()).hexdigest()==match['native']['first_native16_RGB_sha256']
                        and match['native']['selected_native_frames']==len(oldpts)
                        and match['native']['first_selected_relative_pts']==float(oldpts[0])
                        and match['native']['last_selected_relative_pts']==float(oldpts[-1]))
                match['old_native_reference_equal'] = same
                if not same:
                    match.update(status='incomplete', reason='old_four_native_reference_mismatch')
        status_counts = Counter(x['status'] for x in records)
        classes = {split:Counter(x['class_id'] for x in records if x['split']==split and x['status']=='eligible'
                                and x['class_id'] < 100) for split in ('train','val')}
        run.save('source_parents.json', parents)
        run.save('native_scan_records.json', records)
        summary = dict(expected_rows=len(rows), audited_rows=len(records), status_counts=dict(status_counts),
                       reason_counts=dict(Counter(x['reason'] for x in records if x['reason'])),
                       splits={s:dict(eligible_rows=sum(cs.values()), classes=len(cs),
                                      class_counts={str(k):cs[k] for k in range(100)},
                                      zero_eligible_classes=sorted(set(range(100))-set(cs))) for s,cs in classes.items()},
                       reference_control_rows=len(c.get('reference_control_samples', [])),
                       source_parents=len(parents), unique_source_hashed_bytes=sum(x['bytes'] for x in parents.values()),
                       selected_native_frames=sum(x.get('native',{}).get('selected_native_frames',0) for x in records),
                       complete_stride2_windows=sum(x.get('native',{}).get('complete_stride2_windows',0) for x in records),
                       captured_row_wall_seconds=sum(x['wall_seconds'] for x in records), resource=resource,
                       versions=dict(av=av.__version__, cv2=cv2.__version__, numpy=np.__version__, av_libraries=av.library_versions,
                                     ffprobe=subprocess.run(['ffprobe','-version'],check=True,capture_output=True,text=True,timeout=10).stdout.splitlines()[0]),
                       full_candidate_coverage=ids is None, no_incomplete_rows=status_counts.get('incomplete',0)==0,
                       all100classes_eligible=all(len(cs)==100 for cs in classes.values()),
                       GPU_hours=0, new_encoder_calls=0, gallery_selection='none', quality_access='none; train/val technical pixels only',
                       interpretation='Adapted eligibility only; native-short rows explicitly recorded, no padding. Incomplete resource/reference/decoder rows block promotion. Pixel crop correctness previously checked on4sources, full output hashes are replay joins, not linguistic/YouTube authentication. Public pool absence not checkpoint training exclusion.')
        run.save('native_scan_summary.json', summary)
        if status_counts.get('incomplete',0):
            raise ValueError('Native eligibility incomplete; retain records and block cohort promotion')


if __name__ == '__main__':
    main()
