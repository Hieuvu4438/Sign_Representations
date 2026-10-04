"""Check estimated-clock inputs in actual SignRep environment before GPU work."""
import argparse
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from msasl_estimated_clock import reference_interval,scan_estimated_interval
from resources import scratch_preflight


def main():
    p=argparse.ArgumentParser()
    for k in ('config','run-id','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;cv2.setNumThreads(1);resource=scratch_preflight('/dev/shm',256<<20)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
        from test_msasl_estimated_clock import run_tests
        tests=run_tests();run.save('estimated_clock_tests.json',tests)
        allrows=json.loads(Path(c['cohort_runtime'],'adapted_cohort_records.json').read_text());byid={x['sample_id']:x for x in allrows}
        clocks=json.loads(Path(c['clock_runtime'],'clock_records.json').read_text());clockby={x['sample_id']:x for x in clocks}
        candidates=json.loads(Path(c['candidate_records_path']).read_text())['candidates'];candidates={x['sample_id']:x for x in candidates}
        annotations={s:json.loads(Path(c['dataset_root'],f'MSASL_{s}.json').read_text()) for s in ('train','val')};results=[]
        keys=('selected_native_frames','complete_stride2_windows','native16_eligible','first_selected_relative_pts',
              'last_selected_relative_pts','crop_yxyx_pixels','all_resized_RGB_sha256','first_native16_RGB_sha256','raw_pts_ticks_sha256')
        for id in c['sample_ids']:
            prior=byid[id];native=prior['native'];row=annotations[prior['split']][prior['annotation_index']];path=candidates[id]['raw_path']
            ref=reference_interval(path,row,native['time_base'],native['origin_pts'],c['ffprobe'])
            oldclock=clockby.get(id);rgb=None if oldclock is None else oldclock['native_rgb_sha256']
            current=scan_estimated_interval(path,row,ref,expected_rgb_sha256=rgb)
            comparisons={k:json.dumps(current[k])==json.dumps(native[k]) for k in keys}
            comparisons['estimated_clock_hash']=current['estimated_pts_ticks_sha256']==native['clock_pts_ticks_sha256']
            results.append(dict(sample_id=id,source_sha256=prior['source_sha256'],comparisons=comparisons,
                                passed=all(comparisons.values()),native_frames=current['selected_native_frames']))
        summary=dict(rows=len(results),passed=all(x['passed'] for x in results),software_tests=tests,records=results,
                     versions=dict(av=av.__version__,av_libraries=av.library_versions,cv2=cv2.__version__,numpy=np.__version__),
                     resource=resource,GPU_hours=0,new_encoder_calls=0,quality_access='none',
                     scope='All38 affected plus deterministic firsteligible train/val and all3short inputparity in SignRep environment; no wholecohort newdecode or taskutility')
        run.save('inference_input_parity.json',summary)
        if not summary['passed']:raise ValueError('Inference-environment input parity failed; preserve reports')


if __name__=='__main__':main()
