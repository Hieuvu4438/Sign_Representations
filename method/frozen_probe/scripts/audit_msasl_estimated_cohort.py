"""Prospective all3425 estimated-clock contract, revalidating all38 affected rows."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from msasl_estimated_clock import reference_interval,scan_estimated_interval
from resources import scratch_preflight


def main():
    p=argparse.ArgumentParser()
    for key in ('config','run-id','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;cv2.setNumThreads(1)
        if c.get('software_only'):
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
            from test_msasl_estimated_clock import run_tests
            run.save('estimated_clock_tests.json',run_tests());return
        resource=scratch_preflight('/dev/shm',256<<20)
        oldroot=Path(c['native_runtime']);old=json.loads((oldroot/'native_scan_records.json').read_text())
        parents=json.loads((oldroot/'source_parents.json').read_text());source_checks={}
        candidates=json.loads(Path(c['candidate_records_path']).read_text())['candidates']
        assert [x['sample_id'] for x in candidates]==[x['sample_id'] for x in old] and len(old)==3425
        clock=json.loads((Path(c['clock_runtime'])/'clock_records.json').read_text());byid={x['sample_id']:x for x in clock}
        assert [x['sample_id'] for x in old if x['status']=='incomplete']==c['affected_ids']==[x['sample_id'] for x in clock]
        annotations={s:json.loads(Path(c['dataset_root'],f'MSASL_{s}.json').read_text()) for s in ('train','val')}
        for key,identity in parents.items():
            path=Path(c['raw_root'])/(key+'.mp4');stat=path.stat()
            assert stat.st_size==identity['bytes'] and stat.st_mtime_ns==identity['mtime_ns']
            source_checks[key]=dict(identity,stat_unchanged=True,rehashed_this_run=str(path) in c['input_hashes'],
                                   hash_provenance='original fullscan; sixaffected parents rehashed by run context')
        records=[];changed=[]
        for previous,item in zip(old,candidates):
            started=time.monotonic();record=dict(previous)
            for key in ('sample_id','split','class_id','signer_id','parent_id','annotation_index'):assert record[key]==item[key]
            path=Path(item['raw_path']);row=annotations[item['split']][item['annotation_index']]
            if previous['status']=='incomplete':
                evidence=byid[item['sample_id']]
                ref=reference_interval(path,row,evidence['time_base'],evidence['origin_pts'],c['ffprobe'],
                                       c['reference_timeout_seconds'],c['reference_byte_limit'],c['max_decoded_frames'])
                assert ref['raw_ticks']==evidence['raw_ticks'] and ref['best_ticks']==evidence['best_effort_ticks']
                assert evidence['joins']['ffmpeg_rgb_hashes_exact'] and evidence['joins']['ffmpeg_count_exact']
                native=scan_estimated_interval(path,row,ref,c['max_selected_frames'],c['max_decoded_frames'],evidence['native_rgb_sha256'])
                h=ref['header'];box=row['box']
                manual=(math.floor(box[0]*h['height']),math.floor(box[1]*h['width']),math.ceil(box[2]*h['height']),math.ceil(box[3]*h['width']))
                assert native['crop_yxyx_pixels']==manual
                assert native['selected_decoded_ordinals']==evidence['comparison']['best_selected_indices']
                native['clock_pts_ticks_sha256']=native['estimated_pts_ticks_sha256']
                record.update(status='eligible' if native['native16_eligible'] else 'native_short',
                              reason=None if native['native16_eligible'] else 'fewer_than_16_native_frames',native=native,
                              source_sha256=evidence['source_sha256'],validation_origin='new38 estimated-clock decode/crop',
                              independent_reference=dict(command=ref['command'],response_bytes=ref['response_bytes'],
                                  selected_native_frames=native['selected_native_frames'],raw_prefix_exact=True,
                                  selected_RGB_reference='bound MSCLKDIAG002 independent FFmpeg full RGB framehash',
                                  selected_clock_reference='bound MSCLKDIAG002 FFmpeg rational selected clock; no outside-suffix gate rescue'))
                changed.append(dict(sample_id=item['sample_id'],source_sha256=record['source_sha256'],native=native,
                                    reference=ref,manual_crop_yxyx_pixels=manual,prior_clock_record_sha256=c['input_hashes'][str(Path(c['clock_runtime'])/'clock_records.json')]))
            else:
                assert previous['status'] in ('eligible','native_short') and previous['independent_reference']['integer_pts_exact']
                # Prior immutable ffprobe_interval rejects ANY raw/best inequality in the full reference.
                native=dict(previous['native'])
                native.update(clock='ffprobe best_effort_timestamp; inherited entire reference raw==best',
                              clock_pts_ticks_sha256=native['native_pts_ticks_sha256'],raw_pts_ticks_sha256=native['native_pts_ticks_sha256'])
                record.update(native=native,validation_origin='inherited3387 immutable raw==best/count/crop/RGB proof')
            record.update(input_contract=c['protocol'],prior_raw_clock_status=previous['status'],adaptation_wall_seconds=time.monotonic()-started)
            stat=path.stat();identity=parents[item['parent_id']]
            assert stat.st_size==identity['bytes'] and stat.st_mtime_ns==identity['mtime_ns']
            records.append(record)
        counts=Counter(x['status'] for x in records)
        splits={s:Counter(x['class_id'] for x in records if x['split']==s and x['status']=='eligible') for s in ('train','val')}
        summary=dict(expected_rows=3425,accounted_rows=len(records),affected_rows_redecoded=len(changed),
                     inherited_rows=len(records)-len(changed),status_counts=dict(counts),no_incomplete_rows=counts.get('incomplete',0)==0,
                     all100classes_eligible=all(set(v)==set(range(100)) for v in splits.values()),
                     splits={s:dict(eligible_rows=sum(v.values()),class_counts={str(i):v[i] for i in range(100)},
                                    min_support=min(v.values()),classes=len(v)) for s,v in splits.items()},
                     selected_native_frames=sum(x['native']['selected_native_frames'] for x in records),
                     complete_stride2_windows=sum(x['native']['complete_stride2_windows'] for x in records),
                     short_rows=[dict(sample_id=x['sample_id'],native_frames=x['native']['selected_native_frames']) for x in records if x['status']=='native_short'],
                     affected_native_frames=sum(x['native']['selected_native_frames'] for x in changed),
                     source_parents=len(source_checks),clock_policy='best_effort estimated clock for ALL candidates, retaining raw clock provenance',
                     prior_raw_clock_full_gate='failed; immutable, not rescued',GPU_hours=0,new_encoder_calls=0,gallery_selection='none',quality_access='none',
                     resource=resource,interpretation='Adapted input availability only; quality-blind prospective estimated clock, no padding/source or annotation authenticity claim. Inherited source SHAs stat-checked, six rehashed; wholecorpus not rehashed this run.')
        run.save('adapted_cohort_records.json',records);run.save('changed_scan_records.json',changed)
        run.save('source_stat_validation.json',source_checks);run.save('adapted_cohort_summary.json',summary)
        if not summary['no_incomplete_rows'] or not summary['all100classes_eligible']:raise ValueError('Adapted input cohort gate failed')


if __name__=='__main__':main()
