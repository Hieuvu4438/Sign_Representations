"""Locate every failed native-PTS pair without changing any native selection."""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

import av
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from resources import scratch_preflight


def disorder(ticks,left,right,tb,origin):
    pairs=[]
    for i,(a,b) in enumerate(zip(ticks,ticks[1:])):
        if b<=a:
            ta,tbtime=a*tb-origin,b*tb-origin
            pairs.append(dict(decoded_indices=[i,i+1],integer_pts=[a,b],kind='equal' if a==b else 'backwards',
                              relative_seconds=[float(ta),float(tbtime)],
                              both_selected=left<=ta<right and left<=tbtime<right,
                              both_before_start=ta<left and tbtime<left,
                              either_selected=(left<=ta<right or left<=tbtime<right)))
    return pairs


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    with BoundedCPUContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        c=run.config;resource=scratch_preflight('/dev/shm',256<<20)
        old=Path(c['failed_runtime']);audit=json.loads((old/'native_scan_records.json').read_text())
        failed=[x for x in audit if x['status']=='incomplete'];assert [x['sample_id'] for x in failed]==c['failed_sample_ids']
        ann=json.loads(Path(c['train_metadata']).read_text());results=[]
        for item in failed:
            row=ann[item['annotation_index']];path=Path(c['raw_root'])/(item['parent_id']+'.mp4')
            left,right=Fraction(str(row['start_time'])),Fraction(str(row['end_time']))
            ticks=[];corrupt_indices=[]
            with av.open(str(path)) as container:
                assert len(container.streams.video)==1
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                assert stream.start_time is not None and stream.time_base is not None
                timebase=stream.time_base;origin_tick=int(stream.start_time);origin=origin_tick*timebase
                container.seek(math.floor((origin+left)/timebase),stream=stream,backward=True,any_frame=False)
                for frame in container.decode(video=0):
                    if len(ticks)>=c['max_decoded_frames']:raise ValueError('Diagnostic decode bound exhausted')
                    assert frame.pts is not None and frame.time_base==timebase
                    if frame.is_corrupt:corrupt_indices.append(len(ticks))
                    ticks.append(int(frame.pts))
                    if frame.pts*timebase-origin>=right:break
            selected=[t for t in ticks if left<=t*timebase-origin<right]
            interval=f'{float(origin+left):.9f}%{float(origin+right+1):.9f}'
            cmd=['ffprobe','-v','error','-threads','1','-select_streams','v:0','-read_intervals',interval,
                 '-show_entries','stream=time_base,start_pts,width,height:frame=pkt_pts,best_effort_timestamp','-of','json',str(path)]
            proc=subprocess.run(cmd,capture_output=True,check=True,timeout=c['reference_timeout_seconds'])
            assert len(proc.stdout)<=c['reference_byte_limit'] and len(proc.stderr)<=c['reference_byte_limit']
            reference=json.loads(proc.stdout);headers=reference['streams'];assert len(headers)==1;header=headers[0]
            ref=[int(x['pkt_pts']) for x in reference['frames']]
            best=[int(x['best_effort_timestamp']) for x in reference['frames']]
            assert len(ref)<=c['max_decoded_frames']
            assert Fraction(header['time_base'])==timebase and int(header['start_pts'])==origin_tick
            refselected=[t for t in ref if left<=t*timebase-origin<right]
            pairs=disorder(ticks,left,right,timebase,origin)
            selectedpairs=disorder(selected,left,right,timebase,origin)
            results.append(dict(sample_id=item['sample_id'],parent_id=item['parent_id'],annotation_index=item['annotation_index'],
                class_id=item['class_id'],annotation_seconds=[float(left),float(right)],time_base=str(timebase),origin_pts=origin_tick,
                decoded_native_ticks=ticks,selected_native_ticks=selected,nonincreasing_pairs=pairs,
                selected_nonincreasing_pairs=selectedpairs,corrupt_decoded_indices=corrupt_indices,
                selected_native16_count=len(selected),source_sha256=c['input_hashes'][str(path)],
                independent_reference=dict(command=cmd,header=header,decoded_ticks=ref,selected_ticks=refselected,
                    nonincreasing_pairs=disorder(ref,left,right,timebase,origin),best_effort_equals_original_ticks=best==ref,
                    selected_ticks_exact=selected==refselected,response_bytes=len(proc.stdout)),
                only_preroll_disorder=bool(pairs) and all(x['both_before_start'] for x in pairs) and not selectedpairs))
        run.save('pts_diagnosis_records.json',results)
        run.save('pts_diagnosis_summary.json',dict(rows=len(results),source_parents=len({x['parent_id'] for x in results}),
            all_failed_rows_accounted=True,only_preroll_disorder_rows=sum(x['only_preroll_disorder'] for x in results),
            selected_disorder_rows=sum(bool(x['selected_nonincreasing_pairs']) for x in results),
            selected_native16_count_min=min(x['selected_native16_count'] for x in results),
            reference_selected_exact_rows=sum(x['independent_reference']['selected_ticks_exact'] for x in results),
            reference_best_effort_exact_rows=sum(x['independent_reference']['best_effort_equals_original_ticks'] for x in results),
            pair_kinds=dict(Counter(p['kind'] for x in results for p in x['nonincreasing_pairs'])),resource=resource,
            GPU_hours=0,new_encoder_calls=0,gallery_selection='none',quality_access='none; train technical timestamps only',
            interpretation='Observability only, no frame order/selection/timestamp correction or cohort promotion. Subsequent fix/availability decision requires prospective protocol and full failed-row revalidation.'))


if __name__=='__main__':main()
