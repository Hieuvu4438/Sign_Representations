"""All38 dual clocks and independent RGB framehash, with no cohort correction."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from msasl_clock_audit import clock_comparison, parse_framehash
from resources import scratch_preflight


def capture(cmd, config):
    proc = subprocess.run(cmd, capture_output=True, timeout=config['reference_timeout_seconds'])
    if max(len(proc.stdout),len(proc.stderr)) > config['reference_byte_limit']:
        raise ValueError('Reference output exceeds registered per-pipe bound')
    return dict(command=cmd,returncode=proc.returncode,stdout=proc.stdout.decode('utf-8'),
                stderr=proc.stderr.decode('utf-8'),stdout_bytes=len(proc.stdout),stderr_bytes=len(proc.stderr))


def main():
    p=argparse.ArgumentParser()
    for name in ('config','run-id','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config
        if c.get('software_only'):
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
            from test_msasl_clock_audit import run_tests
            run.save('clock_tests.json',run_tests());return
        import av
        resource=scratch_preflight('/dev/shm',256<<20)
        previous=json.loads((Path(c['diagnostic_runtime'])/'pts_diagnosis_records.json').read_text())
        assert [x['sample_id'] for x in previous]==c['sample_ids']
        ann=json.loads(Path(c['train_metadata']).read_text());results=[]
        versions={name:capture([c[name],'-version'],c) for name in ('ffprobe','ffmpeg')}
        for item in previous:
            row=ann[item['annotation_index']];path=Path(c['raw_root'])/(item['parent_id']+'.mp4')
            assert path.stat().st_size <= c['parent_byte_limit']
            tb=Fraction(item['time_base']);origin_tick=item['origin_pts'];origin=origin_tick*tb
            left,right=Fraction(str(row['start_time'])),Fraction(str(row['end_time']))
            interval=f'{float(origin+left):.9f}%{float(origin+right+1):.9f}'
            cmd=[c['ffprobe'],'-v','error','-threads','1','-select_streams','v:0','-read_intervals',interval,
                 '-show_entries','stream=time_base,start_pts,width,height:frame=pkt_pts,pkt_dts,best_effort_timestamp,pkt_pos,key_frame,pict_type,coded_picture_number',
                 '-of','json',str(path)]
            response=capture(cmd,c)
            if response['returncode'] != 0:raise ValueError('ffprobe failed: '+response['stderr'])
            ref=json.loads(response['stdout']);header=ref['streams'][0];frames=ref['frames']
            assert len(ref['streams'])==1 and 0<len(frames)<=c['max_decoded_frames']
            assert Fraction(header['time_base'])==tb and int(header['start_pts'])==origin_tick
            raw=[int(x['pkt_pts']) for x in frames];best=[int(x['best_effort_timestamp']) for x in frames]
            oldref=item['independent_reference']['decoded_ticks'];assert raw==oldref
            comparison=clock_comparison(raw,best,str(tb),origin_tick,row['start_time'],row['end_time'])
            native=[];pixels=[];corrupt=[]
            with av.open(str(path)) as container:
                assert len(container.streams.video)==1
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                assert stream.time_base==tb and stream.start_time==origin_tick
                container.seek(math.floor((origin+left)/tb),stream=stream,backward=True,any_frame=False)
                for frame in container.decode(video=0):
                    assert frame.pts is not None and frame.time_base==tb
                    if frame.is_corrupt:corrupt.append(len(native))
                    native.append(int(frame.pts))
                    rgb=frame.to_ndarray(format='rgb24')
                    assert rgb.shape==(header['height'],header['width'],3) and rgb.nbytes<=c['maximum_frame_bytes']
                    pixels.append(hashlib.sha256(rgb).hexdigest());del rgb
                    if len(native)==len(raw):break
            prefix=item['decoded_native_ticks']
            joins=dict(native_full_raw_exact=native==raw,old_native_prefix_exact=native[:len(prefix)]==prefix)
            # Absolute seek with keyframe preroll, no frame rate conversion; raw RGB hashes only.
            fcmd=[c['ffmpeg'],'-v','error','-nostdin','-threads','1','-filter_threads','1','-copyts',
                  '-seek_timestamp','1','-ss',f'{float(origin+left):.9f}','-noaccurate_seek','-i',str(path),
                  '-map','0:v:0','-an','-sn','-dn','-frames:v',str(len(raw)),'-vsync','0',
                  '-c:v','rawvideo','-threads','1','-pix_fmt','rgb24','-enc_time_base',str(tb),
                  '-f','framehash','-hash','sha256','pipe:1']
            ff=capture(fcmd,c);parsed=None;parse_error=None
            try:parsed=parse_framehash(ff['stdout'])
            except ValueError as e:parse_error=str(e)
            joins.update(ffmpeg_exit_zero=ff['returncode']==0,ffmpeg_parse_error=parse_error,
                         ffmpeg_count_exact=False,ffmpeg_rgb_hashes_exact=False,ffmpeg_best_clock_exact=False)
            if parsed is not None:
                hashes=parsed['frames'];otb=Fraction(parsed['time_base'])
                joins.update(ffmpeg_count_exact=len(hashes)==len(raw),
                             ffmpeg_rgb_hashes_exact=[x['sha256'] for x in hashes]==pixels,
                             ffmpeg_best_clock_exact=[x['pts']*otb for x in hashes]==[t*tb for t in best],
                             ffmpeg_packet_sizes_exact=all(x['bytes']==header['height']*header['width']*3 for x in hashes))
            results.append(dict(sample_id=item['sample_id'],parent_id=item['parent_id'],annotation_index=item['annotation_index'],
                source_sha256=c['input_hashes'][str(path)],class_id=item['class_id'],annotation_seconds=[str(left),str(right)],
                time_base=str(tb),origin_pts=origin_tick,header=header,reference_frames=frames,raw_ticks=raw,best_effort_ticks=best,
                native_ticks=native,native_rgb_sha256=pixels,corrupt_indices=corrupt,comparison=comparison,joins=joins,
                ffprobe=response,ffmpeg=ff,ffmpeg_framehash=parsed))
            print(json.dumps(dict(sample_id=item['sample_id'],completed=len(results),joins=joins)),flush=True)
        run.save('clock_records.json',results)
        required=('native_full_raw_exact','old_native_prefix_exact','ffmpeg_exit_zero','ffmpeg_count_exact',
                  'ffmpeg_rgb_hashes_exact','ffmpeg_best_clock_exact','ffmpeg_packet_sizes_exact')
        run.save('clock_summary.json',dict(rows=len(results),parents=len({x['parent_id'] for x in results}),
            all_registered_rows_accounted=[x['sample_id'] for x in results]==c['sample_ids'],
            join_pass_counts={k:sum(x['joins'].get(k,False) for x in results) for k in required},
            all_frame_correspondence_gates_pass=all(all(x['joins'].get(k,False) for k in required) and not x['corrupt_indices'] for x in results),
            best_clock_monotone_rows=sum(not x['comparison']['best_nonincreasing_indices'] for x in results),
            best_selected_monotone_rows=sum(not x['comparison']['selected_best_nonincreasing_indices'] for x in results),
            changed_membership_rows=sum(not x['comparison']['selected_indices_identical'] for x in results),
            changed_tick_rows=sum(bool(x['comparison']['changed_tick_indices']) for x in results),
            added_frames=sum(len(x['comparison']['added_by_best_indices']) for x in results),
            removed_frames=sum(len(x['comparison']['removed_by_best_indices']) for x in results),
            selected_raw_frames=sum(len(x['comparison']['raw_selected_indices']) for x in results),
            selected_best_frames=sum(len(x['comparison']['best_selected_indices']) for x in results),
            maximum_absolute_clock_delta_seconds=max(x['comparison']['maximum_absolute_clock_delta_seconds'] for x in results),
            versions=versions,pyav_version=av.__version__,pyav_library_versions=av.library_versions,resource=resource,
            GPU_hours=0,new_encoder_calls=0,quality_access='none',gallery_sampling='none',
            interpretation='Diagnostic only. Best effort is estimated, not ground truth; all original failures remain. No adapter/cohort promotion regardless of observed parity.'))


if __name__=='__main__':main()
