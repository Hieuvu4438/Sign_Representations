"""Bounded four-source CPU timestamp/crop/decoder feasibility, never quality."""
import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest

import av
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import array_spec
from msasl_native import decode_interval
from resources import scratch_preflight
from temporal import window_plan
from native_pts_artifact import native_pts_bytes


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;cv2.setNumThreads(1)
        if c.get('software_only'):
            result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(
                unittest.TestSuite([unittest.defaultTestLoader.discover('method/frozen_probe/tests',pattern='test_msasl_native.py'),
                                   unittest.defaultTestLoader.discover('method/frozen_probe/tests',pattern='test_native_pts_artifact.py')]))
            run.save('native_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
            if not result.wasSuccessful():raise ValueError('Native crop/PTS/seek oracles failed')
            return
        resource=scratch_preflight('/dev/shm',256<<20);reports=[]
        versions=dict(av=av.__version__,cv2=cv2.__version__,numpy=np.__version__,av_linked_libraries=av.library_versions)
        ffversion=subprocess.run(['ffprobe','-version'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()[0]
        for item in c['samples']:
            key=item['sample_id'];path=Path(item['path']);row=item['annotation']
            headercmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                       'stream=width,height,time_base,start_time,duration,nb_frames,avg_frame_rate,r_frame_rate:format=duration','-of','json',str(path)]
            h=json.loads(subprocess.run(headercmd,check=True,capture_output=True,text=True,timeout=15).stdout)
            assert len(h['streams'])==1;s=h['streams'][0]
            if int(s.get('nb_frames','0'))<1 or int(s['nb_frames'])>20000:raise ValueError('Reference full-source frame bound failed')
            native,pts,meta,first_raw=decode_interval(path,row,max_frames=240,max_decoded=2048)
            referencecmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]
            raw=subprocess.run(referencecmd,check=True,capture_output=True,timeout=45).stdout
            if len(raw)>2<<20:raise ValueError('Reference PTS response exceeds bound')
            ref=json.loads(raw)['frames'];assert len(ref)==int(s['nb_frames'])
            origin=Fraction(s['start_time']);alltimes=[Fraction(f['best_effort_timestamp_time'])-origin for f in ref]
            assert all(b>a for a,b in zip(alltimes,alltimes[1:]))
            # ffprobe reports decimal rounded timestamps; selection is independent
            # of audit parser and uses the same declared half-open decimal limits.
            start,end=Fraction(str(row['start_time'])),Fraction(str(row['end_time']))
            indices=[i for i,t in enumerate(alltimes) if start<=t<end]
            assert len(indices)==len(pts)
            assert max(abs(float(alltimes[i]-t)) for i,t in zip(indices,pts))<=c['pts_reference_atol_seconds']
            cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            if not cap.isOpened():raise ValueError('Independent OpenCV decoder cannot open')
            cap.set(cv2.CAP_PROP_POS_FRAMES,indices[0]);pixel_equal=0
            try:
                for j,i in enumerate(indices):
                    ok,bgr=cap.read();assert ok
                    # Direct independent yxyx coordinate arithmetic, not helper call.
                    H,W=bgr.shape[:2];import math
                    box=row['box'];y0,x0,y1,x1=(math.floor(box[0]*H),math.floor(box[1]*W),math.ceil(box[2]*H),math.ceil(box[3]*W))
                    expected=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)[y0:y1,x0:x1],(224,224),interpolation=cv2.INTER_LINEAR)
                    assert np.array_equal(expected,native[j]);pixel_equal+=1
            finally:cap.release()
            plan=window_plan([float(t) for t in pts],16,2)
            for name,array in ((f'{key}_first_native_window.npy',native[:16]),(f'{key}_native_pts.npy',np.asarray([float(t) for t in pts],dtype='<f8'))):
                if name.endswith('_native_pts.npy'):
                    run.store.write_bytes(name,native_pts_bytes(array))
                else:
                    run.save_array(name,array,array_spec(array.shape,array.dtype))
            # Two panels show raw frame and declared body crop; no linguistic label judgement.
            H,W=first_raw.shape[:2];small=cv2.resize(first_raw,(int(round(W*224/H)),224),interpolation=cv2.INTER_AREA)
            panel=np.concatenate((small,native[0]),axis=1)
            ok,encoded=cv2.imencode('.png',cv2.cvtColor(panel,cv2.COLOR_RGB2BGR));assert ok
            run.store.write_bytes(f'{key}_crop_preview.png',encoded.tobytes())
            reports.append(dict(sample_id=key,split=item['split'],annotation_index=item['annotation_index'],parent_id=item['parent_id'],
                label=row['label'],signer_id=row['signer_id'],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                annotation_seconds=[row['start_time'],row['end_time']],legacy_frame_interval=[row['start'],row['end']],
                metadata_FPS=row['fps'],decoded_dimensions=[meta['stream_width'],meta['stream_height']],
                decoded_metadata_dimensions_equal=[meta['stream_width'],meta['stream_height']]==[row['width'],row['height']],
                native=meta,reference_header=h,reference_source_frames=len(ref),reference_selected_frame_indices=indices,
                reference_PTS_match=True,independent_decoded_resized_RGB_exact_frames=pixel_equal,
                first_selected_relative_PTS_seconds=float(pts[0]),last_selected_relative_PTS_seconds=float(pts[-1]),
                complete_stride2_windows=len(plan),uncovered_tail_frames=len(pts)-plan[-1]['stop_frame_exclusive'],
                commands=[headercmd,referencecmd],public_pool_member=False))
        run.save('native_audit.json',dict(passed=True,samples=reports,resource=resource,versions=versions,ffprobe_version=ffversion,
            GPU_hours=0,new_encoder_calls=0,test_quality_access='none; four train/val technical sources only',
            interpretation='raw path/content not independently authenticated against YouTube; accepted source-ID mapping and decoder/crop/PTS parity only. Half-open annotation seconds adapted, not claimed official cutting parity. Not upstream normalization/checkpoint feature parity or full selected scientific cohort coverage.'))


if __name__=='__main__':main()
