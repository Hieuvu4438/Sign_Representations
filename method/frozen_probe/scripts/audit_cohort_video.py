"""Full selected-cohort decode audit, no encoder inference or performance."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_signrep import sha
from video_contract import summarize_timeline,fingerprint_indices,dhash64

REPO=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status={'run_id':args.run_id,'protocol':config['protocol'],'seed':args.seed,
            'status':'running','pid':os.getpid(),'host':platform.node(),'GPU_hours':0,
            'start_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable]+sys.argv,
            'script_sha256':sha(__file__),'config_sha256':sha(args.config)}
    report={'run_id':args.run_id,'cohort_sha256':config['cohort_sha256'],
            'scope':'full selected-cohort pixel decode/timeline; no encoder features',
            'samples':[],'failures':[],'passed':False}

    def save(name,value):
        temp=args.output_dir/(name+'.tmp')
        temp.write_text(json.dumps(value,separators=(',',':'))+'\n')
        temp.replace(args.output_dir/name)

    def timeout_handler(*_): raise TimeoutError('Registered900s audit budget exhausted')

    save('status.json',status)
    signal.signal(signal.SIGALRM,timeout_handler)
    signal.alarm(config['timeout_seconds'])
    try:
        import cv2
        cv2.setNumThreads(1)
        cohort_path=REPO/config['cohort']
        if sha(cohort_path)!=config['cohort_sha256']: raise ValueError('Cohort hash changed')
        cohort=json.loads(cohort_path.read_text())
        samples=cohort['samples']
        if len(samples)!=config['expected_samples'] or any(s['split'] not in ('train','val') for s in samples):
            raise ValueError('Cohort count/access mismatch')
        if 3*sum(s['source_bytes'] for s in samples)>config['max_planned_source_read_bytes']:
            raise ValueError('Logical read estimate exceeds registered limit')
        snapshots=args.output_dir/'source_snapshot';snapshots.mkdir()
        paths=[Path(__file__),REPO/'method/frozen_probe/video_contract.py',
               REPO/'method/frozen_probe/temporal.py',REPO/'method/frozen_probe/scripts/audit_signrep.py']
        report['code_hashes']={str(p.relative_to(REPO)):sha(p) for p in paths}
        for path in paths: shutil.copyfile(path,snapshots/path.name)
        shutil.copyfile(args.config,args.output_dir/'config.json')
        ffprobe_version=subprocess.check_output(['ffprobe','-version'],text=True).splitlines()[0]
        report['environment']={'python':sys.version,'executable':sys.executable,'cv2':cv2.__version__,
                               'ffprobe':ffprobe_version,'decoder_threads':1}
        for number,sample in enumerate(samples,1):
            entry={k:sample[k] for k in ('sample_id','split','participant_id','class_id','source_sha256')}
            cap=None
            try:
                path=Path(sample['input_path'])
                if path.stat().st_size!=sample['source_bytes'] or sha(path)!=sample['source_sha256']:
                    raise ValueError('Video source changed since cohort registration')
                command=['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_entries',
                         'stream=width,height,avg_frame_rate,r_frame_rate,duration:frame=best_effort_timestamp_time',
                         '-of','json',str(path)]
                response=subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
                if response.stderr.strip(): raise ValueError(f'ffprobe decode error: {response.stderr[:500]}')
                metadata=json.loads(response.stdout)
                pts=[float(f['best_effort_timestamp_time']) for f in metadata.get('frames',[])]
                if not 16<=len(pts)<=config['max_frames_per_clip']: raise ValueError(f'Unsupported clip length:{len(pts)}')
                stream=metadata['streams'][0]
                indices=fingerprint_indices(len(pts),config['fingerprint_quantiles'])
                selected=set(indices);fingerprints={}
                cap=cv2.VideoCapture(str(path))
                if not cap.isOpened(): raise ValueError('OpenCV cannot open source')
                count=0
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    if frame.shape!=(stream['height'],stream['width'],3): raise ValueError('Frame dimensions changed')
                    if count in selected:
                        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        thumb=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
                        fingerprints[count]={'dhash64':dhash64(thumb),'gray_std':float(thumb.std()),
                                             'gray_range':int(thumb.max())-int(thumb.min())}
                    count+=1
                    if count>config['max_frames_per_clip']:raise ValueError('Decoded frame bound exceeded')
                entry.update(summarize_timeline(pts,count,config['window_length'],config['stride']))
                entry.update(stream=stream,pts_s=pts,fingerprint_frame_indices=indices,
                             fingerprints=[fingerprints[i] for i in indices],passed=True)
                report['samples'].append(entry)
            except TimeoutError:
                raise
            except Exception as error:
                entry.update(passed=False,error=repr(error))
                report['failures'].append(entry)
            finally:
                if cap is not None:cap.release()
            if number%50==0 or number==len(samples):
                status['processed_samples']=number
                save('status.json',status);save('video_audit.json',report)
                print(json.dumps({'processed':number,'expected':len(samples),'failures':len(report['failures'])}),flush=True)
        windows=sum(s['windows'] for s in report['samples'])
        report['summary']={'expected':len(samples),'passed_samples':len(report['samples']),
                           'failed_samples':len(report['failures']),'total_valid_windows':windows,
                           'uncovered_tail_frames':sum(s['uncovered_tail_frames'] for s in report['samples']),
                           'float32_feature_bytes':windows*768*4,'float32_features_latent_activity_bytes':windows*(768*2+2)*4,
                           'planned_logical_source_reads':3*sum(s['source_bytes'] for s in samples),
                           'by_split':{split:{'samples':sum(s['split']==split for s in report['samples']),
                                             'frames':sum(s['frames'] for s in report['samples'] if s['split']==split),
                                             'windows':sum(s['windows'] for s in report['samples'] if s['split']==split)} for split in ('train','val')}}
        report['passed']=not report['failures'] and len(report['samples'])==len(samples)
        save('video_audit.json',report)
        if not report['passed']:raise ValueError('Full cohort coverage failed; retain failure IDs, no silent exclusion')
        status.update(status='completed',exit_code=0,output_sha256=sha(args.output_dir/'video_audit.json'))
    except Exception as error:
        status.update(status='failed',exit_code=1,error=repr(error))
        (args.output_dir/'error.txt').write_text(traceback.format_exc())
        save('video_audit.json',report)
    finally:
        signal.alarm(0)
        status.update(end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
        save('status.json',status)
    print(json.dumps(status),flush=True)
    if status['status']!='completed':sys.exit(1)


if __name__=='__main__':main()
