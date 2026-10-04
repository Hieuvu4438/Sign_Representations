"""Transparent fingerprint pair screening, never automatically drops samples."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_signrep import sha


def compare_pair(a,b):
    if len(a['fingerprints'])!=5 or len(b['fingerprints'])!=5:raise ValueError('Expected five fingerprints')
    ratio=min(a['frames'],b['frames'])/max(a['frames'],b['frames'])
    if ratio<.8:return None
    distances=[(int(x['dhash64'],16)^int(y['dhash64'],16)).bit_count()
               for x,y in zip(a['fingerprints'],b['fingerprints']) if x['gray_std']>=5 and y['gray_std']>=5]
    if len(distances)<3:return None
    mean=sum(distances)/len(distances)
    return {'mean_hamming':mean,'max_hamming':max(distances),'distances':distances,
            'information_positions':len(distances),'frame_count_ratio':ratio,
            'candidate':mean<=6 and max(distances)<=12}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status={'run_id':args.run_id,'status':'running','GPU_hours':0,'pid':os.getpid(),
            'host':platform.node(),'seed':args.seed,'start_utc':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable]+sys.argv,'script_sha256':sha(__file__),'input_sha256':sha(args.input)}
    def save(name,value):
        temp=args.output_dir/(name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(args.output_dir/name)
    def timeout_handler(*_):raise TimeoutError('30s duplicate screen ceiling exhausted')
    save('status.json',status);signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(30)
    try:
        audit=json.loads(args.input.read_text())
        if not audit['passed'] or len(audit['samples'])!=1691 or any(s['split'] not in ('train','val') for s in audit['samples']):
            raise ValueError('Only complete registered train/dev audit supported')
        shutil.copyfile(__file__,args.output_dir/'screen_duplicates.py')
        samples=sorted(audit['samples'],key=lambda s:s['sample_id']);candidates=[]
        tested={'cross_split':0,'within_split':0};eligible={'cross_split':0,'within_split':0}
        for index,a in enumerate(samples):
            for b in samples[index+1:]:
                kind='cross_split' if a['split']!=b['split'] else 'within_split';tested[kind]+=1
                result=compare_pair(a,b)
                if result is None:continue
                eligible[kind]+=1
                if result.pop('candidate'):
                    candidates.append({'pair_type':kind,'a':{k:a[k] for k in ('sample_id','split','participant_id','class_id')},
                                       'b':{k:b[k] for k in ('sample_id','split','participant_id','class_id')},**result})
        candidates.sort(key=lambda x:(x['mean_hamming'],x['max_hamming'],x['a']['sample_id'],x['b']['sample_id']))
        report={'run_id':args.run_id,'input_sha256':sha(args.input),'thresholds':{'min_information_positions':3,
                'min_gray_std':5,'min_frame_count_ratio':.8,'mean_hamming_max':6,'max_hamming_max':12},
                'pairs_tested':tested,'eligible_pairs':eligible,'candidates':candidates,
                'candidate_counts':{kind:sum(c['pair_type']==kind for c in candidates) for kind in tested},
                'scope':'screen only; no uniqueness/session/pretrain disjointness claim',
                'automatic_exclusions':0,'visual_review':'pending if cross-split candidates'}
        save('screen.json',report)
        if (args.output_dir/'screen.json').stat().st_size>1024*1024:raise ValueError('Candidate output exceeds registered budget')
        status.update(status='completed',exit_code=0,output_sha256=sha(args.output_dir/'screen.json'))
    except Exception as error:
        status.update(status='failed',exit_code=1,error=repr(error));(args.output_dir/'error.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0);status.update(end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
        save('status.json',status)
    print(json.dumps(status),flush=True)
    if status['status']!='completed':sys.exit(1)


if __name__=='__main__':main()
