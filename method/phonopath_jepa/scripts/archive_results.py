#!/usr/bin/env python3
"""Archive verified numeric results, then remove only PhonoPath's bulky runtime.

No shared dataset, environment, unrelated method, or outside checkpoint is removed.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import numpy as np

REPO=Path('/home/haipd/Sign_Representations')
METHOD=REPO/'method/phonopath_jepa'
ARCHIVE=REPO/'artifacts/phonopath_jepa_results_20261019'
SERVICES=('phonopath-a1-full160.service','phonopath-a1-e040-retrieval.service','phonopath-a1-e040-test-retrieval.service','phonopath-a1-e160-test-retrieval.service')
EVALUATIONS={'epoch040_val':'retrieval_e040','epoch040_test':'retrieval_e040_test','epoch160_test':'retrieval_e160_test'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


def allocation(path):
    seen=set();total=0
    for p in path.rglob('*'):
        if p.is_file() and not p.is_symlink():
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);total+=s.st_blocks*512
    return total


def stop_services():
    saved={}
    for service in SERVICES:
        result=subprocess.run(['systemctl','--user','show',service,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result','-p','NRestarts'],capture_output=True,text=True)
        saved[service]=result.stdout
        unit=Path.home()/'.config/systemd/user'/service
        if unit.exists():
            target=ARCHIVE/'services'/service;target.parent.mkdir(exist_ok=True);shutil.copy2(unit,target)
    write(ARCHIVE/'services/status_before_cleanup.json',saved)
    subprocess.run(['systemctl','--user','disable','--now',*SERVICES],check=True)
    for service in SERVICES:
        pid=subprocess.check_output(['systemctl','--user','show',service,'-p','MainPID','--value'],text=True).strip()
        if pid!='0':raise RuntimeError(f'Service child still active: {service}/{pid}')


def prepare():
    if ARCHIVE.exists():raise FileExistsError(f'{ARCHIVE} exists; inspect or use verify/clean')
    ARCHIVE.mkdir(parents=True)
    for directory in ('results','services','metadata'):(ARCHIVE/directory).mkdir()
    stop_services()
    inventory=[];inode_hash={};counts=collections.Counter();allocated=0;seen=set()
    paths=[p for p in METHOD.rglob('*') if p.is_file() and not p.is_symlink()]
    for i,p in enumerate(paths):
        st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in inode_hash:inode_hash[key]=sha256(p)
        if key not in seen:seen.add(key);allocated+=st.st_blocks*512
        inventory.append(dict(path=str(p.relative_to(METHOD)),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=inode_hash[key],
                              device=st.st_dev,inode=st.st_ino,hardlinks=st.st_nlink))
        if i%250==0:print('inventory',i,'/',len(paths),flush=True)
    write(ARCHIVE/'metadata/source_inventory.json',inventory)
    preserved={};recomputed={}
    for name,relative in EVALUATIONS.items():
        src=METHOD/'runtime'/relative;dest=ARCHIVE/'results'/name;dest.mkdir()
        result=json.loads((src/'results.json').read_text())
        assert result['status']=='complete'
        shutil.copy2(src/'results.json',dest/'results.json');shutil.copy2(src/'REPORT_VI.md',dest/'REPORT_VI.md')
        recomputed[name]={}
        for key,record in result['readouts'].items():
            tag={'grp':'grp','grp+P':'grpP','mw':'mw','ord2':'ord2'}[key]
            ranks_path=src/'results'/f'{tag}_query_ranks.npz'
            with np.load(ranks_path,allow_pickle=False) as z:
                values={k:z[k] for k in z.files}
                ranks=values['ranks'];n=len(ranks)
                assert n==result['query_n'] and len(values['ids'])==n
                metrics={'R@1':100*np.mean(ranks<=1),'R@5':100*np.mean(ranks<=5),'R@10':100*np.mean(ranks<=10),
                         'MRR':np.mean(1/ranks),'DCG':np.mean(1/np.log2(1+ranks))}
                gloss=values['gloss'];metrics['macro_R@1']=100*np.mean([np.mean(ranks[gloss==c]<=1) for c in np.unique(gloss)])
                for metric,value in metrics.items():assert abs(float(value)-record['metrics'][metric])<1e-9,(name,key,metric)
                np.savez_compressed(dest/f'{tag}_query_ranks.npz',**values)
            with np.load(dest/f'{tag}_query_ranks.npz',allow_pickle=False) as z:
                for field,value in values.items():assert np.array_equal(value,z[field]),(name,key,field)
            recomputed[name][key]={k:float(v) for k,v in metrics.items()}
        preserved[name]=str(dest.relative_to(ARCHIVE))
    write(ARCHIVE/'metadata/recomputed_metrics.json',recomputed)
    # Preserve every textual provenance/source/config/log file in a compressed
    # record, including the entire training trace. Do not include model weights,
    # feature matrices, shards, or PCA buffers.
    extensions={'.py','.sh','.json','.jsonl','.yaml','.yml','.toml','.md','.log','.xml','.csv','.txt'}
    records=[p for p in paths if p.suffix in extensions and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    small_stats=[p for p in paths if p.name=='stats.pt' and p.stat().st_size<200000]
    selected=records+small_stats
    with tarfile.open(ARCHIVE/'records.tar.gz','w:gz',compresslevel=5) as tf:
        for p in selected:tf.add(p,arcname=str(p.relative_to(REPO)),recursive=False)
    write(ARCHIVE/'metadata/record_members.json',[str(p.relative_to(REPO)) for p in selected])
    # Plain metadata is accessible without unpacking the full trace.
    for rel in ('runtime/full160/train/manifest.json','runtime/full160/train/progress.json','runtime/full160/config.json',
                'runtime/full160/lock.json','runtime/test_ledger.json'):
        target=ARCHIVE/'metadata'/Path(rel).name
        if target.exists():target=ARCHIVE/'metadata'/('full160_'+Path(rel).name)
        shutil.copy2(METHOD/rel,target)
    for key in ('full160','retrieval_e040','retrieval_e040_test','retrieval_e160_test'):
        for name in ('status.json','lock.json'):
            shutil.copy2(METHOD/'runtime'/key/name,ARCHIVE/'metadata'/f'{key}_{name}')
    # Compact loss curve from unique optimizer steps; full original trace is in tar.
    values={};events=[]
    with (METHOD/'runtime/full160/train/train_log.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'loss' in r:
                values[r['step']]=(r['loss'],r['grad_norm'],r['terms']['near']['S'],r['terms']['near']['P'],r['terms']['far']['S'],r['terms']['far']['P'])
            else:events.append(r)
    assert max(values)==200640
    import csv
    with (ARCHIVE/'loss_by_1000_steps.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step_start','step_end','records','epoch_end','loss','grad_norm','near_S','near_P','far_S','far_P'])
        for start in range(1,200641,1000):
            end=min(200640,start+999);a=np.array([values[i] for i in range(start,end+1) if i in values])
            writer.writerow([start,end,len(a),end/1254,*a.mean(0).tolist()])
    write(ARCHIVE/'metadata/resume_events.json',events)
    result=dict(status='ARCHIVED_PENDING_VERIFICATION',source_root=str(METHOD),physical_bytes_before=allocated,
                unique_training_steps=len(values),optimizer_steps=200640,epochs=160,model_weights_retained=False,
                feature_matrices_retained=False,pca_parameters_retained=False,raw_training_log_retained=True,
                per_query_ranks_retained=True,results=preserved,created_unix=time.time())
    write(ARCHIVE/'archive_manifest.json',result)
    print(json.dumps(result,indent=2),flush=True)


def verify():
    inventory=json.loads((ARCHIVE/'metadata/source_inventory.json').read_text())
    source_hash={f'method/phonopath_jepa/{r["path"]}':r['sha256'] for r in inventory}
    wanted=json.loads((ARCHIVE/'metadata/record_members.json').read_text());actual=[]
    with tarfile.open(ARCHIVE/'records.tar.gz','r:gz') as tf:
        for member in tf:
            if not member.isfile():raise ValueError('Unexpected record member')
            stream=tf.extractfile(member);h=hashlib.sha256()
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
            if h.hexdigest()!=source_hash[member.name]:raise ValueError(f'Archive hash mismatch: {member.name}')
            actual.append(member.name)
    assert sorted(actual)==sorted(wanted)
    checked=0
    for name in EVALUATIONS:
        dest=ARCHIVE/'results'/name;result=json.loads((dest/'results.json').read_text())
        for key,record in result['readouts'].items():
            tag={'grp':'grp','grp+P':'grpP','mw':'mw','ord2':'ord2'}[key]
            with np.load(dest/f'{tag}_query_ranks.npz',allow_pickle=False) as z:
                assert len(z['ranks'])==result['query_n']
                assert abs(100*np.mean(z['ranks']<=1)-record['metrics']['R@1'])<1e-9
            checked+=1
    checksums={str(p.relative_to(ARCHIVE)):sha256(p) for p in ARCHIVE.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS.json','VERIFIED.json')}
    write(ARCHIVE/'SHA256SUMS.json',checksums)
    verification=dict(status='VERIFIED',text_record_members=len(actual),readout_rank_sets_verified=checked,
                      complete_training_trace_in_archive=True,verified_unix=time.time(),archive_sha256=sha256(ARCHIVE/'records.tar.gz'))
    write(ARCHIVE/'VERIFIED.json',verification);print(json.dumps(verification,indent=2),flush=True)


def clean():
    verified=json.loads((ARCHIVE/'VERIFIED.json').read_text());assert verified['status']=='VERIFIED'
    for rel,digest in json.loads((ARCHIVE/'SHA256SUMS.json').read_text()).items():
        if sha256(ARCHIVE/rel)!=digest:raise ValueError(f'Preserved artifact changed: {rel}')
    for service in SERVICES:
        result=subprocess.run(['systemctl','--user','show',service,'-p','MainPID','--value'],capture_output=True,text=True)
        if result.stdout.strip() not in ('','0'):raise RuntimeError(f'Service still live: {service}')
    old=json.loads((ARCHIVE/'archive_manifest.json').read_text())
    removed=[]
    runtime=METHOD/'runtime'
    if runtime.is_symlink():raise RuntimeError('Refusing symlink runtime')
    if runtime.exists():shutil.rmtree(runtime);removed.append(str(runtime))
    # Remove the CPU/smoke binary weights outside runtime; keep small evidence JSON/XML.
    for p in (METHOD/'evidence').rglob('*'):
        if p.is_file() and p.suffix in ('.pt','.npz','.npy'):
            p.unlink();removed.append(str(p))
    for name in ('__pycache__','.pytest_cache'):
        for p in sorted(METHOD.rglob(name),key=lambda q:len(q.parts),reverse=True):
            if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
    for service in SERVICES:
        unit=Path.home()/'.config/systemd/user'/service
        if unit.exists():unit.unlink();removed.append(str(unit))
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','reset-failed',*SERVICES],capture_output=True,text=True)
    archive_bytes=allocation(ARCHIVE);remaining=allocation(METHOD)
    report=dict(status='COMPLETED',removed_paths=removed,physical_method_bytes_before=old['physical_bytes_before'],
                physical_method_bytes_after=remaining,archive_bytes=archive_bytes,
                reclaimed_in_method_bytes=old['physical_bytes_before']-remaining,
                net_reclaimed_bytes_after_archive=old['physical_bytes_before']-remaining-archive_bytes,
                free_disk_bytes_now=shutil.disk_usage(METHOD).free,shared_pose_preserved=True,other_methods_preserved=True,
                code_configs_tests_preserved=True,model_weights_deleted=True,resume_or_inference_from_old_weights_possible=False,
                training_and_results_archive=str(ARCHIVE),completed_unix=time.time())
    write(ARCHIVE/'CLEANUP_REPORT.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','verify','clean']);a=ap.parse_args()
    {'prepare':prepare,'verify':verify,'clean':clean}[a.action]()
if __name__=='__main__':main()
