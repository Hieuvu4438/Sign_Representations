"""Bounded public annotation release and local metadata/source/split audit."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import unittest
from urllib.request import Request, urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from msasl_metadata import youtube_id, issues
from resources import scratch_preflight


def archive_files(raw, expanded_cap):
    result = {}; names = []
    wanted = {'MSASL_train.json','MSASL_val.json','MSASL_test.json','MSASL_classes.json','MSASL_synonym.json'}
    total = 0
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for entry in archive.infolist():
                names.append(entry.filename); total += entry.file_size
                if total > expanded_cap: raise ValueError('Archive expansion bound exceeded')
                basename = Path(entry.filename).name
                if basename in wanted:
                    if basename in result: raise ValueError('Duplicate selected archive member')
                    with archive.open(entry) as stream: result[basename] = stream.read(entry.file_size+1)
                    if len(result[basename]) != entry.file_size: raise ValueError('Archive member size differs')
    else:
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:*') as archive:
            for entry in archive:
                names.append(entry.name); total += max(entry.size,0)
                if total > expanded_cap: raise ValueError('Archive expansion bound exceeded')
                basename = Path(entry.name).name
                if entry.isfile() and basename in wanted:
                    if basename in result: raise ValueError('Duplicate selected archive member')
                    with archive.extractfile(entry) as stream: result[basename] = stream.read(entry.size+1)
                    if len(result[basename]) != entry.size: raise ValueError('Archive member size differs')
    if set(result) != wanted: raise ValueError('Expected five annotation members not found')
    return result, names


def main():
    p=argparse.ArgumentParser()
    for name in ('config','run-id','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config
        if c.get('software_only'):
            result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(
                unittest.defaultTestLoader.discover('method/frozen_probe/tests',pattern='test_msasl_metadata.py'))
            run.save('metadata_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
                                               failures=len(result.failures),errors=len(result.errors)))
            if not result.wasSuccessful():raise ValueError('Metadata contract oracles failed')
            return
        resource=scratch_preflight('/dev/shm',256<<20)
        with urlopen(Request(c['release_url'],headers={'User-Agent':'sign-representation-metadata-audit'}),timeout=25) as response:
            raw=response.read(c['archive_download_cap']+1)
            http=dict(status=response.status,url=c['release_url'],final_url=response.geturl(),
                      headers={name:response.headers.get(name) for name in ('Content-Length','ETag','Last-Modified','Content-Type')},
                      bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if len(raw)>c['archive_download_cap']:raise ValueError('Archive download bound exceeded')
        run.store.write_bytes('official_annotation_archive.bin',raw)
        members,names=archive_files(raw,c['archive_expanded_cap'])
        root=Path(c['dataset_root']); equality={}; local={}
        for name, released in members.items():
            localbytes=(root/name).read_bytes(); left=json.loads(localbytes);right=json.loads(released)
            local[name]=left
            differences=[]; count=None
            if isinstance(left,list) and isinstance(right,list) and len(left)==len(right):
                count=sum(x==y for x,y in zip(left,right))
                for i,(x,y) in enumerate(zip(left,right)):
                    if x!=y and len(differences)<10:
                        differences.append(dict(row=i,local_keys=sorted(x) if isinstance(x,dict) else None,
                                                released_keys=sorted(y) if isinstance(y,dict) else None))
            equality[name]=dict(local_sha256=hashlib.sha256(localbytes).hexdigest(),released_sha256=hashlib.sha256(released).hexdigest(),
                local_bytes=len(localbytes),released_bytes=len(released),same_bytes=localbytes==released,
                parsed_exact_equal=left==right,equal_rows=count,local_rows=len(left),released_rows=len(right),first_differences=differences)
        release=dict(http=http,archive_member_names=names,selected_member_comparisons=equality,
                     scope='annotation identity only; archive retained, no member extraction to disk or media download')
        run.save('release_provenance.json',release)
        classes=local['MSASL_classes.json'];assert isinstance(classes,list) and len(classes)==1000
        poolrows=list(csv.reader(io.StringIO(Path(c['public_pool_csv']).read_text())))
        assert all(len(r)==2 for r in poolrows);pool={r[0] for r in poolrows};assert len(pool)==39197
        split_ids={}; signer_ids={}; legacy_ids={}; clip_ids={}; splits={}; classcounts={}
        table=io.StringIO();writer=csv.writer(table,lineterminator='\n')
        writer.writerow(['split','annotation_index','label','signer_id','legacy_signer','youtube_parent_id','start_time_s','end_time_s','raw_candidate_exists','pool_membership'])
        missing=set(); available=set(); row_count=0
        for split in ('train','val','test'):
            rows=local['MSASL_'+split+'.json'];assert isinstance(rows,list)
            parentids=set();signers=set();legacy=set();clips=set();errors=Counter();examples=[];inpool=0;invalidurl=0;present=0;short=0
            counts=Counter()
            for i,row in enumerate(rows):
                problems=issues(row,len(classes));errors.update(problems)
                if problems and len(examples)<20:examples.append(dict(annotation_index=i,issues=problems))
                key=youtube_id(row.get('url'));exists=bool(key and (root/'video'/(key+'.mp4')).is_file())
                if key:
                    parentids.add(key);inpool+=int(key in pool)
                    if exists:available.add(key)
                    else:missing.add(key)
                    clips.add((key,row.get('start_time'),row.get('end_time')))
                else:invalidurl+=1
                if type(row.get('signer_id')) is int and row['signer_id']>=0:signers.add(row['signer_id'])
                if type(row.get('signer')) is int and row['signer']>=0:legacy.add(row['signer'])
                if type(row.get('label')) is int:counts[row['label']]+=1
                if type(row.get('start')) is int and type(row.get('end')) is int:short+=int(row['end']-row['start']<16)
                present+=int(exists);row_count+=1
                writer.writerow([split,i,row.get('label'),row.get('signer_id'),row.get('signer'),key or '',row.get('start_time'),row.get('end_time'),int(exists),str(key in pool) if key else 'unknown'])
            split_ids[split]=parentids;signer_ids[split]=signers;legacy_ids[split]=legacy;clip_ids[split]=clips;classcounts[split]=dict(counts)
            splits[split]=dict(annotation_rows=len(rows),unique_source_parents=len(parentids),unparsed_URL_rows=invalidurl,
                signer_id_unique_nonnegative=len(signers),legacy_signer_unique_nonnegative=len(legacy),legacy_signer_missing_sentinel_rows=sum(r.get('signer')==-1 for r in rows),
                label_classes=len(counts),raw_candidate_present_rows=present,public_pool_present_rows=inpool,
                unique_parent_ids_in_public_pool=len(parentids&pool),metadata_frame_intervals_below16=short,
                annotation_contract_issue_counts=dict(errors),first_issue_examples=examples,
                unique_exact_parent_time_clips=len(clips),duplicate_parent_time_rows=len(rows)-invalidurl-len(clips))
        intersections={}
        for left,right in (('train','val'),('train','test'),('val','test')):
            intersections[left+'_'+right]=dict(signer_id_count=len(signer_ids[left]&signer_ids[right]),
                legacy_signer_nonnegative_count=len(legacy_ids[left]&legacy_ids[right]),
                source_parent_count=len(split_ids[left]&split_ids[right]),exact_parent_time_count=len(clip_ids[left]&clip_ids[right]))
        # Directory enumeration only; numeric processed filenames lack an accepted annotation join.
        media={}
        for folder in ('video','cut_processed_video'):
            path=root/folder;count=0;bytes_total=0
            if path.is_dir():
                for entry in os.scandir(path):
                    if entry.is_file(follow_symlinks=False) and entry.name.endswith('.mp4'):
                        count+=1;bytes_total+=entry.stat(follow_symlinks=False).st_size
            media[folder]=dict(path=str(path),exists=path.is_dir(),immediate_mp4_files=count,logical_bytes=bytes_total)
        run.store.write_bytes('annotation_rows.csv',table.getvalue().encode('utf-8'))
        run.store.write_bytes('unavailable_source_parent_ids.txt',(''.join(k+'\n' for k in sorted(missing))).encode())
        run.save('metadata_audit.json',dict(splits=splits,split_intersections=intersections,class_counts=classcounts,
            total_annotation_rows=row_count,unique_parents_all_splits=len(set.union(*split_ids.values())),
            raw_candidate_unique_parents_available=len(available),raw_candidate_unique_parents_missing=len(missing),
            local_media_inventory=media,all_local_annotation_members_byte_equal=all(v['same_bytes'] for v in equality.values()),
            all_local_annotation_members_parsed_equal=all(v['parsed_exact_equal'] for v in equality.values()),
            raw_source_path_rule='video/<accepted_annotation_YouTube_ID>.mp4; candidate existence only, not media-byte/source-content proof',
            processed_numeric_clip_join='unverified; do not infer from numeric filename',
            resource=resource,GPU_hours=0,new_encoder_calls=0,test_quality_access='none; metadata only',
            interpretation='public corpus pool overlap is not checkpoint training membership; signer IDs are annotation identities, not independently verified persons; durations/FPS/boxes metadata are not decoded native contracts',
            evaluator='primary paper specifies mean-per-class Top1/Top5; no code parity performed for MS-ASL; planned dictionary retrieval requires adapted protocol'))


if __name__=='__main__':main()
