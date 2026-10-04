"""Independent source/CSV/release recount from saved audit inputs."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit, parse_qsl
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store


def source_id(text):
    if text.startswith('www.youtube.com/watch?'):text='https://'+text
    p=urlsplit(text)
    assert p.scheme in ('http','https') and not p.username and not p.password and not p.port
    if p.netloc in ('youtube.com','www.youtube.com','m.youtube.com'):
        assert p.path=='/watch'
        values=[v for k,v in parse_qsl(p.query,keep_blank_values=True) if k=='v']
        assert len(values)==1;key=values[0]
    else:
        assert p.netloc=='youtu.be' and p.path.count('/')==1;key=p.path[1:]
    assert re.fullmatch('[A-Za-z0-9_-]{11}',key)
    return key


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;root=Path(c['audit_runtime']);dataset=Path(c['dataset_root'])
        audit=json.loads((root/'metadata_audit.json').read_text())
        saved=list(csv.DictReader((root/'annotation_rows.csv').open()))
        pool={r[0] for r in csv.reader(Path(c['public_pool_csv']).open())};assert len(pool)==39197
        with zipfile.ZipFile(root/'official_annotation_archive.bin') as archive:
            members={Path(name).name:name for name in archive.namelist()}
            for n in ('train','val','test','classes','synonym'):
                name='MSASL_'+n+'.json';assert archive.read(members[name])==(dataset/name).read_bytes()
        position=0;sets={};counts={};allparents=set();presentparents=set();missingparents=set()
        for split in ('train','val','test'):
            rows=json.loads((dataset/f'MSASL_{split}.json').read_bytes())
            parents=set();signers=set();legacy=set();clips=set();labels=Counter();present=Counter();poolcounts=Counter();clean_present=Counter()
            inpool=0;short=0;sentinel=0
            for i,r in enumerate(rows):
                key=source_id(r['url']);item=saved[position];position+=1
                exists=(dataset/'video'/(key+'.mp4')).is_file();ismember=key in pool
                assert item['split']==split and int(item['annotation_index'])==i and item['youtube_parent_id']==key
                for field in ('label','signer_id'):assert int(item[field])==r[field]
                assert int(item['legacy_signer'])==r['signer']
                assert float(item['start_time_s'])==r['start_time'] and float(item['end_time_s'])==r['end_time']
                assert int(item['raw_candidate_exists'])==int(exists) and item['pool_membership']==str(ismember)
                parents.add(key);signers.add(r['signer_id']);clips.add((key,r['start_time'],r['end_time']))
                if r['signer']>=0:legacy.add(r['signer'])
                sentinel+=int(r['signer']==-1);short+=int(r['end']-r['start']<16);inpool+=ismember
                labels[r['label']]+=1
                if exists:present[r['label']]+=1;presentparents.add(key)
                else:missingparents.add(key)
                if ismember:poolcounts[r['label']]+=1
                if exists and not ismember:clean_present[r['label']]+=1
            s=audit['splits'][split]
            expected=dict(annotation_rows=len(rows),unique_source_parents=len(parents),unparsed_URL_rows=0,
                signer_id_unique_nonnegative=len(signers),legacy_signer_unique_nonnegative=len(legacy),
                legacy_signer_missing_sentinel_rows=sentinel,label_classes=len(labels),raw_candidate_present_rows=sum(present.values()),
                public_pool_present_rows=inpool,unique_parent_ids_in_public_pool=len(parents&pool),metadata_frame_intervals_below16=short,
                unique_exact_parent_time_clips=len(clips),duplicate_parent_time_rows=len(rows)-len(clips))
            assert all(s[k]==v for k,v in expected.items())
            assert {int(k):v for k,v in audit['class_counts'][split].items()}==labels
            sets[split]=(parents,signers,legacy,clips);allparents|=parents
            counts[split]=dict(expected,candidate_classes_with_any_raw=len(present),candidate_classes_without_raw=sorted(set(range(1000))-set(present)),
                raw_candidate_rows_outside_public_pool=sum(clean_present.values()),candidate_classes_outside_public_pool=len(clean_present),
                candidate_support_by_class={str(k):present[k] for k in range(1000)})
        assert position==len(saved)==25513
        assert len(allparents)==audit['unique_parents_all_splits'] and len(presentparents)==audit['raw_candidate_unique_parents_available']
        assert len(missingparents)==audit['raw_candidate_unique_parents_missing']
        assert set((root/'unavailable_source_parent_ids.txt').read_text().splitlines())==missingparents
        for pair in ('train_val','train_test','val_test'):
            l,r=pair.split('_');s=audit['split_intersections'][pair]
            for j,k in enumerate(('source_parent_count','signer_id_count','legacy_signer_nonnegative_count','exact_parent_time_count')):
                assert len(sets[l][j]&sets[r][j])==s[k]
        stores={p:verify_store(p) for p in c['completed_runtimes']}
        run.save('metadata_verification.json',dict(passed=True,CSV_rows_verified=position,release_members_byte_equal=5,
            stores=stores,independent_counts=counts,scope='same-agent independent arithmetic; metadata only, no physical identity/media decoding/quality or confirmed checkpoint disjointness; outside pool is an operational source-list comparison'))


if __name__=='__main__':main()
