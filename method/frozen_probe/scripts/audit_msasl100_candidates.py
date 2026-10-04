"""Metadata-only candidate coverage for a prospective MS-ASL100 pilot."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from msasl_metadata_v2 import youtube_id,issues


def main():
    p=argparse.ArgumentParser()
    for n in ('config','run-id','output-dir'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config;root=Path(c['dataset_root']);pool={r[0] for r in csv.reader(Path(c['public_pool_csv']).open())}
        allrows={s:json.loads((root/f'MSASL_{s}.json').read_bytes()) for s in ('train','val','test')}
        parentsets={s:{youtube_id(r['url']) for r in rows} for s,rows in allrows.items()}
        grouped={};parentmulti=set();interval_labels=defaultdict(set)
        for s,rows in allrows.items():
            for r in rows:interval_labels[(youtube_id(r['url']),r['start_time'],r['end_time'])].add(r['label'])
        for left,right in (('train','val'),('train','test'),('val','test')):
            grouped[left+'_'+right]=sorted(parentsets[left]&parentsets[right]);parentmulti|=set(grouped[left+'_'+right])
        candidates=[];exclusions=[];summary={};conflicts={k for k,v in interval_labels.items() if len(v)>1}
        for split in ('train','val'):
            counts=Counter();retained=Counter();uniqueparents=set();signers=set();seen=set();candidatebytes={}
            for index,row in enumerate(allrows[split]):
                if not 0<=row['label']<100:continue
                key=youtube_id(row['url']);path=root/'video'/f'{key}.mp4';interval=(key,row['start_time'],row['end_time']);reason=None
                if issues(row,1000):reason='metadata_contract'
                elif not path.is_file():reason='raw_candidate_absent'
                elif key in pool:reason='source_in_bound_public_pool'
                elif key in parentmulti:reason='source_parent_across_official_splits'
                elif interval in conflicts:reason='exact_interval_multiple_class_labels'
                elif not 0<row['end_time']-row['start_time']<=8:reason='outside_declared_short_clip_duration'
                elif interval in seen:reason='duplicate_exact_interval'
                item=dict(sample_id=f'{split}_{index:05d}',split=split,annotation_index=index,class_id=row['label'],signer_id=row['signer_id'],
                    parent_id=key,start_time_seconds=row['start_time'],end_time_seconds=row['end_time'],metadata_FPS=row['fps'],
                    legacy_frame_interval=[row['start'],row['end']],box_yxyx=row['box'],raw_path=str(path))
                if reason:
                    counts[reason]+=1;exclusions.append(dict(item,reason=reason))
                else:
                    seen.add(interval);retained[row['label']]+=1;uniqueparents.add(key);signers.add(row['signer_id'])
                    size=path.stat().st_size;candidatebytes[key]=size;candidates.append(dict(item,raw_candidate_bytes=size))
            summary[split]=dict(official_MSASL100_annotation_rows=sum(0<=r['label']<100 for r in allrows[split]),
                metadata_candidate_rows=sum(retained.values()),candidate_classes=len(retained),zero_candidate_classes=sorted(set(range(100))-set(retained)),
                class_counts={str(k):retained[k] for k in range(100)},exclusion_counts=dict(counts),unique_candidate_parents=len(uniqueparents),
                candidate_signer_ids=sorted(signers),unique_parent_compressed_bytes=sum(candidatebytes.values()),
                maximum_parent_compressed_bytes=max(candidatebytes.values(),default=0),
                candidate_metadata_duration_seconds=sum(x['end_time_seconds']-x['start_time_seconds'] for x in candidates if x['split']==split))
        assert all(summary[s]['official_MSASL100_annotation_rows']==summary[s]['metadata_candidate_rows']+sum(summary[s]['exclusion_counts'].values()) for s in summary)
        run.save('candidate_records.json',dict(candidates=candidates,exclusions=exclusions))
        run.save('candidate_coverage.json',dict(splits=summary,crosssplit_parent_IDs=grouped,exact_interval_multiple_label_count_all_classes=len(conflicts),
            prospective_gallery_seeds=[1701,1702,1703],GPU_hours=0,new_encoder_calls=0,test_quality_access='none; test metadata source grouping only',
            native_decoder_and_feature_gate='not run; metadata candidates cannot be promoted automatically to scientific cohort',
            scope='MS-ASL100 is official label<100 subset; prospective adapted shortclip/available/outsidepublicpool/sourcegroup/dedup coverage only. No actual checkpoint-unseen claim, quality selection, official score reproduction or native16 feasibility proof.'))


if __name__=='__main__':main()
