"""Independent source-metadata recount; no imported partition/census algorithms."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store
from cache import file_sha


def read(path):
    return json.loads(Path(path).read_text())


def check_counts(actual, rows, classes):
    expected=Counter(r['class_id'] for r in rows)
    assert actual['clips']==len(rows)
    assert actual['class_counts']=={str(k):expected[k] for k in classes}
    assert actual['missing_classes']==[k for k in classes if expected[k]==0]
    assert actual['signers']==len(set(r['signer'] for r in rows))
    assert actual['sources']==len(set(r['source_group'] for r in rows))


def main():
    p=argparse.ArgumentParser()
    for name in ('config','run-id','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        parent=Path(run.config['measured_runtime']);s=read(parent/'status.json');c=read(parent/'config.json')
        assert s['status']=='completed' and s['exit_code']==0
        checked=verify_store(parent);assert file_sha(parent/'config.json')==s['config_sha256']
        for name,digest in s['output_hashes'].items():assert file_sha(parent/name)==digest
        rows=read(parent/'metadata_rows.json');tasks=read(parent/'candidate_partitions.json')
        census=read(parent/'metadata_census.json');assert census['passed_metadata_joins']
        assert all(census[k]==0 for k in ('feature_arrays_loaded','quality_scores_computed','thresholds_fitted','new_encoder_calls'))
        assert census['test_access']=='none'
        totals={};checked_cells=0
        for role,entries in rows.items():
            indexed={r['sample_id']:r for r in entries};assert len(indexed)==len(entries)
            assert all(r['split'] in ('train','val') for r in entries)
            classes=list(range(100));role_summary=census['roles'][role]
            if role!='MSASL100':
                prior=read(c['asl_inputs'][role]['identity_config']);cohort=read(prior['cohort_manifest'])
                source={r['sample_id']:r for r in cohort['samples']};assert set(source)==set(indexed)
                for key,sample in source.items():
                    out=indexed[key]
                    assert (out['class_id'],out['split'],out['signer'],out['source_sha256'],out['class_name'])==(
                        sample['class_id'],sample['split'],sample['participant_id'],sample['source_sha256'],sample['gloss'])
                train=[r for r in entries if r['split']=='train'];dev=[r for r in entries if r['split']=='val']
                signer_set=set(r['signer'] for r in train)
                keys={x:hashlib.sha256(('H007-calibration-v1|'+role+'|'+x).encode()).hexdigest() for x in signer_set}
                expected_signers=sorted(signer_set,key=lambda x:(keys[x],x))[:max(1,len(signer_set)//4)]
                assert role_summary['calibration_signers']==expected_signers
                calibration=[r for r in train if r['signer'] in expected_signers]
                pool=[r for r in train if r['signer'] not in expected_signers]
                assert tasks[role]['calibration_ids']==sorted(r['sample_id'] for r in calibration)
                assert tasks[role]['gallery_pool_ids']==sorted(r['sample_id'] for r in pool)
                for field in ('sample_id','source_group','source_sha256','signer'):
                    assert not set(r[field] for r in calibration)&set(r[field] for r in pool)
                for part,q in [('full',entries),('train',train),('dev',dev),('calibration',calibration),('gallery_pool',pool)]:
                    check_counts(role_summary[part],q,classes)
                pools=[('calibration',calibration),('dev',dev)]
            else:
                cfg=read(Path(c['msasl_inputs']['runtime'])/'config.json');source=read(cfg['extraction_manifest'])
                assert set(x['sample_id'] for x in source)==set(indexed)
                for sample in source:
                    out=indexed[sample['sample_id']]
                    assert (out['class_id'],out['split'],out['signer'],out['source_group'],out['source_sha256'])==(
                        sample['class_id'],sample['split'],str(sample['signer_id']),sample['parent_id'],sample['source_sha256'])
                train=[r for r in entries if r['split']=='train'];dev=[r for r in entries if r['split']=='val']
                for part,q in [('full',entries),('train',train),('dev',dev)]:check_counts(role_summary[part],q,classes)
                pools=[('dev',dev)];dictionary=read(c['msasl_inputs']['dictionary_manifest'])
            assert tasks[role]['dev_ids']==sorted(r['sample_id'] for r in dev)
            for seed,entry in tasks[role]['seeds'].items():
                if role!='MSASL100':
                    present=set(r['class_id'] for r in pool)
                    assert entry['support']['feasible']==(present==set(classes))
                    if present!=set(classes):
                        assert entry['support']['sample_ids']==[]
                        assert entry['support']['missing_classes']==[k for k in classes if k not in present]
                        continue
                    support=entry['support']['sample_ids'];assert len(support)==100
                    for cls,clip in enumerate(support):
                        candidates=[r['sample_id'] for r in pool if r['class_id']==cls]
                        expected=min(candidates,key=lambda key:(hashlib.sha256(
                            f'H007-support-v1|{role}|{seed}|{cls}|{key}'.encode()).hexdigest(),key))
                        assert clip==expected
                else:
                    support=dictionary['supports'][seed]
                prefix=f'H007-class-v1|{role}|{seed}|'
                order=sorted(classes,key=lambda cls:(hashlib.sha256((prefix+str(cls)).encode()).hexdigest(),str(cls)))
                assert entry['class_order']==order
                assert entry['common_known_classes']==order[:10] and entry['common_unknown_classes']==order[50:]
                for size,gallery in entry['galleries'].items():
                    members=set(order[:int(size)])
                    assert gallery['class_ids']==sorted(members)
                    assert gallery['support_ids']==[support[k] for k in sorted(members)]
                    for part,q in pools:
                        actual=entry['membership'][size] if role=='MSASL100' else entry['membership'][size][part]
                        expected=dict(total=len(q),known=sum(x['class_id'] in members for x in q),
                            unknown=sum(x['class_id'] not in members for x in q),
                            common_known_anchor=sum(x['class_id'] in order[:10] for x in q),
                            common_unknown=sum(x['class_id'] in order[50:] for x in q))
                        assert actual==expected;checked_cells+=1
            totals[role]=len(entries)
        run.save('metadata_verification.json',dict(passed=True,source_rows=totals,all_count_cells_exact=True,
            membership_cells_checked=checked_cells,measured_store=checked,feature_arrays_loaded=0,
            quality_scores_computed=0,test_access='none',scope='same-agent independent direct metadata recount, not external review'))


if __name__=='__main__':main()
