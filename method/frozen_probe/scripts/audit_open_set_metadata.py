"""Registered metadata-only feature census and prospective calibration split."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_cpu_context import BoundedCPUContext
from cache import file_sha, object_sha
from open_set_metadata import (asl_rows,msasl_rows,counts,calibration_partition,
                               choose_support,hash_order,nested_task,membership)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser()
    for name in ('config','run-id','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config
        if c['stage']=='tests':
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
            from test_open_set_metadata import run_tests
            run.save('metadata_tests.json',run_tests());return
        all_rows,summary,tasks={}, {}, {}
        for role,spec in c['asl_inputs'].items():
            report=read(spec['report']);prior=read(spec['identity_config']);cohort=read(prior['cohort_manifest'])
            m=read(report['cache_manifest']);status=read(Path(spec['runtime'])/'status.json')
            if (status['status']!='completed' or file_sha(Path(spec['runtime'])/'config.json')!=status['config_sha256'] or not report['passed'] or
                    file_sha(report['cache_manifest'])!=report['cache_manifest_sha256'] or
                    m['identity']!=prior['cache_identity'] or m['identity_sha256']!=object_sha(m['identity'])):
                raise ValueError('ASL terminal report/identity binding differs')
            rows=asl_rows(cohort,m,Path(report['cache_manifest']).parent,role)
            if len(rows)!=report['source_count']:raise ValueError('Report coverage differs')
            for r in rows:
                f=Path(r['feature']['path'])
                if f.is_symlink() or f.stat().st_size!=r['feature']['bytes'] or file_sha(f)!=r['feature']['sha256']:
                    raise ValueError('Changed ASL feature payload (hash-only read)')
            classes=list(range(len(cohort['classes'])))
            train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='val']
            signers=sorted({r['signer'] for r in train})
            selected=hash_order(signers,'H007-calibration-v1|'+role)[:max(1,len(signers)//4)]
            calibration,pool=calibration_partition(rows,selected)
            summary[role]=dict(full=counts(rows,classes),train=counts(train,classes),dev=counts(dev,classes),
                calibration=counts(calibration,classes),gallery_pool=counts(pool,classes),
                calibration_signers=selected,source_group_scope='exact source SHA, physical/session identity unknown',
                feature_contract=m['identity']['contract'],feature_payload_bytes=m['payload_bytes'])
            tasks[role]=dict(calibration_ids=sorted(r['sample_id'] for r in calibration),
                             gallery_pool_ids=sorted(r['sample_id'] for r in pool),dev_ids=sorted(r['sample_id'] for r in dev),seeds={})
            for seed in c['support_seeds']:
                chosen=choose_support(pool,classes,f'H007-support-v1|{role}|{seed}')
                entry=dict(support=chosen)
                if chosen['feasible']:
                    order=hash_order(classes,f'H007-class-v1|{role}|{seed}')
                    galleries=nested_task(rows,chosen['sample_ids'],classes,order,c['gallery_sizes'])
                    entry.update(class_order=order,galleries=galleries,common_known_classes=order[:10],common_unknown_classes=order[50:])
                    entry['membership']={n:{part:membership(q,g['class_ids'],order[:10],order[50:])
                        for part,q in [('calibration',calibration),('dev',dev)]} for n,g in galleries.items()}
                tasks[role]['seeds'][str(seed)]=entry
            all_rows[role]=rows
        spec=c['msasl_inputs'];cfg=read(Path(spec['runtime'])/'config.json');status=read(Path(spec['runtime'])/'status.json')
        if (status['status']!='completed' or status['exit_code']!=0 or
                file_sha(Path(spec['runtime'])/'config.json')!=status['config_sha256']):
            raise ValueError('MS-ASL incomplete/changed extraction')
        items=read(cfg['extraction_manifest']);saved=[];refs={};next_start=0
        for shard in cfg['shards']:
            if shard['start']!=next_start:raise ValueError('Noncontiguous extraction shards')
            part=read(Path(spec['runtime'])/shard['records_name'])
            if len(part)!=shard['stop']-shard['start']:raise ValueError('Incomplete shard records')
            for j,r in enumerate(part):
                if r['sample_id'] in refs:raise ValueError('Duplicate shard sample')
                refs[r['sample_id']]=dict(path=str(Path(spec['runtime'])/shard['array_name']),row=j)
            saved.extend(part);next_start=shard['stop']
        if next_start!=len(items) or [r['sample_id'] for r in items]!=cfg['sample_ids']:
            raise ValueError('Full extraction membership differs')
        rows=msasl_rows(items,saved,refs);classes=list(range(100));dictionary=read(spec['dictionary_manifest'])
        q=[r for r in rows if r['split']=='val'];train=[r for r in rows if r['split']=='train']
        if sorted(r['sample_id'] for r in q)!=dictionary['query_ids']:raise ValueError('MS-ASL query join differs')
        all_rows['MSASL100']=rows
        summary['MSASL100']=dict(full=counts(rows,classes),train=counts(train,classes),dev=counts(q,classes),
            calibration='not proposed: cached train clips are existing gallery support union',
            feature_contract='MSASL_ESTCLK_v1 bodycrop/native16 stride2 full_activity FP64 L2',
            source_group_scope='publisher parent video ID; physical/session identity unknown')
        tasks['MSASL100']=dict(dev_ids=dictionary['query_ids'],seeds={})
        for seed in c['support_seeds']:
            ids=dictionary['supports'][str(seed)];order=hash_order(classes,f'H007-class-v1|MSASL100|{seed}')
            galleries=nested_task(rows,ids,classes,order,c['gallery_sizes'])
            tasks['MSASL100']['seeds'][str(seed)]=dict(class_order=order,galleries=galleries,
                common_known_classes=order[:10],common_unknown_classes=order[50:],
                membership={n:membership(q,g['class_ids'],order[:10],order[50:]) for n,g in galleries.items()})
        pair=[all_rows[x] for x in ('ASLCitizen_base','ASLCitizen_target')]
        cross={field:sorted({r[field] for r in pair[0]} & {r[field] for r in pair[1]})
               for field in ('sample_id','source_sha256','class_name','signer')}
        summary['cross_ASLCitizen_metadata_intersections']=cross
        summary['cross_corpus_physical_signer_identity']='unknown; do not join numeric IDs across releases'
        run.save('metadata_rows.json',all_rows)
        run.save('candidate_partitions.json',tasks)
        run.save('metadata_census.json',dict(passed_metadata_joins=True,roles=summary,feature_arrays_loaded=0,
            quality_scores_computed=0,thresholds_fitted=0,test_access='none',new_encoder_calls=0,
            scope='prospective metadata feasibility only; no quality/novelty/pilot claim'))


if __name__=='__main__':main()
