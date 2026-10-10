"""Close eight completed bounded stores and descriptive exports, no new scores."""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bounded_artifacts import verify_store


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    r=Path('method/research_lab/records');roots={};runs={};bindings={}
    keys=('MSFEATTEST001','MSFEATCOST001','MSFULLTEST001','MSEVALTEST001','MSFULLFEAT001','MSEVAL001','MSEVALVERIFY001','MSREPORT001')
    for key in keys:
        reg=json.loads((r/f'experiments/{key}_registration.json').read_text());root=Path(reg['output_dir']);roots[key]=root
        c=json.loads((root/'config.json').read_text());s=json.loads((root/'status.json').read_text())
        assert s['status']=='completed' and s['exit_code']==0 and s['run_id']==reg['run_id']
        assert sha(reg['config'])==sha(root/'config.json')==reg['config_sha256']==s['config_sha256']
        for path,digest in c['input_hashes'].items():
            if path in bindings:assert bindings[path]==digest
            bindings[path]=digest
        store=verify_store(root);manifest=json.loads((root/'artifact_manifest.json').read_text())
        assert manifest['plan']['file_caps']=={**c['artifact_file_caps'],'artifact_manifest.json':c['artifact_manifest_cap']}
        assert manifest['plan']['ceiling_bytes']==reg['durable_bytes_ceiling']==c['artifact_ceiling_bytes']
        for path,digest in s['output_hashes'].items():assert sha(root/path)==digest,path
        runs[key]=dict(run_id=s['run_id'],status=s['status'],pid=s['pid'],start_utc=s['start_utc'],end_utc=s['end_utc'],
            wall_seconds=s['wall_seconds'],GPU_seconds=s['GPU_hours']*3600,store=store,
            config_sha256=reg['config_sha256'],status_sha256=sha(root/'status.json'),manifest_sha256=sha(root/'artifact_manifest.json'))
    for path,digest in bindings.items():assert sha(path)==digest,path
    proof=json.loads((roots['MSEVALVERIFY001']/'geometry_eval_verification.json').read_text());assert proof['passed'] and proof['original_space_fullrank_rows']==13464
    report=json.loads((roots['MSEVAL001']/'geometry_eval.json').read_text());summary=json.loads((roots['MSREPORT001']/'report_summary.json').read_text())
    rows=list(csv.DictReader((roots['MSREPORT001']/'metrics.csv').open(newline='')));assert len(rows)==24
    for row in rows:
        m,g=row['mode'],row['gallery'];assert row['query_count']=='748' and row['class_count']=='100'
        metrics=report['mean_over_fixed_galleries'][m] if g=='mean_fixed_three' else report['metrics'][m][g]
        for kind in ('micro','macro'):
            for metric in ('dcg','mrr','top1','top5'):assert float(row[kind+'_'+metric])==metrics[kind][metric]
    assert summary['primary']==report['primary'] and summary['primary']['passed']
    assert summary['RR_trails_raw_every_gallery'] and summary['RR_trails_centered_every_gallery']
    means=report['mean_over_fixed_galleries'];assert summary['RR_minus_raw_DCG']==means['RR']['micro']['dcg']-means['raw']['micro']['dcg']
    assert summary['RR_minus_centered_DCG']==means['RR']['micro']['dcg']-means['centered']['micro']['dcg']
    # Actual draw/list validation is transparent reporting, not an external review.
    import numpy as np
    meta=json.loads((roots['MSEVAL001']/'shared_metadata.json').read_text())['ranking']
    ranks=np.load(roots['MSEVAL001']/'ranks.npy',allow_pickle=False).reshape(6,3,748,100)
    labels=np.asarray(meta['labels']);rr=ranks[0,0,:,0]==labels;raw=ranks[4,0,:,0]==labels
    pools={'shared_success':np.flatnonzero(rr & raw),'shared_failure':np.flatnonzero(~rr & ~raw),
        'RR_only_correct':np.flatnonzero(rr & ~raw),'raw_only_correct':np.flatnonzero(~rr & raw)}
    cases=json.loads((roots['MSREPORT001']/'descriptive_cases.json').read_text());rng=np.random.default_rng(1701)
    for name,pool in pools.items():
        ids=sorted(rng.choice(pool,size=min(3,len(pool)),replace=False).tolist());case=cases['cases'][name]
        assert case['eligible_bucket_count']==len(pool) and [x['sample_id'] for x in case['sampled']]==[meta['query_ids'][i] for i in ids]
        for qi,entry in zip(ids,case['sampled']):
            assert entry['class_id']==int(labels[qi]) and entry['official_signer_id']==meta['participants'][qi]
            for mi,mode in enumerate(meta['modes']):
                for si,seed in enumerate(meta['support_seeds']):assert entry['ranks'][mode][str(seed)]==int(np.flatnonzero(ranks[mi,si,qi]==labels[qi])[0])+1
    sys.path.insert(0,'method/stream_reliability');from gpu_accounting import inspect_effective_budget
    budget=inspect_effective_budget(json.loads((roots['MSFULLFEAT001']/'config.json').read_text()))
    output=dict(verified_utc=datetime.now(timezone.utc).isoformat(),passed=True,runs=runs,current_input_bindings_checked=len(bindings),
        effective_budget=budget,total_captured_wall_seconds=sum(x['wall_seconds'] for x in runs.values()),
        total_charged_GPU_seconds=sum(x['GPU_seconds'] for x in runs.values()),total_logical_bytes=sum(x['store']['logical_bytes'] for x in runs.values()),
        primary=report['primary'],RR_trails_raw_and_centered_every_gallery=True,
        full_rank_rows_verified=13464,maximum_original_space_score_error=proof['maximum_score_error'],
        table_rows_verified=24,case_bucket_counts={k:len(v) for k,v in pools.items()},
        source_sha256=sha(__file__),scope='Same-agent current bindings/all eight bounded stores/verified scores and descriptive table/case recount. Figure visually inspected separately. No new encoder, fitting, statistical hypothesis or external scientificreview.')
    with (r/'inventory/MSASL_QUALITY_CYCLE_BINDING_VERIFICATION.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps({k:v for k,v in output.items() if k not in ('runs','effective_budget','primary')},indent=2))


if __name__=='__main__':main()
