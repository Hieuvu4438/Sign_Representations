"""Read existing artifacts only; aggregation never launches experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import yaml
from .utils import atomic_json,read_jsonl,sha256
from .eval.bootstrap import paired_gloss_bootstrap
from .eval.minimal_pairs import score_triplets


def minimal_pairs(config_path,output=None):
    cfg=yaml.safe_load(Path(config_path).read_text())
    if set(cfg)!={'schema_version','annotations','embeddings','index','output','synthetic'} or cfg['schema_version']!=1:
        raise ValueError('Unknown/missing minimal-pairs configuration')
    if not Path(cfg['annotations']).exists():
        result={'status':'BLOCKED_EXTERNAL','reason':'Verified natural minimal-pair annotations unavailable',
                'synthetic':cfg['synthetic'],'annotations_path':cfg['annotations']}
    else:
        x=np.load(cfg['embeddings'],allow_pickle=False)
        vectors={r['video_id']:x[r['row_index']] for r in read_jsonl(cfg['index']) if r['status']=='ok'}
        rows=read_jsonl(cfg['annotations'])
        result=score_triplets(vectors,rows)
        # Multiple anchors/triplets for the same lexical pair share one cluster.
        clusters={}
        annotations={r['triplet_id']:r for r in rows}
        for row in result['per_triplet']:
            ann=annotations[row['triplet_id']]
            key=tuple(sorted((ann['anchor_gloss_id'],ann['negative_gloss_id'])))
            clusters.setdefault(key,[]).append(row['margin'])
        if clusters:
            groups=list(clusters.values());rng=np.random.default_rng(2026)
            draws=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(1000)]
            result.update(margin_ci95=np.quantile([d.mean() for d in draws],[.025,.975]).tolist(),
                          accuracy_ci95=np.quantile([(d>0).mean() for d in draws],[.025,.975]).tolist(),
                          bootstrap_cluster='lexical_pair',bootstrap_clusters=len(groups),bootstrap_replicates=1000)
        result.update(synthetic=cfg['synthetic'],annotations_sha256=sha256(cfg['annotations']))
    atomic_json(output or cfg['output'],result)
    return result


def aggregate(config_path,output=None):
    cfg=yaml.safe_load(Path(config_path).read_text())
    if set(cfg)!={'schema_version','runs','contrasts','require_complete','output'} or cfg['schema_version']!=1:
        raise ValueError('Unknown/missing aggregation configuration')
    table,missing,loaded=[],[],{}
    for item in cfg['runs']:
        if set(item)!={'variant','seed','run_dir','eval_dir'}: raise ValueError('Invalid matrix row')
        run=Path(item['run_dir']);evaluation=Path(item['eval_dir'])
        if not (evaluation/'metrics.json').exists() or (item['variant']!='A0' and not (run/'run_manifest.json').exists()):
            missing.append(item);continue
        metric=json.loads((evaluation/'metrics.json').read_text())
        manifest=json.loads((run/'run_manifest.json').read_text()) if item['variant']!='A0' else None
        if manifest and (manifest['variant']!=item['variant'] or manifest['seed']!=item['seed']): raise ValueError('Run identity differs from matrix')
        if manifest and cfg['require_complete'] and manifest['status']!='complete': raise ValueError('Incomplete training run')
        row={k:item[k] for k in ('variant','seed')}
        row.update({k:metric[k] for k in ('macro_gloss_r1','macro_gloss_r5','micro_r1','r5','mrr','attempted_set_scored_coverage')})
        row.update(successful_updates=manifest['successful_updates'] if manifest else 0,
                   elapsed_seconds=manifest['elapsed_seconds'] if manifest else 0,
                   synthetic=metric['synthetic'],stage=metric['stage'],metrics_sha256=sha256(evaluation/'metrics.json'))
        table.append(row);loaded[(item['variant'],item['seed'])]=(item,manifest,metric)
    if missing and cfg['require_complete']: raise ValueError('Missing planned artifacts: '+str(missing))
    comparisons=[]
    for first,second in cfg['contrasts']:
        seeds=sorted({s for v,s in loaded if v==first}&{s for v,s in loaded if v==second})
        paired=[]
        for seed in seeds:
            ia,ma,a=loaded[(first,seed)];ib,mb,b=loaded[(second,seed)]
            for field in ('protocol_sha256','eval_labels_sha256','stage','synthetic'):
                if a[field]!=b[field]: raise ValueError('Evaluation contrast contract differs: '+field)
            if ma and mb:
                for field in ('init_checkpoint_sha256','cache_manifest_sha256','ssl_manifest_sha256','split_fingerprint','successful_updates','effective_batch'):
                    if ma[field]!=mb[field]: raise ValueError('Controlled contrast differs: '+field)
                if sha256(Path(ia['run_dir'])/'window_plan.jsonl')!=sha256(Path(ib['run_dir'])/'window_plan.jsonl'):
                    raise ValueError('Controlled contrast windows/masks differ')
            delta=paired_gloss_bootstrap(a['per_query'],b['per_query'])
            paired.append({'seed':seed,**delta})
        if paired:
            deltas=[x['mean_delta'] for x in paired]
            comparisons.append({'contrast':second+' - '+first,'per_seed':paired,'mean_delta':float(np.mean(deltas)),
                                'seed_std':float(np.std(deltas,ddof=1)) if len(deltas)>1 else None,
                                'seed_count':len(deltas),'positive_seeds':sum(x>0 for x in deltas),
                                'uncertainty_scope':'Per-seed CIs resample evaluation glosses; seed std describes observed initialization variability'})
    path=Path(output or cfg['output']);path.parent.mkdir(parents=True,exist_ok=True)
    if table:
        with path.open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    else: path.write_text('variant,seed\n')
    result={'status':'complete' if not missing else 'incomplete','metrics_units':'fractions, not percent','table':table,'contrasts':comparisons,'missing':missing,
            'config_sha256':sha256(config_path),'csv_sha256':sha256(path)}
    atomic_json(path.with_suffix('.json'),result)
    return result


def build_report(config_path,output=None):
    cfg=yaml.safe_load(Path(config_path).read_text())
    if set(cfg)!={'schema_version','aggregate','state','limitations','output'} or cfg['schema_version']!=1:
        raise ValueError('Unknown/missing report configuration')
    data=json.loads(Path(cfg['aggregate']).read_text())
    text=['# SIGNREL bounded experiment report','',f"Aggregation status: {data['status']}. Metrics below are percentages.",'',
          '| Variant | Seed | Stage | Macro R@1 | Micro R@1 | Macro R@5 | MRR | Coverage | Updates |',
          '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in data['table']:
        text.append(f"| {row['variant']} | {row['seed']} | {row['stage']} | {100*row['macro_gloss_r1']:.2f} | {100*row['micro_r1']:.2f} | {100*row['macro_gloss_r5']:.2f} | {row['mrr']:.4f} | {100*row['attempted_set_scored_coverage']:.1f} | {row['successful_updates']} |")
    text.extend(['','## Controlled contrasts',''])
    for comparison in data['contrasts']:
        text.append(f"{comparison['contrast']}: mean {100*comparison['mean_delta']:+.2f} percentage points over {comparison['seed_count']} seeds; positive in {comparison['positive_seeds']} seeds.")
        for seed in comparison['per_seed']:
            text.append(f"- Seed {seed['seed']}: {100*seed['mean_delta']:+.2f} pp, paired gloss bootstrap 95% CI [{100*seed['ci95'][0]:+.2f}, {100*seed['ci95'][1]:+.2f}] pp.")
        text.extend(['',comparison['uncertainty_scope'],''])
    text.extend(['## Interpretation','', 'Synthetic fixtures verify software only.' if any(r['synthetic'] for r in data['table']) else
                 'This bounded subset evaluates frozen gloss retrieval. A positive mean alone does not establish a robust method improvement or paper readiness.',
                 '', '## Limitations','']+[f'- {item}' for item in cfg['limitations']])
    text.extend(['','## Implementation state','',Path(cfg['state']).read_text(),'','## Provenance','',f"Aggregate SHA256: `{sha256(cfg['aggregate'])}`; report config SHA256: `{sha256(config_path)}`."])
    path=Path(output or cfg['output']);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(text)+'\n')
    return {'status':'written','output':str(path),'sha256':sha256(path)}
