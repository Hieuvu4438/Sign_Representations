"""Repair owned feature-report provenance from verified trained checkpoints."""
import argparse
import copy
import json
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'signrel/config.py').exists())
sys.path.insert(0,str(ROOT))
import torch
from signrel.data.manifest import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feature-root',default='features/controlled_pilot')
    p.add_argument('--out-root',default='reports/provenance_repairs')
    args = p.parse_args()
    cache,records = {},[]
    out = Path(args.out_root)
    (out/'original_reports').mkdir(parents=True,exist_ok=True)
    for path in sorted(Path(args.feature_root).rglob('*.report.json')):
        original = path.read_text()
        meta = json.loads(original)
        checkpoint = meta.get('arguments',{}).get('checkpoint')
        if not checkpoint:
            continue
        checkpoint = Path(checkpoint).resolve()
        if ROOT/'runs' not in checkpoint.parents:
            raise ValueError('Repair only locally generated owned run checkpoints')
        if checkpoint not in cache:
            digest = sha256(checkpoint)
            state = torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            if 'student_encoder' not in state or 'upstream_init' not in state:
                raise ValueError('Not a compatible trained encoder checkpoint')
            cache[checkpoint] = {'upstream_init':copy.deepcopy(state['upstream_init']),
                'restored_checkpoint':{'path':str(checkpoint),'sha256':digest,'global_step':state['global_step'],
                    'source_hash':state['source_hash'],'component':'student_encoder','strict_load':True}}
            del state
        origin = cache[checkpoint]
        assert meta['checkpoint_sha256'] == origin['restored_checkpoint']['sha256'], 'Report checkpoint hash mismatch'
        feature = path.with_name(path.name.removesuffix('.report.json')+'.npz')
        feature_hash = sha256(feature)
        before = sha256(path)
        if meta.get('upstream_init') != origin['upstream_init'] or meta.get('restored_checkpoint') != origin['restored_checkpoint']:
            backup = out/'original_reports'/path.name
            if backup.exists() and backup.read_text() != original:
                raise FileExistsError('An unrelated earlier backup exists')
            backup.write_text(original)
            meta['constructor_upstream_init'] = meta.get('upstream_init')
            meta.update(copy.deepcopy(origin))
            meta['metadata_repair'] = {'version':1,'source_sha256':sha256(__file__),'original_report_sha256':before,
                'feature_sha256':feature_hash,'reason':'Constructor skipped upstream weights before strict trained-encoder restoration; report original initialization from verified training checkpoint'}
            temporary = path.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(meta,indent=2,allow_nan=False))
            temporary.replace(path)
            assert sha256(feature) == feature_hash, 'Feature bytes unexpectedly changed'
        records.append({'report':str(path),'original_report_sha256':before,'report_sha256':sha256(path),
                        'feature_sha256':feature_hash,'checkpoint_sha256':origin['restored_checkpoint']['sha256']})
    (out/'summary.json').write_text(json.dumps({'source_sha256':sha256(__file__),'records':records,
        'count':len(records),'scope':'Report metadata only; feature arrays, metric reports and training checkpoints remain byte-identical'},indent=2))
    print(json.dumps({'verified_reports':len(records),'checkpoints':len(cache)}))


if __name__ == '__main__':
    main()
