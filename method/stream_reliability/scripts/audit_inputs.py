"""Four deterministic train/dev raw keypoint schemas; no GPU/reliability labels."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO/'method/frozen_probe'))
sys.path.insert(0, str(ROOT))
from cache import file_sha
from run_context import RunContext


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import numpy as np
        allowed = {('numpy', 'dtype'):np.dtype, ('numpy', 'ndarray'):np.ndarray,
                   ('numpy.core.multiarray', '_reconstruct'):np._core.multiarray._reconstruct,
                   ('numpy._core.multiarray', '_reconstruct'):np._core.multiarray._reconstruct}
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported raw-input pickle global: {module}.{name}')
        return allowed[module, name]


def describe(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'fiu' or value.size > 2000000:
            raise ValueError('Bounded numeric array required')
        return dict(kind='array', shape=list(value.shape), dtype=str(value.dtype),
                    finite=bool(np.isfinite(value).all()), min=float(value.min()) if value.size else None,
                    max=float(value.max()) if value.size else None)
    if isinstance(value, list):
        rows = [describe(x) for x in value]
        shapes = sorted(set(tuple(row.get('shape', [])) for row in rows))
        return dict(kind='list', length=len(rows), element_shapes=[list(s) for s in shapes],
                    element_dtypes=sorted(set(row.get('dtype','unknown') for row in rows)),
                    finite=all(row.get('finite',False) for row in rows),
                    first=rows[0] if rows else None,
                    global_min=min(row['min'] for row in rows if row.get('min') is not None) if rows else None,
                    global_max=max(row['max'] for row in rows if row.get('max') is not None) if rows else None)
    raise ValueError('Unsupported raw field; preserve attempt rather than infer semantics')


def main():
    parser = argparse.ArgumentParser()
    for name in ('config','output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with RunContext(args.output_dir,args.run_id,args.config,args.seed) as run:
        for name,digest in config['input_hashes'].items():
            if file_sha(REPO/name) != digest:raise ValueError('Bound source/cohort/audit changed')
        cohort = json.loads((REPO/config['cohort_manifest']).read_text())
        videos = json.loads((REPO/config['video_audit']).read_text())
        indexed={s['sample_id']:s for s in cohort['samples']}
        native={s['sample_id']:s for s in videos['samples']}
        selected=[]
        for split in ('train','val'):
            selected.extend(sorted(k for k,s in indexed.items() if s['split']==split)[:2])
        if selected != config['sample_ids']:raise ValueError('Fixed train/dev-only schema sample changed')
        rows,logical=[],0
        for key in selected:
            path=Path(config['keypoint_root'])/(Path(key).stem+'.pkl')
            logical+=path.stat().st_size
            if path.stat().st_size>config['max_file_bytes'] or logical>config['max_total_bytes']:
                raise ValueError('Raw input audit byte ceiling exceeded')
            data=path.read_bytes();value=ArrayUnpickler(io.BytesIO(data)).load()
            if not isinstance(value,dict) or not value or len(value)>12:
                raise ValueError('Bounded named raw fields required')
            fields={name:describe(item) for name,item in value.items()}
            rows.append(dict(sample_id=key,split=indexed[key]['split'],participant_id=indexed[key]['participant_id'],
                             raw_path=str(path),raw_bytes=len(data),raw_sha256=hashlib.sha256(data).hexdigest(),
                             native_frames=native[key]['frames'],fields=fields,
                             stream_timestamp_metadata_keys=[n for n in value if any(t in n.lower() for t in ('time','fps','pts'))],
                             frame_count_agreement={name:details.get('length',details.get('shape',[None])[0] if details.get('shape') else None)==native[key]['frames'] for name,details in fields.items()},
                             coordinate_ontology='unknown; array dimensions do not establish landmark naming',
                             score_semantics='unknown; key name scores is not calibrated failure ground truth',
                             preprocessing_provenance='unknown; local filename/schema cannot establish author MediaPipe pipeline'))
        external=args.output_dir/'upstream_source_snapshot';external.mkdir()
        upstream_hashes={}
        for name in config['upstream_files']:
            path=REPO/name;upstream_hashes[name]=file_sha(path)
            dest=external/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        assets=[]
        for name in config['model_assets']:
            path=REPO/name;assets.append(dict(path=name,exists=path.is_file(),bytes=path.stat().st_size if path.is_file() else None,
                                             release_mapping='unknown; existence only, no model load/hash in this input-only audit'))
        run.save('input_audit.json',dict(run_id=args.run_id,samples=rows,N=4,source_read_bytes=logical,
                 upstream_code_hashes=upstream_hashes,model_assets=assets,
                 upstream_input_interface=dict(face_dim=384,left_hand_dim=384,right_hand_dim=384,body_dim=14,
                     inputs='three separately extracted DINO crop arrays plus normalized body array; requires explicit synchronization',
                     reference='pinned SHuBERT source, default config; not checkpoint-state/config parity'),
                 SHuBERT_ready_four_streams=False,natural_failure_labels_verified=False,GPU_hours=0,
                 scope='raw train/dev schema/provenance audit only; no embedding extraction, classifier, threshold or natural failure labeling',
                 next_dependencies=['identify raw coordinate ontology/extractor provenance','verify timestamps/crop/handedness mapping',
                     'produce or identify author-compatible four streams','verify score/missingness meanings and masks',
                     'audit strict checkpoint state/environment before any model parity']))
        run.status['output_hashes']={'input_audit.json':file_sha(run.output/'input_audit.json')}


if __name__ == '__main__':main()
