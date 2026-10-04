"""First registered MS-ASL scores with frozen saved geometry and full cohort."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from bounded_cpu_context import BoundedCPUContext
from bounded_artifacts import verify_store
from msasl_saved_features import load_complete_features
from msasl_geometry_eval import frozen_geometry, score_frozen_dictionary, EVAL_MODES
from resources import scratch_preflight


def main():
    p=argparse.ArgumentParser()
    for key in ('config','run-id','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    with BoundedCPUContext(a.output_dir,a.run_id,a.config,a.seed) as run:
        c=run.config
        resource=scratch_preflight('/dev/shm',c['workspace_plan_bytes'])
        directory=Path(c['dictionary_runtime']);base=Path(c['base_runtime'])
        dictionary=json.loads((directory/'dictionary_manifest.json').read_text())
        items=json.loads((directory/'extraction_manifest.json').read_text())
        values,records,feature_check=load_complete_features(c['feature_runtime'],items)
        assert values.shape==(c['expected_clips'],c['dimension'])==(1037,768)
        assert dictionary['support_seeds']==c['support_seeds']==[1701,1702,1703]
        assert dictionary['class_ids']==list(range(c['expected_classes']))==list(range(100))
        assert len(dictionary['query_ids'])==c['expected_queries']==748
        assert list(EVAL_MODES)==c['modes']
        assert json.loads((base/'status.json').read_text())['status']=='completed'
        base_check=verify_store(base)
        load=lambda n:np.load(base/n,allow_pickle=False)
        mean=load('mean.npy');bases={k:load('basis_'+k+'.npy') for k in ('R','T')}
        spectra={k:load('spectrum_'+k+'.npy') for k in ('R','T')}
        projected=frozen_geometry(values,mean,bases,spectra)
        arrays,metadata,summary=score_frozen_dictionary(projected,items,dictionary,
            c['primary_gain_floor'],c['bootstrap_replicates'],a.seed)
        assert len(set(metadata['participants']))==c['expected_signers']==19
        assert metadata['prediction_rows']==13464
        for name,array in arrays.items():run.save_array(name,array,c['array_specs'][name])
        run.save('shared_metadata.json',dict(ranking=metadata,supports=dictionary['supports'],
            target_loaded_ids=[x['sample_id'] for x in items], feature_runtime=c['feature_runtime'],base_runtime=c['base_runtime'],
            score_columns='Class IDs 0..99 in ascending gallery order; rows match ranking.row_order',
            pooled_vector_provenance='Completed42shards/all1037native inputs; seven bound costvectors reused',
            geometry='Exact saved ASL Citizen mean/R,T bases/R,T ascending spectra; no target mean/covariance/eigenfit'))
        summary.update(resource=resource,feature_check=feature_check,base_check=base_check,GPU_hours=0,
            new_encoder_calls=0,target_fit_calls=0,test_access='none; MS-ASL validation',
            versions=dict(numpy=np.__version__),unresolved='Encoder pretraining membership, lexeme overlap, physical signer identity and novelty remain unverified')
        run.save('geometry_eval.json',summary)


if __name__=='__main__':main()
