"""Frozen dev-only instance-retrieval selection, separate from fixed-budget final."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from signcoord.data.manifest import read_manifest,audit_overlap
from signcoord.eval.features import extract_features
from signcoord.eval.retrieval import evaluate_retrieval
from signcoord.provenance import atomic_json,execution_stamp,object_hash,sha256_file,tensor_state_hash
from .checkpoint import atomic_checkpoint,rng_state,restore_rng


def selection_inputs(config):
    evaluation=config['evaluation']
    if not evaluation['selection_enabled']:return None
    paths={key:Path(evaluation['selection_'+key]).resolve() for key in ('gallery_manifest','query_manifest','labels_path')}
    gallery=read_manifest(paths['gallery_manifest'],{'train'})
    query=read_manifest(paths['query_manifest'],{'dev'})
    if set(r['sample_id'] for r in gallery)&set(r['sample_id'] for r in query):
        raise ValueError('selection gallery/query IDs overlap')
    def source_keys(records):
        return ({r['media_sha256'] for r in records},
                {(r['dataset_id'],r['source_group_id']) for r in records},
                {(r['dataset_id'],r['source_video_id']) for r in records},
                {r['duplicate_group_id'] for r in records if r.get('duplicate_group_id') is not None})
    if any(a&b for a,b in zip(source_keys(gallery),source_keys(query))):
        raise ValueError('selection gallery/query media or source overlap')
    overlap=audit_overlap(gallery,query)
    datasets={r['dataset_id'] for r in gallery+query}
    expected='synthetic_fixture' if config['model']['backbone']=='tiny_fixture' else evaluation['primary_dataset']
    if datasets!={expected}:raise ValueError('selection dataset does not match the declared primary dataset')
    labels=json.loads(paths['labels_path'].read_text())
    ids={r['sample_id'] for r in gallery+query}
    if set(labels)!=ids:raise ValueError('selection labels must contain exactly gallery/train and query/dev IDs')
    if any(not isinstance(value,(str,int)) or isinstance(value,bool) for value in labels.values()):
        raise ValueError('selection labels must be explicit string/integer class identities')
    protocol={'policy':'fixed_budget_final_primary_dev_rec1_max_earliest_tie_secondary_v1',
         'metric':'rec1','gallery_split':'train','query_split':'dev','prototype_gallery':False,
         'dataset':expected,'gallery_count':len(gallery),'query_count':len(query),
         'gallery_ordered_ids_hash':object_hash([r['sample_id'] for r in gallery]),
         'query_ordered_ids_hash':object_hash([r['sample_id'] for r in query]),
         'input_hashes':{key:sha256_file(path) for key,path in paths.items()},
         'signer_disjoint_verified':overlap['signer_disjoint_verified'],
         'known_shared_signers':sorted({str(r['signer_id']) for r in gallery if r['signer_id'] is not None}
                                      &{str(r['signer_id']) for r in query if r['signer_id'] is not None})}
    if expected=='asl_citizen' and not overlap['signer_disjoint_verified']:
        raise ValueError('ASL-Citizen selection requires the supplied signer-disjoint train/dev split')
    return {'paths':paths,'protocol':protocol,'protocol_hash':object_hash(protocol),'labels':labels}


class FrozenRetrievalSelection:
    def __init__(self,config,run,inputs):
        self.config,self.run,self.inputs=copy.deepcopy(config),Path(run).resolve(),inputs
        self.state={'protocol_hash':inputs['protocol_hash'],'best_metric':None,'best_update':None,
                    'best_checkpoint_path':str(self.run/'checkpoints/best_dev.pt'),
                    'best_checkpoint_sha256':None,'history':[]}

    def restore(self,state):
        if state is None or state['protocol_hash']!=self.inputs['protocol_hash']:
            raise ValueError('resume selection data/label/protocol hashes changed')
        self.state=copy.deepcopy(state)
        if state['best_checkpoint_sha256'] is not None:
            prior=Path(state['best_checkpoint_path']);destination=self.run/'checkpoints/best_dev.pt'
            if sha256_file(prior)!=state['best_checkpoint_sha256']:
                raise ValueError('resume best-dev checkpoint changed')
            if prior.resolve()!=destination.resolve():
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(prior,destination)
            self.state['best_checkpoint_path']=str(destination)

    def verify_inputs(self):
        for key,path in self.inputs['paths'].items():
            if sha256_file(path)!=self.inputs['protocol']['input_hashes'][key]:
                raise ValueError('frozen selection input changed during training: '+key)

    def evaluate(self,core,update,device):
        """Uses a fresh encoder; preserve all RNG affected by its initialization."""
        snapshot=rng_state(device)
        try:
            return self._evaluate(core,update,device)
        finally:
            restore_rng(snapshot)

    def _evaluate(self,core,update,device):
        self.verify_inputs()
        parent=self.run/'dev_selection';parent.mkdir(parents=True,exist_ok=True)
        attempt=Path(tempfile.mkdtemp(prefix=f'update_{update:08d}_',dir=parent))
        encoder_state={'encoder.'+name:value.detach() for name,value in core.encoder.state_dict().items()}
        encoder_hash=tensor_state_hash(core.encoder.state_dict())
        temporary=attempt/'encoder_snapshot.pt'
        atomic_checkpoint(temporary,{'schema_version':'signcoord_checkpoint_v1','microstep_boundary':0,
                          'model_state':encoder_state,'resolved_config':self.config,
                          'optimizer_update':update,'encoder_state_hash':encoder_hash})
        began=time.perf_counter()
        gallery_report=extract_features(temporary,self.inputs['paths']['gallery_manifest'],attempt/'gallery.npz',
                                        str(device),include_sequence=False)
        query_report=extract_features(temporary,self.inputs['paths']['query_manifest'],attempt/'query.npz',
                                      str(device),include_sequence=False)
        labels=self.inputs['labels']
        with np.load(attempt/'gallery.npz',allow_pickle=False) as gallery,np.load(attempt/'query.npz',allow_pickle=False) as query:
            gids,qids=gallery['sample_ids'].tolist(),query['sample_ids'].tolist()
            result=evaluate_retrieval(gallery['z'],query['z'],[labels[i] for i in gids],[labels[i] for i in qids],
                                      gids,qids,query['groups'].tolist())
        self.verify_inputs()
        if not np.isfinite(result['metrics']['rec1']):raise ValueError('nonfinite dev selection metric')
        result.update(**execution_stamp(),status='passed',optimizer_update=update,
                      encoder_state_hash=encoder_hash,selection_protocol_hash=self.inputs['protocol_hash'],
                      selection_protocol=self.inputs['protocol'],checkpoint_snapshot_sha256=sha256_file(temporary),
                      gallery_feature_metadata=gallery_report,query_feature_metadata=query_report,
                      test_evaluation=False,intermediate_features_retained=self.config['evaluation']['selection_keep_features'],
                      elapsed_frozen_evaluation_seconds=time.perf_counter()-began)
        atomic_json(attempt/'retrieval.json',result)
        temporary.unlink()
        if not self.config['evaluation']['selection_keep_features']:
            for name in ('gallery.npz','queries.npz','query.npz','gallery.npz.json','query.npz.json'):
                path=attempt/name
                if path.exists():path.unlink()
        return {'optimizer_update':update,'metric':result['metrics']['rec1'],'encoder_state_hash':encoder_hash,
                'evaluation_report':str(attempt/'retrieval.json'),'evaluation_report_sha256':sha256_file(attempt/'retrieval.json'),
                'selection_protocol_hash':self.inputs['protocol_hash']}

    def record(self,result):
        if result['selection_protocol_hash']!=self.state['protocol_hash']:
            raise ValueError('dev result selection protocol mismatch')
        if self.state['history'] and result['optimizer_update']<=self.state['history'][-1]['optimizer_update']:
            raise ValueError('selection updates must increase; cannot silently replay an older checkpoint')
        self.state['history'].append(result)
        improved=self.state['best_metric'] is None or result['metric']>self.state['best_metric']
        if improved:
            self.state['best_metric']=result['metric'];self.state['best_update']=result['optimizer_update']
        return improved

    def save_history(self):
        atomic_json(self.run/'checkpoint_selection.json',{'protocol':self.inputs['protocol'],**self.state})
