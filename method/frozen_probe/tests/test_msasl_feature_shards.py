"""Fixtures for wrong cache identities and complete-only final shard boundaries."""
import copy
import numpy as np
from msasl_feature_shards import reuse_cost_vectors,shard_plan


def run_tests():
    manifest=[dict(sample_id=id,source_sha256='a'*64,clock_pts_ticks_sha256='b'*64,all_resized_RGB_sha256='c'*64,
                   first_native16_RGB_sha256='d'*64,complete_stride2_windows=1,selected_native_frames=16) for id in ('A','B')]
    rows=[dict(sample_id=x['sample_id'],input=dict(x,native_frames=16),windows=1) for x in manifest]
    vectors=np.array([[1,0,0],[0,1,0]],dtype=np.float64)
    result=reuse_cost_vectors(rows,vectors,manifest,3);assert list(result)==['A','B'] and np.array_equal(result['B'][0],[0,1,0])
    for changed in ('source_sha256','clock_pts_ticks_sha256','all_resized_RGB_sha256','first_native16_RGB_sha256'):
        wrong=copy.deepcopy(rows);wrong[0]['input'][changed]='0'*64
        try:reuse_cost_vectors(wrong,vectors,manifest,3)
        except ValueError:pass
        else:raise AssertionError('Mismatched cache accepted '+changed)
    for badrows,badvalues in ((rows+rows,vectors),(rows[:1],vectors),(rows,vectors.astype(np.float32)),(rows,np.zeros_like(vectors))):
        try:reuse_cost_vectors(badrows,badvalues,manifest,3)
        except ValueError:pass
        else:raise AssertionError('Duplicate/incomplete/invalid reused output accepted')
    plan=shard_plan(5,2,3)
    assert [(x['start'],x['stop']) for x in plan]==[(0,2),(2,4),(4,5)]
    assert [x['spec']['shape'] for x in plan]==[[2,3],[2,3],[1,3]]
    full=shard_plan(1037);assert len(full)==42 and full[-1]['start']==1025 and full[-1]['stop']==1037 and full[-1]['spec']['shape']==[12,768]
    return dict(passed=True,groups=['strict_cost_vector_ID_input_shape_norm_guards','complete_only_partial_final_shards'],GPU_hours=0,
                scope='Software cache/shard integrity only, no model or taskquality')
