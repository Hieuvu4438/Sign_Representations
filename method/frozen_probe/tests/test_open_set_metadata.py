"""Hand-counted join, leakage and missing-vocabulary oracles; no science pilot."""
from copy import deepcopy
from open_set_metadata import (validate_rows, calibration_partition, choose_support,
                               nested_task, membership, asl_rows, msasl_rows)


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError('Unsafe metadata accepted')


def row(key, cls, signer, split='train', source=None):
    return dict(sample_id=key, class_id=cls, signer=signer, split=split,
                source_group=source or key, source_sha256=source or key)


def run_tests():
    rows=[row('a',0,'A'),row('b',1,'A'),row('c',0,'B'),row('d',1,'B'),
          row('e',0,'C','val'),row('f',1,'C','val')]
    calibration,pool=calibration_partition(rows,['A'])
    assert [r['sample_id'] for r in calibration]==['a','b']
    assert [r['sample_id'] for r in pool]==['c','d']
    assert choose_support(pool,[0,1],'fixed')['sample_ids']==['c','d']
    tasks=nested_task(rows,['c','d'],[0,1],[1,0],[1,2])
    assert tasks=={'1':{'class_ids':[1],'support_ids':['d']},
                   '2':{'class_ids':[0,1],'support_ids':['c','d']}}
    assert membership(rows[-2:],[1],[1],[0])==dict(total=2,known=1,unknown=1,
                                                    common_known_anchor=1,common_unknown=1)
    rejected(lambda:membership(rows[-2:],[0,1],[1],[0]))
    # One missing class must not turn into a smaller vocabulary or a selected retry.
    assert choose_support([rows[2]],[0,1],'fixed')==dict(feasible=False,missing_classes=[1],sample_ids=[])
    bad=deepcopy(rows);bad[2]['source_group']='a'
    rejected(lambda:calibration_partition(bad,['A']))
    bad=deepcopy(rows);bad[-1]['signer']='A';rejected(lambda:validate_rows(bad))
    bad=deepcopy(rows);bad[-1]['split']='test';rejected(lambda:validate_rows(bad))
    rejected(lambda:validate_rows(rows+[rows[0]]))
    bad=deepcopy(rows);bad[0]['class_id']=True;rejected(lambda:validate_rows(bad))
    rejected(lambda:nested_task(rows,['c','e'],[0,1],[0,1],[1]))
    samples=[dict(sample_id=x['sample_id'],split=x['split'],class_id=x['class_id'],
                  participant_id=x['signer'],source_sha256=x['source_sha256'],
                  gloss=['ZERO','ONE'][x['class_id']]) for x in rows]
    entries=[{**{k:s[k] for k in ('sample_id','split','class_id','participant_id','source_sha256')},
              'shape':[2,768],'status':'completed','file':s['sample_id']+'.npz',
              'sha256':'h','bytes':8} for s in samples]
    cohort=dict(passed=True,samples=samples,classes=['ZERO','ONE'])
    manifest=dict(status='completed',samples=entries,payload_bytes=48)
    normalized=asl_rows(cohort,manifest,'/fixture','ASL')
    assert len(normalized)==6 and sum(r['windows'] for r in normalized)==12
    bad=deepcopy(manifest);bad['samples'][0]['participant_id']='different'
    rejected(lambda:asl_rows(cohort,bad,'/fixture','ASL'))
    bad=deepcopy(manifest);bad['samples'][0]['file']='../escaped.npz'
    rejected(lambda:asl_rows(cohort,bad,'/fixture','ASL'))
    bad=deepcopy(manifest);bad['samples'][1]['file']=bad['samples'][0]['file']
    rejected(lambda:asl_rows(cohort,bad,'/fixture','ASL'))
    bad=deepcopy(manifest);bad['payload_bytes']=49
    rejected(lambda:asl_rows(cohort,bad,'/fixture','ASL'))
    # MS-ASL fixture has two independent native source rows and matching saved joins.
    items=[dict(sample_id=x['sample_id'],split=x['split'],class_id=x['class_id'],
                signer_id=x['signer'],parent_id=x['source_group'],source_sha256=x['source_sha256'],
                input_contract='MSASL_ESTCLK_v1',complete_stride2_windows=2,selected_native_frames=18,
                clock_pts_ticks_sha256='p',all_resized_RGB_sha256='r',first_native16_RGB_sha256='f')
           for x in [rows[0],rows[-1]]]
    records=[dict(sample_id=i['sample_id'],windows=2,pooled_vector_sha256='v',
                  input={**{k:i[k] for k in ('source_sha256','clock_pts_ticks_sha256',
                       'all_resized_RGB_sha256','first_native16_RGB_sha256')},
                         'source_identity_join':True,'native_frames':18}) for i in items]
    refs={i['sample_id']:dict(path='pooled.npy',row=j) for j,i in enumerate(items)}
    assert [r['sample_id'] for r in msasl_rows(items,records,refs)]==['a','f']
    bad=deepcopy(records);bad[0]['input']['source_sha256']='wrong'
    rejected(lambda:msasl_rows(items,bad,refs))
    rejected(lambda:msasl_rows(items,records,{}))
    return dict(passed=True,manual_groups=5,scope='software joins/leakage/coverage/membership only',
                groups=['fixed partition/support/nesting/counts','missing class and common-query drift',
                        'split/source/signer/type guards','ASL manifest joins/path/accounting',
                        'MS-ASL native metadata/shard identity'])
