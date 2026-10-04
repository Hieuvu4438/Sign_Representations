"""Complete-only pooled-vector shards and strict reuse of registered cost inputs."""
import numpy as np
from bounded_artifacts import array_spec


def shard_plan(rows,chunk=25,dimension=768):
    if any(type(v) is not int or v<1 for v in (rows,chunk,dimension)):raise ValueError('Invalid feature shard dimensions')
    return [dict(index=i,start=start,stop=min(start+chunk,rows),
                 array_name=f'pooled_{i:03d}.npy',records_name=f'features_{i:03d}.json',
                 spec=array_spec((min(chunk,rows-start),dimension),'<f8')) for i,start in enumerate(range(0,rows,chunk))]


def reuse_cost_vectors(records,vectors,manifest,dimension=768):
    byid={x['sample_id']:x for x in manifest};ids=[x['sample_id'] for x in records]
    if len(byid)!=len(manifest) or not ids or len(set(ids))!=len(ids) or not set(ids)<=set(byid):
        raise ValueError('Cost reuse ID completeness/uniqueness failed')
    values=np.asarray(vectors)
    if values.dtype!=np.float64 or values.shape!=(len(records),dimension) or not np.isfinite(values).all():
        raise ValueError('Cost vectors shape/dtype/finite contract failed')
    if np.max(np.abs(np.linalg.norm(values,axis=1)-1))>1e-12:raise ValueError('Cost pooled vectors are not unit length')
    reused={}
    for row,value in zip(records,values):
        item=byid[row['sample_id']]
        for key in ('source_sha256','clock_pts_ticks_sha256','all_resized_RGB_sha256','first_native16_RGB_sha256'):
            if row['input'][key]!=item[key]:raise ValueError('Cost reuse input identity changed '+key)
        if row['windows']!=item['complete_stride2_windows'] or row['input']['native_frames']!=item['selected_native_frames']:
            raise ValueError('Cost reuse native window/frame count changed')
        reused[row['sample_id']]=(value.copy(),row)
    return reused
