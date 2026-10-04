"""Bounded variable-length timestamp NPY payload; never input-frame padding."""
import io
import numpy as np
from bounded_artifacts import array_spec


def native_pts_bytes(values,max_count=240):
    array=np.asarray(values,dtype='<f8')
    if array.ndim!=1 or not 16<=len(array)<=max_count or not np.isfinite(array).all() or np.any(np.diff(array)<=0):
        raise ValueError('Invalid bounded native timestamp vector')
    stream=io.BytesIO();np.lib.format.write_array(stream,array,version=(2,0),allow_pickle=False)
    payload=stream.getvalue()
    if len(payload)!=array_spec(array.shape,array.dtype)['file_bytes'] or len(payload)>array_spec((max_count,),'<f8')['file_bytes']:
        raise ValueError('Unexpected timestamp NPY byte bound')
    return payload
