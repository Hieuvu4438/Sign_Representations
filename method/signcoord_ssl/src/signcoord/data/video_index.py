"""Content-bound decoded PTS indices; RGB-only and reusable across epochs."""
import json
import os
import tempfile
from pathlib import Path

import av
import numpy as np

from signcoord.provenance import object_hash,sha256_file
from .video_io import video_metadata

POLICY='decoded_pts_constant_geometry_v1'
_VERIFIED={}
_METADATA={}


def fingerprint(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def verify_media(path, expected):
    """Hash once per stable file stat in this process; detect ordinary mutation."""
    path=Path(path).resolve(); before=fingerprint(path)
    if _VERIFIED.get((str(path),expected)) != before:
        if sha256_file(path)!=expected: raise ValueError('media changed since manifest freeze')
        if fingerprint(path)!=before: raise ValueError('media changed during content verification')
        _VERIFIED[(str(path),expected)]=before
    return before


def create_video_index(media,root,expected_hash=None):
    digest=expected_hash or sha256_file(media)
    before=verify_media(media,digest)
    key=object_hash({'media_sha256':digest,'policy':POLICY,'av_version':av.__version__})
    path=Path(root)/(key+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        pts,w,h=read_video_index(path,digest)
        return path,pts,w,h
    pts,w,h=video_metadata(media)
    if fingerprint(media)!=before: raise ValueError('media changed while building PTS index')
    metadata={'schema_version':'video_index_v1','media_sha256':digest,'policy':POLICY,
              'av_version':av.__version__,'width':w,'height':h,'frame_count':len(pts)}
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.')
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,pts_s=pts,metadata=json.dumps(metadata,sort_keys=True))
            f.flush();os.fsync(f.fileno())
        read_video_index(tmp,digest)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return path,pts,w,h


def read_video_index(path,media_hash,index_hash=None):
    if index_hash is not None and sha256_file(path)!=index_hash:
        raise ValueError('video index content differs from manifest')
    with np.load(path,allow_pickle=False) as d:
        metadata=json.loads(str(d['metadata'])); pts=d['pts_s'].copy()
    if (metadata.get('schema_version')!='video_index_v1' or metadata.get('media_sha256')!=media_hash
            or metadata.get('policy')!=POLICY or metadata.get('av_version')!=av.__version__):
        raise ValueError('video index source/decoder policy mismatch')
    if pts.ndim!=1 or len(pts)!=metadata['frame_count'] or not len(pts) or not np.isfinite(pts).all() or not (np.diff(pts)>0).all():
        raise ValueError('invalid decoded PTS index')
    if metadata['width']<=0 or metadata['height']<=0: raise ValueError('invalid indexed frame geometry')
    return pts,metadata['width'],metadata['height']


def record_metadata(record):
    verify_media(record['media_path'],record['media_sha256'])
    if record.get('video_index_path'):
        if not record.get('video_index_sha256'): raise ValueError('manifest index requires a frozen index SHA256')
        return read_video_index(record['video_index_path'],record['media_sha256'],record['video_index_sha256'])
    key=(str(Path(record['media_path']).resolve()),record['media_sha256'],av.__version__)
    if key not in _METADATA:_METADATA[key]=video_metadata(record['media_path'])
    pts,w,h=_METADATA[key]
    return pts.copy(),w,h
