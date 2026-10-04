"""Collect a frozen estimated-clock crop clip with exact registered input joins."""
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import av
import cv2
import numpy as np

from msasl_estimated_clock import reference_interval,select_estimated_ordinals
from msasl_native import crop_bounds


def collect_native_inputs(item,ffprobe,max_native_frames=327,max_decoded=4096,reference=None,source_identity=None):
    path=Path(item['raw_path']);before=None
    if source_identity is not None:
        before=path.stat()
        if (before.st_size!=source_identity['bytes'] or before.st_mtime_ns!=source_identity['mtime_ns']
                or source_identity['sha256']!=item['source_sha256']):
            raise ValueError('Source changed since bound native census')
    row=dict(start_time=item['start_time_seconds'],end_time=item['end_time_seconds'],box=item['box_yxyx'])
    ref=reference if reference is not None else reference_interval(path,row,item['time_base'],item['origin_pts'],ffprobe,max_decoded=max_decoded)
    header=ref['header'];tb=Fraction(item['time_base']);origin=item['origin_pts']
    if Fraction(header['time_base'])!=tb or int(header['start_pts'])!=origin:raise ValueError('Reference origin/timebase changed')
    raw,best=ref['raw_ticks'],ref['best_ticks']
    indices=select_estimated_ordinals(raw,best,str(tb),origin,row['start_time'],row['end_time'])
    if not 16<=len(indices)<=max_native_frames or len(indices)!=item['selected_native_frames'] or indices[-1]+1>max_decoded:
        raise ValueError('Frozen native frame count/ordinal bound failed; no padding or dropping')
    chosen=set(indices);last=indices[-1];frames=[];ticks=[];all_sha=hashlib.sha256();first_sha=hashlib.sha256();decoded=0;bounds=None
    with av.open(str(path)) as container:
        if len(container.streams.video)!=1:raise ValueError('Expected one video stream')
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        if stream.time_base!=tb or stream.start_time!=origin:raise ValueError('Native origin/timebase changed')
        left=Fraction(str(row['start_time']));container.seek(math.floor((origin*tb+left)/tb),stream=stream,backward=True,any_frame=False)
        for ordinal,frame in enumerate(container.decode(video=0)):
            if ordinal>last:break
            if frame.pts is None or frame.time_base!=tb or frame.is_corrupt or int(frame.pts)!=raw[ordinal]:
                raise ValueError('Native/reference raw prefix mismatch or corruption')
            decoded+=1
            if ordinal not in chosen:continue
            if frame.width!=header['width'] or frame.height!=header['height'] or frame.width>3840 or frame.height>2160:
                raise ValueError('Decoded resolution contract failed')
            bounds=crop_bounds(row['box'],frame.width,frame.height)
            if list(bounds)!=list(item['crop_yxyx_pixels']):raise ValueError('Frozen crop changed')
            y0,x0,y1,x1=bounds;rgb=frame.to_ndarray(format='rgb24')
            output=cv2.resize(rgb[y0:y1,x0:x1],(224,224),interpolation=cv2.INTER_LINEAR)
            data=output.tobytes(order='C');all_sha.update(data)
            if len(frames)<16:first_sha.update(data)
            frames.append(output);ticks.append(best[ordinal])
    clock_hash=hashlib.sha256(np.asarray(ticks,dtype='<i8').tobytes()).hexdigest()
    if (decoded!=last+1 or len(frames)!=len(indices) or clock_hash!=item['clock_pts_ticks_sha256']
            or all_sha.hexdigest()!=item['all_resized_RGB_sha256'] or first_sha.hexdigest()!=item['first_native16_RGB_sha256']):
        raise ValueError('Frozen input RGB/PTS/count identity failed')
    if source_identity is not None:
        after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Source changed during decode')
    pts=[float((t-origin)*tb) for t in ticks]
    return frames,pts,dict(sample_id=item['sample_id'],source_sha256=item['source_sha256'],
        source_identity_join=source_identity is not None,native_frames=len(frames),decoded_prefix_frames=decoded,
        complete_stride2_windows=(len(frames)-16)//2+1,clock_pts_ticks_sha256=clock_hash,
        all_resized_RGB_sha256=all_sha.hexdigest(),first_native16_RGB_sha256=first_sha.hexdigest(),
        selected_decoded_ordinals=indices,estimated_ticks=ticks,crop_yxyx_pixels=list(bounds),reference_command=ref.get('command'),
        input_contract=item['input_contract'],clock='estimated best_effort; decoded native order')
