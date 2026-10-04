"""CPU native-frame crop adapter for an explicitly adapted MS-ASL protocol."""
from fractions import Fraction
import math

import av
import cv2
import numpy as np


def crop_bounds(box,width,height):
    if type(width) is not int or type(height) is not int or width<1 or height<1:
        raise ValueError('Invalid decoded dimensions')
    if len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box):
        raise ValueError('Invalid normalized box')
    y0,x0,y1,x1=box
    if not (0<=y0<y1<=1 and 0<=x0<x1<=1):raise ValueError('Invalid yxyx box')
    return (math.floor(y0*height),math.floor(x0*width),math.ceil(y1*height),math.ceil(x1*width))


def selected_indices(pts,start,end):
    left,right=Fraction(str(start)),Fraction(str(end))
    if not 0<=left<right:raise ValueError('Invalid source interval')
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Invalid native PTS')
    return [i for i,t in enumerate(pts) if left<=t<right]


def decode_interval(path,row,max_frames=240,max_decoded=2048):
    """Annotation seconds relative to stream start; half-open, no rate conversion."""
    left,right=Fraction(str(row['start_time'])),Fraction(str(row['end_time']))
    if not 0<=left<right or right-left>8:raise ValueError('Bounded interval contract failed')
    resized=[];pts=[];digests=[];bounds=None;previous=None;decoded=0;first_raw=None
    import hashlib
    with av.open(path if hasattr(path,'read') else str(path)) as container:
        if len(container.streams.video)!=1:raise ValueError('Expected one video stream')
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        if stream.start_time is None or stream.time_base is None:raise ValueError('Unknown stream origin')
        origin=stream.start_time*stream.time_base
        metadata=dict(time_base=str(stream.time_base),origin_pts=stream.start_time,origin_seconds=float(origin),
                      stream_width=stream.width,stream_height=stream.height,
                      average_rate=str(stream.average_rate),reported_stream_frames=stream.frames,
                      stream_duration_seconds=None if stream.duration is None else float(stream.duration*stream.time_base))
        container.seek(math.floor((origin+left)/stream.time_base),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(video=0):
            decoded+=1
            if decoded>max_decoded:raise ValueError('Seek decode bound exceeded')
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:raise ValueError('Missing PTS/corrupt native frame')
            t=frame.pts*frame.time_base-origin
            if previous is not None and t<=previous:raise ValueError('Native PTS not strictly increasing')
            previous=t
            if t<left:continue
            if t>=right:break
            if len(pts)>=max_frames:raise ValueError('Selected native frame bound exceeded')
            if frame.width>3840 or frame.height>2160:raise ValueError('Decoded resolution exceeds plan')
            rgb=frame.to_ndarray(format='rgb24')
            current=crop_bounds(row['box'],frame.width,frame.height)
            if bounds is not None and current!=bounds:raise ValueError('Dynamic decoded crop shape')
            bounds=current;y0,x0,y1,x1=bounds
            output=cv2.resize(rgb[y0:y1,x0:x1],(224,224),interpolation=cv2.INTER_LINEAR)
            if first_raw is None:first_raw=rgb.copy()
            resized.append(output);pts.append(t);digests.append(hashlib.sha256(output.tobytes()).hexdigest())
    if len(pts)<16:raise ValueError('No complete native16-frame input; no padding allowed')
    return np.stack(resized),pts,dict(metadata,crop_yxyx_pixels=bounds,decoded_from_keyframe_frames=decoded,
        selected_native_frames=len(pts),resized_frame_sha256=digests,interval_policy='annotation half-open seconds relative to stream.start_time; no resampling/padding'),first_raw
