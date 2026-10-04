"""Explicit estimated-clock MS-ASL adapter, preserving decoded frame ordinals.

Raw timestamps are retained, never sorted or overwritten. The estimator comes
from pinned ffprobe, not from a claim of annotation timing ground truth.
"""
from fractions import Fraction
import hashlib
import math
import subprocess
import json

import av
import cv2
import numpy as np

from msasl_native import crop_bounds
from rgb_payload_hash import rgb_sha256


def select_estimated_ordinals(raw, best, time_base, origin_pts, start, end):
    if (not raw or len(raw)!=len(best) or any(type(v) is not int for v in raw+best)
            or type(origin_pts) is not int):
        raise ValueError('Clock ordinal/observed integer contract failed')
    tb=Fraction(time_base);left,right=Fraction(str(start)),Fraction(str(end))
    if tb<=0 or not 0<=left<right or right-left>8:
        raise ValueError('Bounded annotation/timebase contract failed')
    relative=[(v-origin_pts)*tb for v in best]
    indices=[i for i,t in enumerate(relative) if left<=t<right]
    if not indices:return []
    if indices!=list(range(indices[0],indices[-1]+1)):
        raise ValueError('Estimated clock reenters selected interval')
    selected=[relative[i] for i in indices]
    if any(b<=a for a,b in zip(selected,selected[1:])):
        raise ValueError('Selected estimated clock is not strictly increasing')
    if any(t>=left for t in relative[:indices[0]]) or any(t<right for t in relative[indices[-1]+1:]):
        raise ValueError('Before/selected/after clock partition failed')
    return indices


def reference_interval(path,row,time_base,origin_pts,ffprobe,timeout=30,byte_limit=1<<20,max_decoded=4096):
    tb=Fraction(time_base);origin=origin_pts*tb
    left,right=Fraction(str(row['start_time'])),Fraction(str(row['end_time']))
    interval=f'{float(origin+left):.9f}%{float(origin+right+1):.9f}'
    cmd=[ffprobe,'-v','error','-threads','1','-select_streams','v:0','-read_intervals',interval,
         '-show_entries','stream=time_base,start_pts,width,height:frame=pkt_pts,best_effort_timestamp','-of','json',str(path)]
    p=subprocess.run(cmd,check=True,capture_output=True,timeout=timeout)
    if max(len(p.stdout),len(p.stderr))>byte_limit:raise ValueError('Reference response bound exceeded')
    data=json.loads(p.stdout);headers=data['streams'];frames=data['frames']
    if len(headers)!=1 or not 0<len(frames)<=max_decoded:raise ValueError('Reference stream/frame count failed')
    header=headers[0]
    if Fraction(header['time_base'])!=tb or int(header['start_pts'])!=origin_pts:
        raise ValueError('Reference origin/timebase changed')
    raw,best=[int(x['pkt_pts']) for x in frames],[int(x['best_effort_timestamp']) for x in frames]
    return dict(header=header,raw_ticks=raw,best_ticks=best,command=cmd,response_bytes=len(p.stdout),stderr_bytes=len(p.stderr))


def scan_estimated_interval(path,row,reference,max_selected=1024,max_decoded=4096,expected_rgb_sha256=None):
    header=reference['header'];tb=Fraction(header['time_base']);origin=int(header['start_pts'])
    raw,best=reference['raw_ticks'],reference['best_ticks']
    indices=select_estimated_ordinals(raw,best,str(tb),origin,row['start_time'],row['end_time'])
    if len(indices)>max_selected or (indices and indices[-1]+1>max_decoded):
        raise ValueError('Selected/decode ordinal bound exceeded')
    if expected_rgb_sha256 is not None and len(expected_rgb_sha256)!=len(raw):
        raise ValueError('RGB reference ordinal length mismatch')
    bounds=None;raw_selected=[];best_selected=[];raw_rgb_hashes=[];decoded_raw=[]
    all_sha,first_sha=hashlib.sha256(),hashlib.sha256()
    selected=set(indices);last=indices[-1] if indices else -1
    with av.open(str(path)) as container:
        if len(container.streams.video)!=1:raise ValueError('Expected one native video stream')
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        if stream.time_base!=tb or stream.start_time!=origin:raise ValueError('Native origin/timebase changed')
        left=Fraction(str(row['start_time']));container.seek(math.floor((origin*tb+left)/tb),stream=stream,backward=True,any_frame=False)
        if last>=0:
            for ordinal,frame in enumerate(container.decode(video=0)):
                if ordinal>last:break
                if frame.pts is None or frame.time_base!=tb or frame.is_corrupt or int(frame.pts)!=raw[ordinal]:
                    raise ValueError('Native/reference prefix or corrupt-frame contract failed')
                decoded_raw.append(int(frame.pts))
                if ordinal not in selected:continue
                if frame.width!=header['width'] or frame.height!=header['height'] or frame.width>3840 or frame.height>2160:
                    raise ValueError('Native resolution/header contract failed')
                current=crop_bounds(row['box'],frame.width,frame.height)
                if bounds is not None and current!=bounds:raise ValueError('Dynamic crop shape')
                bounds=current;rgb=frame.to_ndarray(format='rgb24');digest=rgb_sha256(rgb)
                if expected_rgb_sha256 is not None and digest!=expected_rgb_sha256[ordinal]:
                    raise ValueError('Independent decoded RGB ordinal mismatch')
                y0,x0,y1,x1=bounds
                resized=cv2.resize(rgb[y0:y1,x0:x1],(224,224),interpolation=cv2.INTER_LINEAR)
                data=resized.tobytes(order='C');all_sha.update(data)
                if len(raw_selected)<16:first_sha.update(data)
                raw_selected.append(raw[ordinal]);best_selected.append(best[ordinal]);raw_rgb_hashes.append(digest)
    if len(decoded_raw)!=last+1 or len(raw_selected)!=len(indices):raise ValueError('Incomplete native ordinal coverage')
    n=len(indices)
    return dict(time_base=str(tb),origin_pts=origin,origin_seconds=float(origin*tb),
        stream_width=header['width'],stream_height=header['height'],crop_yxyx_pixels=bounds,
        selected_native_frames=n,native16_eligible=n>=16,complete_stride2_windows=max(0,(n-16)//2+1),
        uncovered_tail_frames=n if n<16 else (n-16)%2,decoded_prefix_through_last_selected_frames=len(decoded_raw),
        selected_decoded_ordinals=indices,selected_raw_ticks=raw_selected,selected_estimated_ticks=best_selected,
        selected_raw_RGB_sha256=raw_rgb_hashes,estimated_pts_ticks_sha256=hashlib.sha256(np.asarray(best_selected,dtype='<i8').tobytes()).hexdigest(),
        raw_pts_ticks_sha256=hashlib.sha256(np.asarray(raw_selected,dtype='<i8').tobytes()).hexdigest(),
        all_resized_RGB_sha256=all_sha.hexdigest(),first_native16_RGB_sha256=None if n<16 else first_sha.hexdigest(),
        first_selected_relative_pts=None if not n else float((best_selected[0]-origin)*tb),
        last_selected_relative_pts=None if not n else float((best_selected[-1]-origin)*tb),
        clock='ffprobe best_effort_timestamp; estimated; native decoded ordinals unchanged',
        raw_prefix_exact=True,independent_selected_RGB_exact=expected_rgb_sha256 is not None)
