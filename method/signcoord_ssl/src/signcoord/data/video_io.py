import av
import numpy as np


def video_metadata(path):
    """Decode timestamps without materializing the full RGB video in RAM."""
    pts=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        width,height=stream.width,stream.height
        for frame in container.decode(stream):
            if frame.pts is None: raise ValueError("video frame missing PTS")
            if (frame.width,frame.height)!=(width,height): raise ValueError('video frame geometry changes within source')
            pts.append(float(frame.pts*frame.time_base))
    pts=np.asarray(pts,dtype=np.float64)
    if len(pts)==0 or not np.all(np.diff(pts)>0): raise ValueError("empty/nonmonotonic video PTS")
    return pts,width,height


def decode_frames(path, indices):
    """Materialize only requested source-frame indices, preserving repeats."""
    needed=set(int(i) for i in indices); output={}
    with av.open(str(path)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in needed: output[i]=frame.to_ndarray(format="rgb24")
            if i>=max(needed): break
    if output.keys()!=needed: raise ValueError("requested source frames not decoded")
    return np.stack([output[int(i)] for i in indices])


def decode_video(path):
    frames, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("video frame missing PTS")
            frames.append(frame.to_ndarray(format="rgb24"))
            pts.append(float(frame.pts*frame.time_base))
    if not frames:
        raise ValueError(f"no decoded frames: {path}")
    pts = np.asarray(pts, dtype=np.float64)
    if not np.all(np.diff(pts) > 0):
        raise ValueError("nonmonotonic video PTS")
    return np.stack(frames), pts
