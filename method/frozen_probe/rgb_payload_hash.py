"""Hash logical RGB pixels, excluding ndarray strides and row padding."""
import hashlib


def rgb_sha256(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype.name != 'uint8':
        raise ValueError('Only HWC uint8 RGB payload is supported')
    return hashlib.sha256(rgb.tobytes(order='C')).hexdigest()
