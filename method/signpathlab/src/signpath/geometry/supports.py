"""Set-based source dependencies; IDs use original decoded frames, never aliases."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, order=True)
class Primitive:
    kind: str
    frame: int
    index: int = -1

    def __post_init__(self):
        if self.kind not in {"rgb", "joint", "face", "pose", "feature"} or self.frame < 0:
            raise ValueError("Invalid primitive kind/frame")
        if (self.kind == "rgb" and self.index != -1) or (self.kind != "rgb" and self.index < 0):
            raise ValueError("Invalid primitive index")


def normalized_joint_support(frame, joints):
    return frozenset(Primitive("joint", int(frame), j) for j in set(joints) | {1, 2})


def encode_supports(supports):
    kinds = {"rgb": 0, "joint": 1, "face": 2, "pose": 3, "feature": 4}
    pointers, rows = [0], []
    for support in supports:
        rows.extend((kinds[p.kind], p.frame, p.index) for p in sorted(support))
        pointers.append(len(rows))
    return {"indptr": np.asarray(pointers, dtype=np.int64), "cells": np.asarray(rows, dtype=np.int64).reshape(-1, 3)}


def decode_supports(arrays):
    pointers, cells = arrays["indptr"], arrays["cells"]
    if pointers.dtype != np.int64 or cells.dtype != np.int64 or pointers.ndim != 1 or not len(pointers) or cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("Malformed support encoding")
    if pointers[0] != 0 or pointers[-1] != len(cells) or (np.diff(pointers) < 0).any() or ((cells[:, 0] < 0) | (cells[:, 0] > 4)).any():
        raise ValueError("Malformed support pointers or kind IDs")
    kinds = ("rgb", "joint", "face", "pose", "feature")
    return tuple(frozenset(Primitive(kinds[k], int(f), int(j)) for k, f, j in cells[a:b]) for a, b in zip(pointers[:-1], pointers[1:]))
