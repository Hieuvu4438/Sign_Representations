import torch

BODY_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4), (3, 5), (5, 7), (4, 6), (6, 8)]
HAND_EDGES = [(a, b) for first in (1, 5, 9, 13, 17) for a, b in ((0, first), (first, first+1), (first+1, first+2), (first+2, first+3))]
FACE_EDGES = [(i, i+1) for i in range(8)] + [(9+i, 9+(i+1)%8) for i in range(8)] + [(17, 4), (17, 11)]


def adjacency(joints, edges):
    a = torch.zeros(joints, joints)
    for u, v in edges:
        a[v, u] = a[u, v] = 1
    return a / a.sum(-1, keepdim=True).clamp_min(1)
