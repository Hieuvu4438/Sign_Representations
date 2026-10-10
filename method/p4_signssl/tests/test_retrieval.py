"""Known-answer checks for p4.retrieval (CPU)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import (class_max_scores, metrics_from_ranks, paired_cluster_bootstrap,  # noqa: E402
                          pool_clips, ranks_from_scores)

# gallery: class 0 at e0, class 1 at e1, class 2 at e2 (+ a second class-2 item near e0)
g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.8, 0.6, 0]], np.float32)
gl = np.array([0, 1, 2, 2])
q = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32)
ql = np.array([0, 2, 0])
s = class_max_scores(q, g, gl, 3, device="cpu")
r = ranks_from_scores(s, ql)
# q0: class0 score 1, class2 0.8 -> rank 1. q1: class2 score 1 -> rank 1.
# q2 (true 0): class1=1, class2=0.6, class0=0 -> rank 3
assert list(r) == [1, 1, 3], r
m = metrics_from_ranks(r, ql)
assert abs(m["MRR"] - (1 + 1 + 1 / 3) / 3) < 1e-9
assert abs(m["DCG"] - (1 + 1 + 1 / np.log2(4)) / 3) < 1e-9
assert m["R@1"] == 2 / 3
# tie counts against the query
s_tie = np.array([[0.5, 0.5, 0.1]])
assert ranks_from_scores(s_tie, [0])[0] == 2
# pooling with zero-activity fallback
w = np.array([[1, 0], [3, 0], [5, 5]], np.float32)
p, fb = pool_clips(w, np.array([0, 2, 3]), weights=np.array([0.0, 0.0, 1.0]))
assert fb == 1 and np.allclose(p[0], [2, 0]) and np.allclose(p[1], [5, 5])
p, _ = pool_clips(w, np.array([0, 2, 3]), weights=np.array([1.0, 3.0, 1.0]))
assert np.allclose(p[0], [2.5, 0])
bs = paired_cluster_bootstrap([0, 0, 0, 0], [1, 1, 0, 0], ["a", "a", "b", "b"], n_boot=200)
assert bs["diff"] == 0.5 and bs["n_clusters_positive"] == 1
print("RETRIEVAL TESTS OK")
