"""Schema (133 -> 114, no aliases), full-clip sampling, padding never reaching attention/pooling."""
import numpy as np
import torch

from asm_jepa.data.dataset import SampleBuilder
from asm_jepa.data.sampling import full_clip_indices
from asm_jepa.data.schema import MODEL_TO_RAW, N_JOINTS, PART_SIZES, raw_to_model
from asm_jepa.data.synthetic import synthetic_clip
from asm_jepa.models.asm_jepa import Backbone

from conftest import cfg_for


def test_schema_114_without_aliases():
    assert N_JOINTS == 114 and len(set(MODEL_TO_RAW.tolist())) == 114
    assert PART_SIZES == {"body": 4, "lh": 21, "rh": 21, "face": 68}
    for alias in (0, 1, 2, 3, 4, 9, 10):  # body nose/eyes/ears and body wrists are dropped
        assert alias not in MODEL_TO_RAW
    xy = np.random.rand(3, 133, 2).astype(np.float32)
    m, _, _ = raw_to_model(xy, np.ones((3, 133), np.float32), np.ones((3, 133), bool))
    assert m.shape == (3, 114, 2) and np.array_equal(m[:, 4], xy[:, 91]) and np.array_equal(m[:, 25], xy[:, 112])


def test_full_clip_sampling_endpoints_order_and_padding():
    idx, valid = full_clip_indices(300, 64)
    assert idx[0] == 0 and idx[-1] == 299 and np.all(np.diff(idx) > 0) and valid.all()
    rng = np.random.default_rng(1)
    for n in (65, 70, 100, 1000):
        j, _ = full_clip_indices(n, 64, rng, 0.15)
        assert j[0] == 0 and j[-1] == n - 1 and np.all(np.diff(j) > 0)
    idx, valid = full_clip_indices(40, 64)
    assert valid.sum() == 40 and np.array_equal(idx[:40], np.arange(40)) and (idx[40:] == -1).all()


def test_real_timestamps_are_kept(stats):
    cfg = cfg_for("full")
    b = SampleBuilder(cfg, stats)
    clip = synthetic_clip(5)
    _, _, _, ts, tv, idx = b._select(clip, None, False)
    assert np.allclose(ts[tv], clip.timestamps_sec[idx[tv]])


def test_padding_does_not_reach_attention_or_pooling(stats):
    """Garbage written into padded frames / invalid state tokens must not change the pooled embedding."""
    from asm_jepa.data.schema import PART_ORDER, PART_SLICES

    cfg = cfg_for("base")
    b = SampleBuilder(cfg, stats)
    index = next(i for i in range(200) if synthetic_clip(i).num_frames < 40)
    s = b.build(synthetic_clip(index), None, "eval")
    tv = s["token_valid"]
    assert s["usable"] and not tv.all()
    batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in s.items()}
    corrupt = {k: v.clone() for k, v in batch.items()}
    for k in np.flatnonzero(~tv):
        part, p = divmod(int(k), 16)
        corrupt["joints"][0, p * 4:(p + 1) * 4, PART_SLICES[PART_ORDER[part]]] = 123.0
    bb = Backbone(cfg).eval()
    with torch.no_grad():
        f1, _ = bb.embed_clip(batch)
        f2, _ = bb.embed_clip(corrupt)
    assert torch.allclose(f1["state_part_concat1024_l2"], f2["state_part_concat1024_l2"], atol=1e-5)
