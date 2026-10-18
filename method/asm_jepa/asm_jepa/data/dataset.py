"""Clip -> model sample. One builder serves SSL (two aligned views + mask/closure), supervised training
(one augmented view + label) and deterministic evaluation (no augmentation, no masks).

Labels never enter SSL samples: SSLDataset reads only `pose_relpath` from ssl_train.jsonl rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from asm_jepa.data import augment as A
from asm_jepa.data.normalization import body_normalize, hand_local, joint_channels, saturation
from asm_jepa.data.observation import observation_filter
from asm_jepa.data.pose_store import read_clip
from asm_jepa.data.sampling import full_clip_indices, patch_times
from asm_jepa.data.schema import N_CHANNELS, N_JOINTS, N_PARTS, N_RELATIONS, PART_ORDER, RELATION_CHANNELS, raw_to_model
from asm_jepa.features import state as S
from asm_jepa.features.dependencies import closure, motion_dependencies, static_dependencies
from asm_jepa.features.masking import sample_raw_mask
from asm_jepa.features.paths import path_tokens
from asm_jepa.features.relations import relation_frames, relation_patch_stats
from asm_jepa.features.tokens import TYPE_GEOMETRY, TYPE_RELATION, TYPE_STATE, layout_from_config

MODES = ("ssl", "train_view", "eval")


def joint_thresholds(value) -> np.ndarray:
    """Scalar, or a per-part mapping {body, lh, rh, face} -> [114] raw-score thresholds."""
    from asm_jepa.data.schema import PART_SLICES

    if isinstance(value, dict):
        if set(value) != set(PART_ORDER):
            raise ValueError(f"per-part confidence_threshold needs keys {PART_ORDER}, got {sorted(value)}")
        out = np.zeros(N_JOINTS, np.float32)
        for p in PART_ORDER:
            out[PART_SLICES[p]] = float(value[p])
        return out
    return np.full(N_JOINTS, float(value), np.float32)


class SampleBuilder:
    def __init__(self, cfg: dict, stats: dict, objective: str = "jepa"):
        self.cfg = cfg
        self.stats = stats
        self.layout = layout_from_config(cfg)
        self.objective = objective
        d = cfg["data"]
        self.T = int(d["max_frames"])
        self.P = int(d["patch_frames"])
        self.n_p = self.T // self.P
        self.static_deps = static_dependencies(self.layout)
        L = self.layout
        self.F, self.C, self.Sn = L.n_families, L.n_centers, L.n_scales
        self.is_state = L.arrays["type"] == TYPE_STATE
        self.thresholds = joint_thresholds(cfg["data"]["confidence_threshold"])

    # ------------------------------------------------------------------ helpers
    def _select(self, clip, rng, train: bool):
        d = self.cfg["data"]
        ob133 = observation_filter(clip.xy, clip.observed, clip.image_wh, d)
        xy, sc, ob = raw_to_model(clip.xy, clip.score, ob133)
        idx, time_valid = full_clip_indices(len(xy), self.T, rng if train else None,
                                            d["temporal_jitter_fraction"] if train else 0.0)
        sel = np.where(idx >= 0, idx, 0)
        xy, sc, ob = xy[sel], sc[sel], ob[sel]
        ts = np.where(time_valid, clip.timestamps_sec[sel], 0.0)
        fv = clip.frame_valid[sel] & time_valid
        finite = np.isfinite(xy).all(-1)
        ob = ob & finite & fv[:, None] & (sc >= self.thresholds[None, :])
        xy = np.where(ob[..., None], xy, 0.0).astype(np.float32)
        conf = np.where(ob, np.clip(sc / d["confidence_scale"], 0.0, 1.0), 0.0).astype(np.float32)
        return xy, ob, conf, ts.astype(np.float64), time_valid, idx

    def _view_features(self, xb, ob, conf, ts, tv):
        f = self.cfg["features"]
        local, lv = hand_local(xb, ob)
        out = {"joints": joint_channels(xb, ob, conf, local, lv, tv, self.stats)}
        if self.layout.relations:
            out["rel"], out["_rel_op"], out["_rel_cp"] = relation_frames(xb, ob, conf, tv, self.stats)
        else:
            out["rel"] = np.zeros((self.T, N_RELATIONS, RELATION_CHANNELS), np.float32)
        if self.layout.motion_enabled:
            pt = path_tokens(xb, ob, conf, tv, ts, self.P, self.cfg["features"]["interval_widths_sec"],
                             self.C, f["path_min_points"], self.stats)
            out["motion_geom"] = pt["geom"].reshape(-1, 3)
            out["motion_time"] = pt["timing"].reshape(-1, 7)
            out["_paths"] = pt
        else:
            out["motion_geom"] = np.zeros((self.F * self.C * self.Sn, 3), np.float32)
            out["motion_time"] = np.zeros((self.F * self.C * self.Sn, 7), np.float32)
        return out

    def _empty(self, mode):
        K = self.layout.n_tokens
        s = {"joints": np.zeros((self.T, N_JOINTS, N_CHANNELS), np.float32),
             "rel": np.zeros((self.T, N_RELATIONS, RELATION_CHANNELS), np.float32),
             "motion_geom": np.zeros((self.F * self.C * self.Sn, 3), np.float32),
             "motion_time": np.zeros((self.F * self.C * self.Sn, 7), np.float32),
             "token_valid": np.zeros(K, bool), "token_weight": np.zeros(K, np.float32),
             "token_time": np.zeros(K, np.float32), "log_duration": np.float32(0.0), "usable": False,
             "anchor_fallback_frames": np.int32(0), "n_real_frames": np.int32(0),
             "saturation_body": np.float32(0.0), "saturation_local": np.float32(0.0)}
        if mode == "ssl":
            for k in ("joints", "rel", "motion_geom", "motion_time"):
                s[k + "_clean"] = s[k].copy()
            s.update(context=np.zeros(K, bool), target=np.zeros(K, bool),
                     mask_units=np.zeros(N_PARTS * self.n_p, bool), mask_fallbacks=np.int32(0),
                     mask_strategy=np.int32(-1))
            s.update(self._anchor_arrays(None))
        return s

    def _anchor_arrays(self, joints_clean, tv=None):
        out = {}
        fn = S.mae_targets if self.objective == "mae" else S.anchor_targets
        dims = S.MAE_DIMS if self.objective == "mae" else {p: 3 * d for p, d in S_ANCHOR.items()}
        if joints_clean is None:
            for p in PART_ORDER:
                out[f"anchor_{p}"] = np.zeros((self.n_p, dims[p]), np.float32)
                out[f"anchor_{p}_valid"] = np.zeros((self.n_p, dims[p]), bool)
            return out
        for p, (v, ok) in fn(joints_clean, tv, self.P).items():
            out[f"anchor_{p}"] = v
            out[f"anchor_{p}_valid"] = ok
        return out

    # ------------------------------------------------------------------ main entry
    def build(self, clip, rng: np.random.Generator | None, mode: str, force_mask: np.ndarray | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(mode)
        cfg = self.cfg
        train = mode in ("ssl", "train_view")
        if clip is None:
            return self._empty(mode)
        xy, ob, conf, ts, tv, _ = self._select(clip, rng, train)
        norm = body_normalize(xy, ob, tv)
        if norm["unusable_pose"]:
            s = self._empty(mode)
            s["n_real_frames"] = np.int32(tv.sum())
            return s
        ag = cfg["augment"]
        if ag["mirror_probability"] or ag["time_reverse_probability"]:
            raise NotImplementedError("mirror / time reversal are not part of the default protocol")
        theta, scale = A.shared_view_params(rng if train else None, ag["rotation_degrees"], ag["scale_range"])
        clean = A.apply_shared_view(norm["xy"], ob, theta, scale)
        if mode == "ssl":
            student = A.student_jitter(clean, ob, rng, ag["student_jitter_sigma"])
        elif mode == "train_view":
            student = A.student_jitter(clean, ob, rng, ag["student_jitter_sigma"])
        else:
            student = clean
        fs = self._view_features(student, ob, conf, ts, tv)
        fc = self._view_features(clean, ob, conf, ts, tv) if mode == "ssl" else fs

        # ---- validity / weights / times from the clean view (flags are identical across views)
        thr = cfg["data"]["part_coverage_threshold"]
        cov, real_cnt, w_state = S.part_patch_coverage(fc["joints"], tv, self.P)
        L = self.layout
        A_ = L.arrays
        K = L.n_tokens
        valid = np.zeros(K, bool)
        weight = np.zeros(K, np.float32)
        ptime, pvalid = patch_times(ts, tv, self.P)
        real_t = ts[tv]
        t0, t1 = float(real_t.min()), float(real_t.max())
        dur = max(t1 - t0, 0.0)
        times = np.zeros(K, np.float64)
        st = A_["type"] == TYPE_STATE
        state_valid = (cov >= thr) & pvalid[None, :]
        valid[st] = state_valid[A_["sub"][st], A_["patch"][st]]
        weight[st] = w_state[A_["sub"][st], A_["patch"][st]]
        times[st] = ptime[A_["patch"][st]]
        if L.relations:
            rcov, rw = relation_patch_stats(fc["_rel_op"], fc["_rel_cp"], tv, self.P)
            rl = A_["type"] == TYPE_RELATION
            rvalid = (rcov >= thr) & pvalid[None, :]
            valid[rl] = rvalid[A_["sub"][rl], A_["patch"][rl]]
            weight[rl] = rw[A_["sub"][rl], A_["patch"][rl]]
            times[rl] = ptime[A_["patch"][rl]]
        deps = self.static_deps
        if L.motion_enabled:
            pt = fc["_paths"]
            mo = A_["type"] >= TYPE_GEOMETRY
            f_, c_, s_ = A_["sub"][mo], A_["center"][mo], A_["scale"][mo]
            valid[mo] = pt["valid"][f_, c_, s_]
            weight[mo] = pt["weight"][f_, c_, s_]
            times[mo] = pt["center_time"][c_]
            deps = motion_dependencies(L, self.static_deps.copy(), pt["support_patches"])
        weight = np.where(valid, np.clip(weight, 0.0, 1.0), 0.0).astype(np.float32)
        tnorm = np.where(valid, (times - t0) / dur if dur > 0 else 0.0, 0.0).astype(np.float32)

        sample = {"joints": fs["joints"], "rel": fs["rel"], "motion_geom": fs["motion_geom"],
                  "motion_time": fs["motion_time"], "token_valid": valid, "token_weight": weight,
                  "token_time": tnorm, "log_duration": np.float32(np.log1p(dur)), "usable": bool(valid[st].any()),
                  "anchor_fallback_frames": np.int32(norm["anchor_fallback"].sum()), "n_real_frames": np.int32(tv.sum())}
        loc_c, lv_c = hand_local(clean, ob)
        sat = saturation(clean, loc_c, ob, lv_c, cfg["data"]["coordinate_saturation"])
        sample["saturation_body"] = np.float32(sat["body_saturation"])
        sample["saturation_local"] = np.float32(sat["local_saturation"])
        if mode == "ssl":
            for k in ("joints", "rel", "motion_geom", "motion_time"):
                sample[k + "_clean"] = fc[k]
            mask, info = sample_raw_mask(state_valid, rng, cfg["mask"])
            if force_mask is not None:  # test hook: fixed raw mask after the identical rng draws
                mask = np.asarray(force_mask, bool).reshape(state_valid.shape) & state_valid
            units = mask.reshape(-1)
            ctx, tgt = closure(valid, deps, units, enabled=cfg["mask"]["dependency_closure"], is_state=self.is_state)
            sample.update(context=ctx, target=tgt, mask_units=units, mask_fallbacks=np.int32(info["fallbacks"]),
                          mask_strategy=np.int32(["random", "part_blocks", "temporal_blocks"].index(info["strategy"])))
            sample.update(self._anchor_arrays(fc["joints"], tv))
            sample["usable"] = bool(sample["usable"] and ctx.any())
        return sample


S_ANCHOR = {"body": 8, "lh": 44, "rh": 44, "face": 60}


# ---------------------------------------------------------------------------------------------- sources
class PoseStoreSource:
    def __init__(self, rows: list[dict], processed_root):
        self.rows = rows
        self.root = Path(processed_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path = self.root / self.rows[i]["pose_relpath"]
        if not path.is_file():
            return None
        return read_clip(path)


class SSLDataset(torch.utils.data.Dataset):
    """Items are (index, seed) pairs from the step-aware batch sampler; returns label-free samples."""

    def __init__(self, source, builder: SampleBuilder):
        self.source, self.builder = source, builder

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        index, seed = item
        rng = np.random.default_rng(seed)
        s = self.builder.build(self.source[index], rng, "ssl")
        s["index"] = np.int64(index)
        return s


class LabeledDataset(torch.utils.data.Dataset):
    """Evaluation / supervised samples. class_id is attached for the evaluator / classifier loss only."""

    def __init__(self, source, builder: SampleBuilder, class_ids, mode: str = "eval", seed: int = 0):
        assert mode in ("eval", "train_view")
        self.source, self.builder, self.class_ids, self.mode, self.seed = source, builder, class_ids, mode, seed
        self.epoch = 0

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            index, seed = item
        else:
            index, seed = item, None
        rng = np.random.default_rng(seed) if (self.mode == "train_view" and seed is not None) else None
        if self.mode == "train_view" and rng is None:
            rng = np.random.default_rng([self.seed, self.epoch, index])
        s = self.builder.build(self.source[index], rng, self.mode)
        s["index"] = np.int64(index)
        s["class_id"] = np.int64(self.class_ids[index] if self.class_ids is not None else -1)
        return s


def worker_init(_):
    torch.set_num_threads(1)
