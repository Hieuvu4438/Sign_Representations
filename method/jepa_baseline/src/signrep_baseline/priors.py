"""Construct priors from STANDARDIZED 3-D teacher pose and original Euler parameters."""

import torch
from .schema import (FidelityChoices, PRIOR_SHAPES, RIGHT_TIPS, LEFT_TIPS, RIGHT_DEST,
                     LEFT_DEST, BODY_SOURCE, BODY_DEST)


def _pairs(pose, observed, sources, destinations, mode):
    # The paper's trailing dimension is xyz, not one Euclidean norm.
    out = pose[:, sources].unsqueeze(2) - pose[:, destinations].unsqueeze(1)
    if mode == "signed_xyz_destination_minus_source":
        out = -out
    if mode == "absolute_xyz":
        out = out.abs()
    valid = observed[:, sources].unsqueeze(2) & observed[:, destinations].unsqueeze(1)
    return out, valid.unsqueeze(-1).expand_as(out)


def activity_prior(pose, observed, correction=1, threshold=0.26):
    """A1 activity for one selected clip. Unknown coverage is never inactive truth.

    U05/U06 convention: wrist remains below torso reference throughout clip;
    complete hand/reference observations required; sum std over all 21×3 coords.
    Coordinates use the original teacher's downward-positive y convention.
    """
    targets = pose.new_zeros(2)
    masks = torch.zeros(2, dtype=torch.bool, device=pose.device)
    if len(pose) <= correction:
        return targets, masks
    reference = pose[:, [0, 3, 6, 7], 1].mean(1)
    for side, start in enumerate([40, 19]):  # active head order: LH, RH (§4)
        points = slice(start, start + 21)
        valid = observed[:, points].all() & observed[:, [0, 3, 6, 7]].all()
        if not valid:
            continue
        below = (pose[:, start, 1] > reference).all()
        stationary = pose[:, points].std(0, correction=correction).sum() < threshold
        targets[side] = (~(below & stationary)).to(pose.dtype)
        masks[side] = True
    return targets, masks


def build_priors(pose, confidence, angles, choices=None, angle_observed=None):
    """One clip: pose T×61×3; confidence T×61; angles body22/right41/left41.

    Pose must already be FK-reconstructed with corpus bone statistics. This function
    does no lifting, scaling by palm size, interpolation, or bend-angle substitution.
    Target scaling occurs once in the reconstruction loss (Table 9).
    """
    choices = choices or FidelityChoices()
    pose = torch.as_tensor(pose, dtype=torch.float32)
    confidence = torch.as_tensor(confidence, device=pose.device)
    if pose.ndim != 3 or pose.shape[1:] != (61, 3) or confidence.shape != pose.shape[:2]:
        raise ValueError("Expected T×61×3 standardized pose and T×61 confidence")
    observed = (confidence > .5) & torch.isfinite(confidence) & torch.isfinite(pose).all(-1)
    safe = torch.where(observed.unsqueeze(-1), pose, torch.zeros_like(pose))
    global_observed = observed.clone()
    # Teacher attaches hands to body wrists (2=right, 5=left). A detected hand
    # still has valid local geometry when its global body anchor is uncertain.
    global_observed[:, 19:40] &= observed[:, 2:3]
    global_observed[:, 40:61] &= observed[:, 5:6]
    targets = {"body_pose": safe.clone()}
    masks = {"body_pose": global_observed.unsqueeze(-1).expand_as(safe).clone()}
    for side, start in [("right", 19), ("left", 40)]:
        hand = safe[:, start:start + 21]
        relative = hand - hand[:, :1]
        valid = observed[:, start:start + 21] & observed[:, start:start + 1]
        targets[f"{side}_pose"] = relative
        masks[f"{side}_pose"] = valid.unsqueeze(-1).expand_as(relative)
    for part, count, key, indices in [
        ("body", 22, "full_body_angles", list(range(19))),
        ("right", 41, "full_right_angles", list(range(19, 40))),
        ("left", 41, "full_left_angles", list(range(40, 61))),
    ]:
        values = torch.as_tensor(angles[part], dtype=pose.dtype, device=pose.device)
        if values.shape != (len(pose), count):
            raise ValueError(f"{part} Euler schema must be T×{count}, got {tuple(values.shape)}")
        if choices.angle_confidence == "teacher_detection":
            if angle_observed is None or part not in angle_observed:
                raise ValueError("Angle supervision requires explicit teacher-detection flags (U05)")
            part_observed = torch.as_tensor(angle_observed[part], device=pose.device, dtype=torch.bool)
            if part_observed.shape != (len(pose),):
                raise ValueError("Angle-detection flags must match clip frame indices")
            good = part_observed[:, None] & torch.isfinite(values)
        else:
            good = observed[:, indices].all(1, keepdim=True) & torch.isfinite(values)
        values = torch.where(good, values, torch.zeros_like(values))
        targets[key] = torch.stack((values.sin(), values.cos()), -1)
        masks[key] = good.unsqueeze(-1).expand_as(targets[key])
    for key, sources, destinations in [
        ("right_dist", RIGHT_TIPS, RIGHT_DEST), ("left_dist", LEFT_TIPS, LEFT_DEST),
        ("body_dist", BODY_SOURCE, BODY_DEST),
    ]:
        point_mask = global_observed if key == "body_dist" else observed
        targets[key], masks[key] = _pairs(safe, point_mask, sources, destinations, choices.distance)
    targets["activity"], masks["activity"] = activity_prior(
        safe, global_observed, choices.activity_std_correction
    )
    for key, shape in PRIOR_SHAPES.items():
        assert targets[key].shape == (len(pose), *shape)
        targets[key] = targets[key].reshape(len(pose), -1)
        masks[key] = masks[key].reshape(len(pose), -1)
    # Masked entries have a finite inert value, and masks remain the authority.
    for key in targets:
        targets[key] = torch.where(masks[key], targets[key], torch.zeros_like(targets[key]))
    return targets, masks
