"""Paper §3/A1 target schema; indices in Supplement Fig.2, not COCO indices."""

from dataclasses import dataclass

PRIOR_SHAPES = {
    "full_body_angles": (22, 2), "full_right_angles": (41, 2),
    "full_left_angles": (41, 2), "body_pose": (61, 3),
    "right_pose": (21, 3), "left_pose": (21, 3),
    "body_dist": (12, 22, 3), "right_dist": (5, 11, 3),
    "left_dist": (5, 11, 3),
}
LOSS_WEIGHTS = {name: (20.0 if "dist" in name else 10.0) for name in PRIOR_SHAPES}
LOSS_WEIGHTS["activity"] = 0.2
TARGET_SCALES = {name: 1.0 for name in PRIOR_SHAPES}
TARGET_SCALES.update(right_pose=2.0, left_pose=2.0, right_dist=4.0, left_dist=4.0)
RIGHT_WRIST, LEFT_WRIST = 19, 40
RIGHT_TIPS = (23, 27, 31, 35, 39)
LEFT_TIPS = (44, 48, 52, 56, 60)
RIGHT_DEST = (19, 20, 23, 24, 27, 28, 31, 32, 35, 36, 39)
LEFT_DEST = (40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60)
BODY_SOURCE = (19, *RIGHT_TIPS, 40, *LEFT_TIPS)
BODY_DEST = (0, 3, 6, 7, 10, 13, 15, 16, 17, 18, 19, 23, 27, 31, 35, 39,
             40, 44, 48, 52, 56, 60)


@dataclass(frozen=True)
class FidelityChoices:
    """Engineering conventions absent from the paper; persist these with every run."""

    distance: str = "signed_xyz_source_minus_destination"
    gram_normalization: str = "tokens_channels"
    smooth_l1_beta: float = 1.0
    reconstruction_reduction: str = "valid_element_mean"
    activity_std_correction: int = 1
    activity_weighting: str = "raw"
    pretrain_batch_counts: str = "total_views"
    angle_confidence: str = "teacher_detection"

    def __post_init__(self):
        if self.distance not in {"signed_xyz_source_minus_destination", "signed_xyz_destination_minus_source", "absolute_xyz"}:
            raise ValueError("Distance requires an explicit per-axis convention (U03)")
        if self.gram_normalization not in {"tokens_channels", "tokens", "none"}:
            raise ValueError("Unknown Gram convention (U04)")
        if self.smooth_l1_beta <= 0:
            raise ValueError("SmoothL1 beta must be positive")
        if self.reconstruction_reduction != "valid_element_mean":
            raise ValueError("Unsupported loss reduction")
        if self.activity_std_correction not in {0, 1}:
            raise ValueError("Unsupported activity standard-deviation convention")
        if self.activity_weighting not in {"raw", "clamp_0_1"}:
            raise ValueError("Unsupported activity weighting")
        if self.pretrain_batch_counts not in {"total_views", "video_pairs"}:
            raise ValueError("Unsupported batch-count interpretation")
        if self.angle_confidence not in {"teacher_detection", "all_observed_points"}:
            raise ValueError("Unsupported angle-observation convention")
