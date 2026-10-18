from torch.utils.data import Dataset
import torch
from .schema import read_manifest, load_pose
from .normalization import canonicalize_sample, select_frames


class PoseDataset(Dataset):
    def __init__(self, manifest, data_config, stage="ssl", sampling="full"):
        self.rows = read_manifest(manifest)
        self.config, self.stage, self.sampling = data_config, stage, sampling
        if stage == "ssl" and any(r["split"] not in ("train", "pretrain") for r in self.rows):
            raise ValueError("SSL training cannot read validation/test splits")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        generator = None
        if isinstance(index, tuple):
            index, seed = index
            generator = torch.Generator().manual_seed(seed)
        row = self.rows[index]
        c = self.config
        sample = canonicalize_sample(load_pose(row["pose_path"]), row, c["confidence_threshold"], c["min_part_valid_fraction"], c["coordinate_profile"])
        n, cap = sample["length"], c["max_frames"]
        if n > cap and self.sampling == "continuous":
            start = int(torch.randint(n - cap + 1, (), generator=generator).item())
            sample = select_frames(sample, torch.arange(start, start + cap))
        elif n > cap and self.sampling == "isolated":
            sample = select_frames(sample, torch.linspace(0, n - 1, cap).round().long())
        if self.stage != "ssl":
            if row.get("gloss_id") is not None:
                if not row.get("gloss_namespace"):
                    raise ValueError("Gloss must have a verified language/corpus namespace")
                sample["label"] = (row["language"], row["gloss_namespace"], str(row["gloss_id"]))
            sample["gloss_sequence"] = row.get("gloss_sequence")
            sample["translation"] = row.get("translation")
        return sample
