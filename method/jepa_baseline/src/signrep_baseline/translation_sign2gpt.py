"""Sign2GPT drop-in feature encoder/dataloader: run with its pinned repo on sys.path.

Only the visual input is replaced; temporal MetaFormer, language adapters, XGLM,
training losses and generation remain the original author modules. No pseudo gloss.
"""

import copy
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from .data import read_manifest


class Model(nn.Module):
    def __init__(self, encoder_params=None):
        super().__init__()
        if encoder_params is None:
            from configs.standards.standard_meta_model_zero_config import get_sign_encoder
            encoder_params = get_sign_encoder()[1]["encoder_params"]
        from models.metaformer.meta_model import MetaFormer
        encoder_params = copy.deepcopy(encoder_params)
        self.dim = encoder_params["emb_params"]["d_model"]
        self.lin = nn.Linear(768, encoder_params["emb_params"]["in_dim"])
        self.bn = nn.BatchNorm1d(encoder_params["emb_params"]["in_dim"])
        self.encoder = MetaFormer(**encoder_params)
        self.post_model = None  # original stage2 explicitly disables the pseudo-gloss head

    def forward(self, frame_features, max_len=None):
        lengths = [len(features) for features in frame_features]
        if not lengths or min(lengths) < 1 or any(f.ndim != 2 or f.shape[1] != 768 for f in frame_features):
            raise ValueError("Sign2GPT expects a list of unpooled T×768 frozen SignRep features")
        maximum = int(max_len) if max_len is not None else max(lengths)
        if maximum < max(lengths):
            raise ValueError("Sign2GPT max_len would truncate frozen features; configure it explicitly")
        observed = self.bn(self.lin(torch.cat(frame_features)))
        padded, cursor = [], 0
        mask = torch.zeros(len(lengths), maximum, dtype=torch.bool, device=observed.device)
        for index, length in enumerate(lengths):
            padded.append(torch.nn.functional.pad(observed[cursor:cursor + length], (0, 0, 0, maximum - length)))
            mask[index, :length] = True
            cursor += length
        output = self.encoder(torch.stack(padded), mask)
        return {"enc_output": output, "dict_post_output": {},
                "dict_feat": {"list_of_original_features": torch.cat(frame_features)}}


class FeatureDataset(Dataset):
    def __init__(self, manifest, split):
        self.rows = read_manifest(manifest, split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(row["features"], allow_pickle=False) as file:
            metadata = json.loads(str(file["metadata_json"]))
            if metadata.get("schema") != "signrep_features_v1" or metadata["stride"] != 2:
                raise ValueError("Translation requires stride2 unpooled frozen SignRep features")
            features = torch.from_numpy(file["features"].copy()).float()
        return {"index": torch.tensor(index), "frames": features,
                "sentence": row["translation"], "file_name": row["video_id"]}

    @staticmethod
    def collate_fn(batch):
        return {key: [item[key] for item in batch] for key in batch[0]}


def get_ds(ds_params, transform=None):
    import ignite.distributed as idist
    dataset = FeatureDataset(ds_params["manifest"], ds_params["split"])
    loader = idist.auto_dataloader(dataset, batch_size=ds_params["bs"], shuffle=ds_params["shuffle"],
                                 num_workers=ds_params["num_workers"], drop_last=ds_params["drop_last"],
                                 collate_fn=dataset.collate_fn, pin_memory=True)
    return loader, {"length": len(dataset), "dict_lem_to_id": None}
