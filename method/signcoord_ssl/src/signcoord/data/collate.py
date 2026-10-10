import torch


def collate_ssl(samples):
    return {"student_inputs": {k: torch.stack([s["student_inputs"][k] for s in samples]) for k in samples[0]["student_inputs"]},
            "targets": {k: torch.stack([s["targets"][k] for s in samples]) for k in samples[0]["targets"]},
            "audit": [s["audit"] for s in samples]}
