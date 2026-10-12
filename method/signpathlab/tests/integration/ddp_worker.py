"""Launched by torchrun; one rank deliberately has no eligible target or context."""
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from signpath.models.teacher_student import ResearchModel
from signpath.training.distributed import initialize, rank
from signpath.training.engine import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    initialize("cpu")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    model = ResearchModel(data["config"])
    model.load_state_dict(data["initial"])
    trainer = Trainer(model, data["config"])
    local = data["batch"].select([rank()])
    result = trainer.update([local])
    torch.save({"state": model.state_dict(), "gradients": {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
                "success": result.succeeded, "step": result.successful_step}, Path(args.output) / f"rank{rank()}.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
