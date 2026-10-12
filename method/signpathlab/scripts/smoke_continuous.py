"""Functional head/metric integration without substituting fixtures for a real SLT/CSLR run."""
import argparse
import torch
from signpath.cli import write_json
from signpath.config import load_config
from signpath.evaluation.ctc import CTCHead
from signpath.evaluation.slt import FixedDecoderSLT
from signpath.evaluation.metrics import corpus_wer
from signpath.evaluation.translation_metrics import translation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    c = load_config(args.config).values
    torch.set_num_threads(1)
    torch.manual_seed(c["experiment"]["seed"])
    task, d = c["downstream"]["task"], c["model"]["embed_dim"]
    dense = torch.randn(2, 12, d, requires_grad=True)
    valid = torch.tensor([[True] * 12, [True] * 9 + [False] * 3])
    if task == "cslr":
        head = CTCHead(d, 16)
        targets, target_lengths = torch.tensor([[3, 3, 5], [6, 7, 0]]), torch.tensor([3, 2])
        input_lengths = valid.sum(1)
        objective = lambda: head.loss(dense, input_lengths, targets, target_lengths)
    elif task == "slt":
        head = FixedDecoderSLT(d, 16, dimension=48, heads=4, layers=2)
        targets = torch.tensor([[1, 3, 4, 5, 6, 2], [1, 7, 8, 9, 2, 0]])
        objective = lambda: head.loss(dense, valid, targets)
    else:
        raise ValueError("Continuous correctness interface requires cslr or slt task")
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        if dense.grad is not None:
            dense.grad.zero_()
        loss = objective()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(dense.grad).all() and dense.grad.abs().sum() > 0
        optimizer.step()
        losses.append(float(loss.detach()))
    head.eval()
    if task == "cslr":
        predictions = head.greedy_decode(dense.detach(), input_lengths)
        references = [targets[i, :int(target_lengths[i])].tolist() for i in range(2)]
        metrics = corpus_wer(references, predictions)
    else:
        predictions = head.greedy_decode(dense.detach(), valid, max_tokens=8)
        references = [[3, 4, 5, 6], [7, 8, 9]]
        metrics = translation_metrics([" ".join(f"token{x}" for x in r) for r in references],
                                      [" ".join(f"token{x}" for x in p) for p in predictions])
    report = {"status": "completed_correctness_only", "synthetic_correctness_only": True, "task": task,
        "losses": losses, "predictions": predictions, "metrics": metrics,
        "real_experiment": {"status": "not_run", "reason": "No authorized How2Sign/OpenASL/continuous gloss manifest configured"}}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
