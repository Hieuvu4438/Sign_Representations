"""Constrained part/span masking with a caller-owned deterministic generator."""
import math
import torch


class MaskGenerator:
    def __init__(self, config):
        self.cfg = config

    def __call__(self, valid, generator):
        cfg = self.cfg
        targets = torch.zeros_like(valid)
        # CPU selection makes the RNG reproducible on CPU and CUDA.
        for b, row in enumerate(valid.detach().cpu()):
            grid = row.reshape(-1, 4)
            total = int(row.sum())
            keep = max(cfg["min_visible_tokens"], math.ceil(total*cfg["min_visible_fraction"]))
            budget = min(round(total*cfg["ratio"]), max(0, total-keep))
            chosen = torch.zeros_like(grid)
            def add(k, p):
                if not grid[k, p] or chosen[k, p] or int(chosen.sum()) >= budget:
                    return False
                remain = grid & ~chosen
                if cfg["keep_one_visible_part_per_segment"] and remain[k].sum() <= 1:
                    return False
                if p in (1, 2) and remain[:, p].sum() <= min(cfg["min_visible_segments_per_hand"], grid[:, p].sum()):
                    return False
                chosen[k, p] = True
                return True
            if budget:
                weights = torch.tensor(cfg["part_weights"])[None].expand_as(grid).flatten() * row
                order = torch.multinomial(weights, int(row.sum()), replacement=False, generator=generator)
                span_budget = round(budget*cfg["span_budget_fraction"])
                for idx in order.tolist():
                    if int(chosen.sum()) >= span_budget:
                        break
                    k, p = divmod(idx, 4)
                    span = int(torch.randint(cfg["min_span_segments"], cfg["max_span_segments"]+1, (), generator=generator))
                    for offset in range(span):
                        if k+offset < len(grid) and int(chosen.sum()) < span_budget:
                            add(k+offset, p)
                for idx in order.tolist():
                    add(*divmod(idx, 4))
            targets[b] = chosen.flatten().to(valid.device)
        return {"target": targets, "visible": valid & ~targets,
                "actual_ratio": targets.sum(1).float() / valid.sum(1).clamp_min(1)}
