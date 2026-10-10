"""Deterministic within-batch derangement of compatible relational targets."""
import hashlib
from collections import defaultdict

import torch
import torch.distributed as dist


def compatible_derangement(basic_valid, times, sample_ids, seed, update):
    if basic_valid.shape[0] != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("shuffle requires distinct sample IDs in each microbatch")
    if times.shape[0] != len(sample_ids):
        raise ValueError("shuffle timeline batch mismatch")
    # Compare interval timing and every basic-valid anchor interval, including
    # padding. Absolute clip origin is irrelevant to relative coordination.
    groups = defaultdict(list)
    support = basic_valid.detach().cpu()
    deltas = torch.diff(times.detach().cpu().double(), dim=1)
    for i in range(len(sample_ids)):
        key = (tuple(torch.round(deltas[i]*1e6).long().tolist()),
               bytes(support[i].flatten().to(torch.uint8).tolist()))
        groups[key].append(i)
    permutation = torch.arange(len(sample_ids), device=basic_valid.device)
    eligible = torch.zeros(len(sample_ids), device=basic_valid.device, dtype=torch.bool)
    for indices in groups.values():
        if len(indices) < 2 or not support[indices[0]].any():
            continue
        ordered = sorted(indices, key=lambda i: sample_ids[i])
        identity = "\0".join([str(seed), str(update)] + [sample_ids[i] for i in ordered])
        rotation = 1 + int(hashlib.sha256(identity.encode()).hexdigest(), 16) % (len(ordered)-1)
        for position, index in enumerate(ordered):
            permutation[index] = ordered[(position+rotation) % len(ordered)]
            eligible[index] = True
    return permutation, eligible, {
        "strategy": "exact_basic_support_and_microsecond_interval_groups_nonzero_cyclic_rotation_v1",
        "scope": "local_microbatch_single_process",
        "eligible_samples": int(eligible.sum()), "changed_samples": int(eligible.sum()),
        "skipped_samples": int((~eligible).sum()),
        "target_sample_ids": [sample_ids[i] if bool(ok) else None
                              for i, ok in zip(permutation.tolist(), eligible.tolist())],
    }


def shuffled_relational_targets(velocity, reliability, support, times, sample_ids, seed, update):
    """Gather detached targets, derive one global derangement, then select this rank."""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world == 1:
        permutation, eligible, log = compatible_derangement(support, times, sample_ids, seed, update)
        return velocity[permutation], reliability[permutation] * eligible[:, None, None], log
    def gather(tensor):
        values = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(values, tensor.detach().contiguous())
        return torch.cat(values)
    global_velocity, global_reliability = gather(velocity), gather(reliability)
    global_support, global_times = gather(support), gather(times)
    ids = [None] * world
    dist.all_gather_object(ids, sample_ids)
    global_ids = [sample_id for rank_ids in ids for sample_id in rank_ids]
    permutation, eligible, log = compatible_derangement(global_support, global_times, global_ids, seed, update)
    local = slice(rank * len(sample_ids), (rank + 1) * len(sample_ids))
    log.update(scope="global_ddp_microbatch_detached_targets",
               global_target_sample_ids=log["target_sample_ids"],
               target_sample_ids=log["target_sample_ids"][local],
               local_eligible_samples=int(eligible[local].sum()))
    return global_velocity[permutation[local]], global_reliability[permutation[local]] * eligible[local, None, None], log
