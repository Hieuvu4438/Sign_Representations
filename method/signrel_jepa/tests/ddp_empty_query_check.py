"""Two-rank analytical regression: one rank has no valid latent queries."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn.parallel import DistributedDataParallel
from signrel.losses.latent import latent_loss
from signrel.data.manifest import sha256


def main():
    torch.distributed.init_process_group('gloo')
    rank = int(os.environ['RANK'])
    linear = torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():
        linear.weight.fill_(2)
    model = DistributedDataParallel(linear)
    pred = model(torch.tensor([[1.],[2.]]))
    target = torch.zeros_like(pred).detach()
    valid = torch.tensor([rank == 1,rank == 1])
    loss = latent_loss(pred,target,valid,normalize='none',distributed=True)
    loss.backward()
    assert torch.allclose(linear.weight.grad,torch.tensor([[1.5]]))
    mean = loss.detach().clone()
    torch.distributed.all_reduce(mean)
    mean /= torch.distributed.get_world_size()
    assert float(mean) == 3.
    rejected = False
    try:
        latent_loss(pred,target,torch.zeros_like(valid),normalize='none',distributed=True)
    except AssertionError:
        rejected = True
    assert rejected
    if rank == 0:
        report = {'ranks':2,'global_query_mean_loss':float(mean),'ddp_gradient':float(linear.weight.grad),
                  'analytical_expected_loss':3.,'analytical_expected_gradient':1.5,
                  'empty_local_rank_supported':True,'globally_empty_queries_rejected':True,
                  'loss_source_sha256':sha256(ROOT/'signrel/losses/latent.py')}
        (ROOT/'reports/ddp_empty_query_loss.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
