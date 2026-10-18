from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from signrepr.losses.supcon import supcon_terms
from signrepr.training.distributed import global_sum, loss_for_backward


def supcon_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    torch.manual_seed(42)
    module = torch.nn.Linear(3, 2)
    model = DistributedDataParallel(module)
    x = torch.tensor([[1., .1, .2], [.2, 1., .4], [.1, .2, 1.], [.9, .1, .2]])
    labels = torch.tensor([0, 0, 1, 1])
    valid = torch.tensor([True, True, True, False])
    sl = slice(rank*2, rank*2+2)
    embedding = torch.nn.functional.normalize(model(x[sl]), dim=-1)
    num, den = supcon_terms(embedding, labels[sl], valid[sl])
    loss_for_backward(num, global_sum(den), 2).backward()
    torch.save([p.grad for p in module.parameters()], Path(output)/f'rank{rank}.pt')
    dist.destroy_process_group()


def test_supcon_autograd_gather_local_anchor_scaling(tmp_path):
    mp.spawn(supcon_worker, args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    x = torch.tensor([[1., .1, .2], [.2, 1., .4], [.1, .2, 1.], [.9, .1, .2]])
    embedding = torch.nn.functional.normalize(model(x), dim=-1)
    num, den = supcon_terms(embedding, torch.tensor([0, 0, 1, 1]), torch.tensor([True, True, True, False]))
    (num/den).backward()
    for rank in range(2):
        grads = torch.load(tmp_path/f'rank{rank}.pt', weights_only=True)
        for grad, p in zip(grads, model.parameters()):
            torch.testing.assert_close(grad, p.grad, atol=2e-5, rtol=2e-4)
