import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def variance_covariance(z, target_std=1.0):
    if dist.is_initialized() and dist.get_world_size() > 1:
        z = torch.cat(all_gather(z), dim=0)
    if z.shape[0] < 2:
        raise ValueError("regularization requires stats batch >=2; accumulation is not stats batch")
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.double() if z.dtype == torch.float64 else z.float()
        centered = z-z.mean(0)
        var = centered.square().sum(0)/(len(z)-1)
        l_var = torch.relu(target_std-(var+1e-4).sqrt()).mean()
        cov = centered.T@centered/(len(z)-1)
        offdiag = cov - torch.diag_embed(cov.diag())
        l_cov = offdiag.square().sum()/z.shape[1]
    return l_var, l_cov
