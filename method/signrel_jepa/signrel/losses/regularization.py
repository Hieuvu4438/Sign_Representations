import torch
import torch.nn.functional as F


def variance_covariance(z, gamma=1.):
    if z.shape[0]<2:
        raise AssertionError("Variance/covariance requires at least two samples")
    z = z.float()
    var = F.relu(gamma-torch.sqrt(z.var(0,unbiased=False)+1e-4)).mean()
    centered = z-z.mean(0)
    cov = centered.T@centered/(z.shape[0]-1)
    off = cov-torch.diag_embed(cov.diag())
    return var,off.square().sum()/z.shape[1]
