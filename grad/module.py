import torch

from grad.simulator_grad import run


class Model(torch.nn.Module):
    def __init__(self, init_allocations):
        super(Model, self).__init__()
        self.allocations = torch.nn.Parameter(
            torch.tensor(list(init_allocations.values()), dtype=torch.float32))

    # @torch.jit.script
    def projection_simplex_sort(self, v, z=1):
        n_features = v.shape[0]
        u = torch.sort(v, descending=True)[0]
        cssv = torch.cumsum(u, dim=0) - z
        ind = torch.arange(n_features).float() + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = torch.clamp(v - theta, min=0)
        return w

    def forward(self, pools, total_assets):
        x = self.allocations * total_assets
        apy = run(x, pools, total_assets)
        return apy
