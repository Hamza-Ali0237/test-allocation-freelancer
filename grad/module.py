import torch

from grad.simulator_grad import run


class Model(torch.nn.Module):
    def __init__(self, init_allocations):
        super(Model, self).__init__()
        self.allocations = torch.nn.Parameter(
            torch.tensor(list(init_allocations.values()), dtype=torch.float32))
        self._softmax = torch.nn.Softmax(dim=0)

    def forward(self, pools, total_assets):
        x = self.allocations
        x = x / torch.sum(x, dim=0)
        x = x * total_assets
        apy = run(x, pools, total_assets)
        return apy
