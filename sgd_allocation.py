import copy
from decimal import Decimal

import torch
from torch import optim

from forest_allocation import RandomForestAllocation
from grad.module import Model


class SGDAllocation:
    def __init__(self, epoch=30, lr=1e-3, device='cpu'):
        self.epoch = epoch
        self.lr = lr
        self._device = device
        self._model = RandomForestAllocation()

    def convert_pool_to_tensor(self, assets_and_pools):
        columns = ['base_rate', 'base_slope', 'borrow_amount', 'kink_slope', 'optimal_util_rate', 'reserve_size']
        data = []
        for id, pool in assets_and_pools['pools'].items():
            del pool['pool_id']
            data.append([pool[column] for column in columns])
        pools = torch.tensor(data, device=self._device, dtype=torch.float32)
        return pools

    def _maximize_apy_allocations(self, assets_and_pools, init_allocations):
        total_assets = torch.tensor(assets_and_pools['total_assets'], device=self._device)
        pools = self.convert_pool_to_tensor(assets_and_pools)

        model = Model(init_allocations)
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)

        for epoch in range(self.epoch):
            optimizer.zero_grad()
            apy = model(pools, total_assets)
            apy = -apy
            apy.backward()
            optimizer.step()

        allocations = model.allocations / torch.sum(model.allocations, dim=0) * total_assets
        return allocations

    def predict_allocation(self, assets_and_pools, initial_allocations=None):
        if initial_allocations is None:
            initial_allocations = self._model.predict_allocation(copy.deepcopy(assets_and_pools))
        allocations = self._maximize_apy_allocations(copy.deepcopy(assets_and_pools), initial_allocations)

        total_assets = Decimal(assets_and_pools['total_assets'])

        allocations = [Decimal(float(allocation)) for allocation in allocations.data.cpu().numpy()]
        sum_allocs = Decimal(sum(allocations))
        allocations = [self.round_down(allocation / sum_allocs) * total_assets for allocation in allocations]
        normalized_allocations = {str(k): float(v) for k, v in enumerate(allocations)}
        return normalized_allocations

    def round_down(self, value, index=100000000):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))
