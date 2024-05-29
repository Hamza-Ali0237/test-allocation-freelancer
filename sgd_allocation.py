import copy
import time
from decimal import Decimal

import torch
import numpy as np
from torch import optim
from torch.autograd import Variable

from forest_allocation import RandomForestAllocation
from grad.simulator_grad import Simulator


class SGDAllocation:
    def __init__(self, epoch=30, lr=0.001, device='cpu'):
        self.epoch = epoch
        self.lr = lr
        self._device = device
        self._simulator = Simulator()
        self._model = RandomForestAllocation()

    def convert_pool_to_tensor(self, assets_and_pools, init_allocations):
        columns = ['base_rate', 'base_slope', 'borrow_amount', 'kink_slope', 'optimal_util_rate', 'reserve_size']
        data = []
        for id, pool in assets_and_pools['pools'].items():
            del pool['pool_id']
            data.append([pool[column] for column in columns])
        pools = torch.tensor(data, device=self._device)

        allocations = Variable(torch.tensor(list(init_allocations.values()), device=self._device), requires_grad=True)
        return pools, allocations

    def _maximize_apy_allocations(self, assets_and_pools, init_allocations):
        total_assets = assets_and_pools['total_assets']
        pools, allocations = self.convert_pool_to_tensor(assets_and_pools, init_allocations)

        optimizer = optim.SGD(params=[allocations], lr=self.lr)

        for epoch in range(self.epoch):
            optimizer.zero_grad()
            _allocations = allocations / torch.sum(allocations)
            _allocations *= total_assets
            t1 = time.time()
            apy = self._simulator.run(_allocations, pools, total_assets)
            # torch.jit.script(query_and_score, example_inputs=(allocations, assets_and_pools))
            t2 = time.time()
            print(f"SGD epoch time: {(t2 - t1) * 1000:.2f} ms", float(apy))
            apy = -apy
            apy.backward()
            optimizer.step()
        # with torch.no_grad():
        #     allocations += torch.min(allocations)
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
