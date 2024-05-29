from typing import Dict

import torch
from torch import Tensor

from constants import *


def borrow_rate(util_rate, pools: torch.Tensor) -> Tensor:
    interest_rate = torch.where(
        util_rate < pools[:, 4],
        pools[:, 0] + (util_rate / pools[:, 4]) * pools[:, 1],
        pools[:, 0] + pools[:, 1] + ((util_rate - pools[:, 4]) / (1 - pools[:, 4])) * pools[:, 3])
    return interest_rate


def supply_rate(util_rate, pool: torch.Tensor) -> torch.Tensor:
    return util_rate * borrow_rate(util_rate, pool)


class Simulator(object):
    COLUMNS = ['borrow_amount', 'reserve_size', 'borrow_rate']

    def __init__(
            self,
            timesteps=TIMESTEPS,
            reversion_speed=REVERSION_SPEED,
            stochasticity=STOCHASTICITY,
            seed=None,
    ):
        self.timesteps = timesteps
        self.reversion_speed = reversion_speed
        self.stochasticity = stochasticity
        self.seed = seed

    # initializes data - by default these are randomly generated
    def init_data(self, init_pools):
        pool_history = torch.stack(tensors=[
            init_pools[:, 2],
            init_pools[:, 5],
            borrow_rate(init_pools[:, 2] / init_pools[:, 5], init_pools)
        ], dim=1)
        return pool_history

    # update the reserves in the pool with given allocations
    def update_reserves_with_allocs(self, allocations, pools, pool_history):
        reserve_size = pool_history[:, 1] + allocations
        pool_history = torch.stack(tensors=[
            pool_history[:, 0],
            reserve_size,
            borrow_rate(pool_history[:, 0] / reserve_size, pools)
        ], dim=1)
        return pool_history

    # initialize pools
    # Function to update borrow amounts and other pool params based on reversion rate and stochasticity
    def generate_new_pool_data(self, pools, pool_history):
        curr_borrow_amounts = pool_history[:, 0]
        curr_reserve_sizes = pool_history[:, 1]
        curr_borrow_rates = pool_history[:, 2]

        median_rate = torch.median(curr_borrow_rates)  # Calculate the median borrow rate
        noise = 0
        rate_changes = (-self.reversion_speed * (curr_borrow_rates - median_rate) + noise)  # Mean reversion principle
        new_borrow_amounts = curr_borrow_amounts + rate_changes * curr_borrow_amounts  # Update the borrow amounts
        amounts = torch.clip(new_borrow_amounts, min=torch.zeros_like(curr_reserve_sizes),
                             max=curr_reserve_sizes)  # Ensure borrow amounts do not exceed reserves
        pool_history = torch.stack([
            amounts,
            curr_reserve_sizes,
            borrow_rate(amounts / curr_reserve_sizes, pools)
        ], dim=1)
        return pool_history

    def calc_apy(self, allocations, pools, pool_history):
        pools_yield = allocations * supply_rate(pool_history[:, 0] / pool_history[:, 1], pools)
        pool_yield = torch.sum(pools_yield)
        return pool_yield

    # run simulation
    def run(self, allocations, pools, initial_balance):
        pct_yield = 0
        pool_history = self.init_data(pools)
        pool_history = self.update_reserves_with_allocs(allocations, pools, pool_history)
        pct_yield += self.calc_apy(allocations, pools, pool_history)
        for ts in range(1, self.timesteps):
            self.generate_new_pool_data(pools, pool_history)
            pct_yield += self.calc_apy(allocations, pools, pool_history)
        pct_yield /= initial_balance
        aggregate_apy = (pct_yield / self.timesteps) * 365
        return aggregate_apy
