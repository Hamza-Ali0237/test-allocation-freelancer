import copy
from decimal import Decimal

from algo import naive_algorithm
from math_solution.solution import MathSolution
from pools import generate_assets_and_pools
from reward import get_rewards
from simulator import Simulator
from src.forest_allocation import RandomForestAllocation
from src.sgd_allocation import SGDAllocation

import cvxpy as cp
import numpy as np

model = RandomForestAllocation()
sgd = SGDAllocation()
math_solution = MathSolution()


def query_and_score(
        allocations,
        assets_and_pools):
    simulator = Simulator()
    simulator.initialize()

    if assets_and_pools is not None:
        simulator.init_data(init_assets_and_pools=copy.deepcopy(assets_and_pools))
    else:
        simulator.init_data()
        assets_and_pools = simulator.assets_and_pools

    # Adjust the scores based on allocations from engine
    apys, max_apy = get_rewards(
        simulator,
        allocations,
        assets_and_pools=assets_and_pools
    )

    # print(f"apys: {apys}")
    # print(f"max_apy:\n{max_apy}")
    return apys, max_apy


def convert_pool(asset_and_pools, e=1e18):
    new_pools = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}}
    new_asset_and_pools = {'total_assets': asset_and_pools['total_assets'] / e, 'pools': new_pools}
    for x, pools in asset_and_pools['pools'].items():
        new_asset_and_pools['pools'][x]['base_slope'] = pools.base_slope / e
        new_asset_and_pools['pools'][x]['kink_slope'] = pools.kink_slope / e
        new_asset_and_pools['pools'][x]['optimal_util_rate'] = pools.optimal_util_rate / e
        new_asset_and_pools['pools'][x]['borrow_amount'] = pools.borrow_amount / e
        new_asset_and_pools['pools'][x]['reserve_size'] = pools.reserve_size / e
        new_asset_and_pools['pools'][x]['base_rate'] = pools.base_rate / e
    # print(f"============>>> updated new_asset_and_pools:: {new_asset_and_pools}")
    return new_asset_and_pools


def convert_allocation(allocation, e=1e18):
    final_allocation = {str(k): float(Decimal(str(float(v) * e))) for k, v in allocation.items()}
    return final_allocation


def calc_simple_allocations(assets_and_pools):
    pools = assets_and_pools['pools']
    total_asset = assets_and_pools['total_assets']
    simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}
    return simple_allocations

def custom_allocation_strategy(assets_and_pools):
    pools = assets_and_pools['pools']
    total_assets = assets_and_pools['total_assets']

    scores = {}
    total_score = 0

    for pool_id, pool in pools.items():
        util_rate = pool.borrow_amount / pool.reserve_size
        deviation = abs(util_rate - pool.optimal_util_rate)
        
        # Introduce weighted factors
        weight_base_rate = 0.1
        weight_slope = 0.1
        weight_utilization = 0.1
        
        # Adjust the scoring formula
        interest_rate_increase = (weight_base_rate * pool.base_rate +
                                  weight_slope * (pool.base_slope + pool.kink_slope) +
                                  weight_utilization * (1 - deviation))
        
        # Use a less aggressive penalty for deviation
        score = interest_rate_increase / (1 + deviation)
        
        scores[pool_id] = score
        total_score += scores[pool_id]

    # Calculate allocations based on the scores
    allocations = {pool_id: (scores[pool_id] / total_score) * total_assets for pool_id in pools}
    return allocations

def compare():
    assets_and_pools = generate_assets_and_pools()

    naive_allocations = naive_algorithm(assets_and_pools)

    model_allocation = model.predict_allocation(convert_pool(assets_and_pools), model='old')

    math_allocation = math_solution.math_allocation(convert_pool(assets_and_pools))

    # Use the optimized custom allocation strategy
    custom_allocations = custom_allocation_strategy(assets_and_pools)
    
    # TODO:
    # Call your allocation solution here and add to allocation_list to compare the result with naive_allocations,
    # model_allocation and math_allocation
    allocation_list = [naive_allocations, convert_allocation(model_allocation), convert_allocation(math_allocation), custom_allocations]
    apys, max_apy = query_and_score(
        allocation_list,
        assets_and_pools)

    return apys


if __name__ == '__main__':
    ...
