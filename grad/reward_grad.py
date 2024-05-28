import copy
import logging
import math
import sys
from typing import List, Dict, Any, Union
from constants import QUERY_TIMEOUT, STEEPNESS, DIV_FACTOR, NUM_POOLS
from misc import check_allocations, supply_rate


def sigmoid_scale(
        axon_time: float,
        num_pools: int = NUM_POOLS,
        steepness: float = STEEPNESS,
        div_factor: float = DIV_FACTOR,
        timeout: float = QUERY_TIMEOUT,
) -> float:
    offset = -float(num_pools) / div_factor
    return (
        (1 / (1 + math.exp(steepness * axon_time + offset)))
        if axon_time < timeout
        else 0
    )


def reward(
        max_apy: float,
        miner_apy: float,
        axon_time: float,
        num_pools: int = NUM_POOLS,
) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    return (0.2 * sigmoid_scale(axon_time, num_pools=num_pools)) + (
            0.8 * miner_apy / max_apy
    )


def calculate_aggregate_apy(
        allocations: Dict[str, float],
        assets_and_pools: Dict[str, Union[Dict[str, float], float]],
        timesteps: int,
        pool_history: Dict[str, Dict[str, Any]],
):
    """
    Calculates aggregate yields given intial assets and pools, pool history, and number of timesteps
    """

    # calculate aggregate yield
    initial_balance = assets_and_pools["total_assets"]
    pct_yield = 0
    for pools in pool_history:
        curr_yield = 0
        for uid, allocs in allocations.items():
            pool_data = pools[uid]
            util_rate = pool_data["borrow_amount"] / pool_data["reserve_size"]
            pool_yield = allocs * supply_rate(util_rate, assets_and_pools["pools"][uid])
            curr_yield += pool_yield
        pct_yield += curr_yield

    pct_yield /= initial_balance
    aggregate_apy = (
                            pct_yield / timesteps
                    ) * 365  # for simplicity each timestep is a day in the simulator

    return aggregate_apy


def get_rewards(simulator, allocations):
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    - allocs: miner allocations along with their respective yields
    """

    init_assets_and_pools = copy.deepcopy(simulator.assets_and_pools)

    # reset simulator for next run
    simulator.reset()

    # miner does not appear to be cheating - so we init simulator data
    simulator.init_data(copy.deepcopy(init_assets_and_pools), allocations)

    # update reserves given allocation
    simulator.update_reserves_with_allocs()
    simulator.run()
    aggregate_apy = calculate_aggregate_apy(
        allocations,
        init_assets_and_pools,
        simulator.timesteps,
        simulator.pool_history,
    )

    return aggregate_apy
