import logging

from forest_allocation import RandomForestAllocation
from misc_custom import call_allocation_algorithm
from pools import generate_assets_and_pools
from reward import get_rewards
from simulator import Simulator

model = RandomForestAllocation()


def query_and_score(
        allocations,
        assets_and_pools):
    # intialize simulator
    simulator = Simulator()
    simulator.initialize()
    # initialize simulator data
    # if there is organic info then generate synthetic info
    if assets_and_pools is not None:
        simulator.init_data(init_assets_and_pools=assets_and_pools)
    else:
        simulator.init_data()

    # Adjust the scores based on allocations from engine
    apys, max_apy = get_rewards(
        simulator,
        allocations_list=allocations,
    )

    print(f"apys: {apys}")
    print(f"max_apy:\n{max_apy}")
    return apys, max_apy


def main():
    assets_and_pools = generate_assets_and_pools()
    pools = assets_and_pools['pools']
    total_asset = assets_and_pools['total_assets']
    simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}
    model_allocation = model.predict_allocation(assets_and_pools)
    dn_allocation = call_allocation_algorithm(assets_and_pools)

    apys, max_apy = query_and_score([simple_allocations, model_allocation, dn_allocation], assets_and_pools)
    # print(f"max_apy: {max_apy}")
    return apys


if __name__ == '__main__':
    main()
