from datetime import time
import time

import tqdm

from misc_custom import call_allocation_algorithm
from pools import generate_assets_and_pools
from reward import get_rewards
from simulator import Simulator
from src.forest_allocation import RandomForestAllocation
from src.sgd_allocation import SGDAllocation

model = RandomForestAllocation()
sgd = SGDAllocation()


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

    # print(f"apys: {apys}")
    # print(f"max_apy:\n{max_apy}")
    return apys, max_apy


def calc_simple_allocations(assets_and_pools):
    pools = assets_and_pools['pools']
    total_asset = assets_and_pools['total_assets']
    simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}
    return simple_allocations


def main(pools, good_allocations):
    assets_and_pools = generate_assets_and_pools()
    assets_and_pools['pools'] = pools

    simple_allocations = calc_simple_allocations(assets_and_pools)
    t1 = time.time()
    model_allocation = model.predict_allocation(assets_and_pools)
    sgd_allocation = sgd.predict_allocation(assets_and_pools, initial_allocations=model_allocation)
    t2 = time.time()
    # print(f"{(t2 - t1) * 1000:.2f} ms")
    dn_allocation = call_allocation_algorithm(assets_and_pools)
    # apys, max_apy = query_and_score(
    #     [good_allocations, simple_allocations, model_allocation, sgd_allocation, dn_allocation],
    #     assets_and_pools)
    apys, max_apy = query_and_score(
        [good_allocations, sgd_allocation],
        assets_and_pools)
    t3 = time.time()
    # print(f"Query: {(t3 - t2) * 1000:.2f} ms")
    # apys, max_apy = query_and_score([simple_allocations], assets_and_pools)
    # print(f"max_apy: {max_apy}")
    return apys


if __name__ == '__main__':
    import json
    import numpy as np

    # order = ['Good', 'Simple', 'RandomForest', 'SGD', 'DN']
    order = ['Good', 'SGD']
    data = []
    with open('charm_training_data.txt', 'r') as f:
        lines = f.readlines()
    for i in tqdm.tqdm(range(len(lines) // 4)):
        pools = json.loads(lines[i * 4 + 0].replace('\'', '"'))
        good_allocations = json.loads(lines[i * 4 + 1].replace('\'', '"'))
        apy = json.loads(lines[i * 4 + 2].replace('\'', '"'))
        apys = main(pools, good_allocations)
        data.append(apys)
        # if i == 20:
        #     break
    data = np.array(data)
    for i, apy in enumerate(data.mean(axis=0)):
        print(f"{order[i]}:", f'{apy:.2f}')
