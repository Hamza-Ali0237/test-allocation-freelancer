import copy
from decimal import Decimal

import numpy as np
import torch
import tqdm
from torch.autograd import Variable
from torch.optim import Adam, AdamW, SGD

from pools import generate_assets_and_pools
from grad.reward_grad import get_rewards
from grad.simulator_grad import Simulator
from forest_allocation import RandomForestAllocation

model = RandomForestAllocation()


def query_and_score(allocations, assets_and_pools):
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
    apy = get_rewards(
        simulator,
        allocations=allocations,
    )
    return apy


def calc_sgb_allocations(assets_and_pools, init_allocations):
    for id, pool in assets_and_pools['pools'].items():
        del pool['pool_id']
        for k in pool.keys():
            pool[k] = torch.tensor(pool[k])

    allocations = {k: Variable(torch.tensor(v), requires_grad=True) for k, v in init_allocations.items()}
    return allocations


def model_test(assets_and_pools):
    from forward import query_and_score
    model_allocation = model.predict_allocation(assets_and_pools)
    apys, max_apy = query_and_score([model_allocation], assets_and_pools)
    print("Model APY:", max_apy)
    return max_apy, model_allocation


def sgd_test(assets_and_pools, init_allocations):
    sgd_allocations = maximize_apy_allocations(assets_and_pools, init_allocations)

    from forward import query_and_score

    sum_all = Decimal(str(sum(sgd_allocations.values())))
    total_assets = Decimal(str(assets_and_pools['total_assets']))
    allocations = {k: float(round_down(Decimal(str(alc)) / sum_all) * total_assets)
                   for k, alc in sgd_allocations.items()}
    apys, max_apy = query_and_score([allocations], assets_and_pools)
    print("SGD APY:", max_apy)
    return max_apy


def round_down(value, index=100000000):
    return ((Decimal(str(index)) * Decimal(value)) // Decimal(1)) / Decimal(index)


def maximize_apy_allocations(assets_and_pools, init_allocations):
    assets_and_pools = copy.deepcopy(assets_and_pools)
    allocations = calc_sgb_allocations(assets_and_pools, init_allocations)

    params = allocations.values()
    optimizer = SGD(params=params, lr=0.2)

    for epoch in range(100):
        optimizer.zero_grad()
        # print('Start epoch, grad', [alc.grad for alc in allocations.values()])
        sum_all = torch.tensor(float(sum(allocations.values())) / float(assets_and_pools['total_assets']),
                               requires_grad=False)
        # print(sum_all)
        normilized_allocations = {k: v / sum_all for k, v in allocations.items()}
        # print('Normalized allocations:', normilized_allocations)
        # print("Sum of allocations", sum(normilized_allocations.values()))
        apy = query_and_score(normilized_allocations, assets_and_pools)
        # print(f"APY: {apy:.2f}")
        apy = -apy
        apy.backward()
        # print([alc.grad for alc in allocations.values()])
        optimizer.step()
    normilized_allocations = {k: float(v) for k, v in allocations.items()}
    return normilized_allocations


def main():
    data = []
    for i in tqdm.tqdm(range(100)):
        assets_and_pools_np = generate_assets_and_pools()

        model_apy, model_allocations = model_test(assets_and_pools_np)
        sgd_apy = sgd_test(assets_and_pools_np, model_allocations)
        data.append((model_apy, sgd_apy))
    data = np.array(data)
    print("Model mean:", data[:, 0].mean())
    print("SGD mean:", data[:, 1].mean())


if __name__ == '__main__':
    main()
