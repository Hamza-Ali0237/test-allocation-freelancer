import copy
import random
import sys
import time
from decimal import Decimal
import numpy as np

from algo import naive_algorithm
from constants import SIMILARITY_THRESHOLD
from pools import generate_assets_and_pools
from reward import get_rewards, format_allocations
from simulator import Simulator
from src.forest_allocation import RandomForestAllocation
from src.sgd_allocation import SGDAllocation

model = RandomForestAllocation()
sgd = SGDAllocation()


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


def convert_pool(asset_and_pools):
    new_pools = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}}
    new_asset_and_pools = {'total_assets': asset_and_pools['total_assets'] / 1e18, 'pools': new_pools}
    for x, pools in asset_and_pools['pools'].items():
        new_asset_and_pools['pools'][x]['base_slope'] = pools.base_slope / 1e18
        new_asset_and_pools['pools'][x]['kink_slope'] = pools.kink_slope / 1e18
        new_asset_and_pools['pools'][x]['optimal_util_rate'] = pools.optimal_util_rate / 1e18
        new_asset_and_pools['pools'][x]['borrow_amount'] = pools.borrow_amount / 1e18
        new_asset_and_pools['pools'][x]['reserve_size'] = pools.reserve_size / 1e18
        new_asset_and_pools['pools'][x]['base_rate'] = pools.base_rate / 1e18
    # print(f"============>>> updated new_asset_and_pools:: {new_asset_and_pools}")
    return new_asset_and_pools


def convert_allocation(allocation):
    final_allocation = {str(k): float(Decimal(str(float(v) * 1e18))) for k, v in allocation.items()}
    return final_allocation


def calc_simple_allocations(assets_and_pools):
    pools = assets_and_pools['pools']
    total_asset = assets_and_pools['total_assets']
    simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}
    return simple_allocations


def distance(_alloc_a, _alloc_b, assets_and_pools):
    total_assets = assets_and_pools["total_assets"]
    alloc_a = np.array(
        list(format_allocations(_alloc_a, assets_and_pools).values()),
        dtype=np.float32,
    )
    alloc_b = np.array(
        list(format_allocations(_alloc_b, assets_and_pools).values()),
        dtype=np.float32,
    )
    head = np.linalg.norm(
        alloc_a - alloc_b
    )

    tail = np.sqrt(float(2 * total_assets ** 2))
    return head / tail


def cal_combination(k, n):
    import itertools
    elements = list(range(n))
    combinations = list(itertools.combinations(elements, k))
    return combinations


def scatter(allocation, assets_and_pools):
    e = 0.01
    list_alloc = list(format_allocations(allocation, assets_and_pools).values())


def old_calculation_scatter(subtract, plus, allocation):
    new_allocation = copy.deepcopy(allocation)

    for i in subtract:
        value = new_allocation[str(i)]
        if value > 10:
            new_allocation[str(i)] = value - 10

    for i in plus:
        new_allocation[str(i)] = new_allocation[str(i)] + 10

    return new_allocation


def calculation_scatter(subtract, plus, allocation):
    new_allocation = copy.deepcopy(allocation)

    n = 10 - len(subtract) - len(plus)
    random_values = [random.choice([0.01, 0.02, 0.03, 0.04, 0.05, -0.01, -0.02, -0.03, -0.04, -0.05]) for _ in range(n - 1)]
    random_values.append(-sum(random_values))
    index = 0
    for i in range(10):
        value = new_allocation[str(i)]
        if i in subtract:
            if value > 10:
                new_allocation[str(i)] = value - 10
        elif i in plus:
            new_allocation[str(i)] = new_allocation[str(i)] + 10
        else:
            new_allocation[str(i)] = new_allocation[str(i)] + random_values[index]
            index += 1

    return new_allocation


def check_distance(allocation_list, assets_and_pools):
    for i in range(len(allocation_list)):
        for j in range(i + 1, len(allocation_list)):
            alloc_a = allocation_list[i]
            alloc_b = allocation_list[j]
            dis = distance(convert_allocation(alloc_a), convert_allocation(alloc_b), assets_and_pools)
            if dis <= SIMILARITY_THRESHOLD:
                print(f"SIMILARITY THRESHOLD REACHING - PUNISHING 👊😠")
                sys.exit()


def verify_distance():
    assets_and_pools = generate_assets_and_pools()
    model_allocation = model.predict_allocation(convert_pool(assets_and_pools))
    first_model_allocation = calculation_scatter([0, 2], [1, 3], model_allocation)
    second_model_allocation = calculation_scatter([1, 3], [0, 2], model_allocation)
    seventh = calculation_scatter([0, 1], [2, 3], model_allocation)
    eighth = calculation_scatter([2, 3], [0, 1], model_allocation)
    ninth = calculation_scatter([0, 3], [1, 2], model_allocation)
    tenth = calculation_scatter([1, 2], [0, 3], model_allocation)

    third_model_allocation = calculation_scatter([4, 6], [5, 7], model_allocation)
    forth_model_allocation = calculation_scatter([5, 7], [4, 6], model_allocation)
    eleventh = calculation_scatter([4, 5], [6, 7], model_allocation)
    twelveth = calculation_scatter([6, 7], [4, 5], model_allocation)
    thirteenth = calculation_scatter([4, 7], [5, 6], model_allocation)
    forthteenth = calculation_scatter([5, 6], [4, 7], model_allocation)

    fifth_model_allocation = calculation_scatter([0, 1], [8, 9], model_allocation)
    sixth_model_allocation = calculation_scatter([8, 9], [0, 1], model_allocation)

    aa = calculation_scatter([0, 1], [4, 5], model_allocation)
    ab = calculation_scatter([4, 5], [0, 1], model_allocation)

    ac = calculation_scatter([0, 1], [6, 7], model_allocation)
    ad = calculation_scatter([6, 7], [0, 1], model_allocation)

    simple_allocations = naive_algorithm(assets_and_pools)
    for k, v in simple_allocations.items():
        v = v / 1e18
        simple_allocations[k] = v

    allocation_list = [model_allocation,
                       first_model_allocation,
                       second_model_allocation,
                       third_model_allocation,
                       forth_model_allocation,
                       fifth_model_allocation,
                       sixth_model_allocation,
                       seventh,
                       eighth,
                       ninth,
                       tenth,
                       eleventh,
                       twelveth,
                       thirteenth,
                       forthteenth,
                       simple_allocations,
                       aa,
                       ab,
                       ac,
                       ad
                       ]
    check_distance(allocation_list, assets_and_pools)
    print(f"DISTANCE IS OKAY")


def verify_distance_index():
    indexes = [[[0, 1], [2, 3]], [[0, 1], [4, 5]], [[0, 1], [6, 7]], [[0, 1], [8, 9]], [[0, 2], [4, 6]],
               [[0, 2], [5, 7]], [[0, 3], [2, 8]], [[0, 3], [4, 7]], [[0, 3], [5, 6]], [[0, 4], [2, 9]],
               [[0, 5], [3, 9]], [[0, 5], [4, 8]], [[0, 7], [6, 8]], [[1, 2], [0, 4]], [[1, 2], [5, 6]],
               [[1, 2], [7, 8]], [[1, 3], [0, 2]], [[1, 3], [4, 6]], [[1, 3], [5, 7]], [[1, 4], [0, 3]],
               [[1, 4], [7, 9]], [[1, 5], [0, 8]], [[1, 5], [2, 9]], [[1, 6], [0, 5]], [[1, 6], [3, 8]],
               [[1, 7], [0, 6]], [[1, 8], [0, 7]], [[1, 9], [4, 8]], [[2, 3], [0, 1]], [[2, 3], [4, 5]],
               [[2, 3], [6, 7]], [[2, 3], [8, 9]], [[2, 4], [0, 7]], [[2, 4], [1, 3]], [[2, 5], [0, 3]],
               [[2, 5], [1, 4]], [[2, 6], [0, 8]], [[2, 6], [1, 5]], [[2, 6], [4, 9]], [[2, 7], [0, 5]],
               [[2, 7], [1, 6]], [[2, 8], [0, 6]], [[2, 8], [1, 7]], [[2, 9], [1, 8]], [[3, 4], [0, 8]],
               [[3, 4], [1, 2]], [[3, 5], [0, 4]], [[3, 5], [1, 8]], [[3, 6], [0, 9]], [[3, 6], [1, 4]],
               [[3, 6], [2, 5]], [[3, 7], [1, 5]], [[3, 7], [2, 4]], [[3, 8], [0, 5]], [[3, 8], [1, 6]],
               [[3, 8], [2, 7]], [[3, 9], [0, 6]], [[3, 9], [1, 7]], [[4, 5], [0, 1]], [[4, 5], [2, 3]],
               [[4, 5], [6, 7]], [[4, 5], [8, 9]], [[4, 6], [0, 2]], [[4, 6], [1, 8]], [[4, 6], [3, 5]],
               [[4, 7], [2, 5]], [[4, 7], [3, 6]], [[4, 8], [1, 5]], [[4, 8], [2, 6]], [[4, 8], [3, 7]],
               [[4, 9], [0, 5]], [[4, 9], [1, 6]], [[4, 9], [2, 7]], [[4, 9], [3, 8]], [[5, 6], [1, 2]],
               [[5, 6], [3, 4]], [[5, 7], [0, 2]], [[5, 7], [1, 3]], [[5, 7], [4, 6]], [[5, 8], [2, 4]],
               [[5, 8], [3, 6]], [[5, 9], [0, 7]], [[5, 9], [2, 6]], [[6, 7], [0, 1]], [[6, 7], [2, 3]],
               [[6, 7], [4, 5]], [[6, 7], [8, 9]], [[6, 8], [0, 3]], [[6, 8], [4, 7]], [[6, 9], [0, 4]],
               [[6, 9], [1, 3]], [[6, 9], [2, 8]], [[6, 9], [5, 7]], [[7, 8], [0, 4]], [[7, 8], [1, 2]],
               [[7, 8], [3, 5]], [[7, 9], [0, 3]], [[7, 9], [1, 4]], [[7, 9], [5, 6]], [[8, 9], [0, 1]],
               [[8, 9], [2, 3]], [[8, 9], [4, 5]], [[8, 9], [6, 7]]]
    # indexes.append([[2, 4], [8, 9]])
    print(len(indexes))
    assets_and_pools = generate_assets_and_pools()
    model_allocation = model.predict_allocation(convert_pool(assets_and_pools))
    allocation_list = [model_allocation]
    count = 0
    for ind in indexes:
        if count < 25:
            sct = old_calculation_scatter(ind[0], ind[1], model_allocation)
        else:
            sct = calculation_scatter(ind[0], ind[1], model_allocation)
        count +=1
        allocation_list.append(sct)
    check_distance(allocation_list, assets_and_pools)
    print(f"DISTANCE IS OKAY")


def main():
    indexes = [[[0, 1], [2, 3]], [[0, 1], [4, 5]], [[0, 1], [6, 7]], [[0, 1], [8, 9]], [[0, 2], [4, 6]],
               [[0, 2], [5, 7]], [[0, 3], [2, 8]], [[0, 3], [4, 7]], [[0, 3], [5, 6]], [[0, 4], [2, 9]],
               [[0, 5], [3, 9]], [[0, 5], [4, 8]], [[0, 7], [6, 8]], [[1, 2], [0, 4]], [[1, 2], [5, 6]],
               [[1, 2], [7, 8]], [[1, 3], [0, 2]], [[1, 3], [4, 6]], [[1, 3], [5, 7]], [[1, 4], [0, 3]],
               [[1, 4], [7, 9]], [[1, 5], [0, 8]], [[1, 5], [2, 9]], [[1, 6], [0, 5]], [[1, 6], [3, 8]],
               [[1, 7], [0, 6]], [[1, 8], [0, 7]], [[1, 9], [4, 8]], [[2, 3], [0, 1]], [[2, 3], [4, 5]],
               [[2, 3], [6, 7]], [[2, 3], [8, 9]], [[2, 4], [0, 7]], [[2, 4], [1, 3]], [[2, 5], [0, 3]],
               [[2, 5], [1, 4]], [[2, 6], [0, 8]], [[2, 6], [1, 5]], [[2, 6], [4, 9]], [[2, 7], [0, 5]],
               [[2, 7], [1, 6]], [[2, 8], [0, 6]], [[2, 8], [1, 7]], [[2, 9], [1, 8]], [[3, 4], [0, 8]],
               [[3, 4], [1, 2]], [[3, 5], [0, 4]], [[3, 5], [1, 8]], [[3, 6], [0, 9]], [[3, 6], [1, 4]],
               [[3, 6], [2, 5]], [[3, 7], [1, 5]], [[3, 7], [2, 4]], [[3, 8], [0, 5]], [[3, 8], [1, 6]],
               [[3, 8], [2, 7]], [[3, 9], [0, 6]], [[3, 9], [1, 7]], [[4, 5], [0, 1]], [[4, 5], [2, 3]],
               [[4, 5], [6, 7]], [[4, 5], [8, 9]], [[4, 6], [0, 2]], [[4, 6], [1, 8]], [[4, 6], [3, 5]],
               [[4, 7], [2, 5]], [[4, 7], [3, 6]], [[4, 8], [1, 5]], [[4, 8], [2, 6]], [[4, 8], [3, 7]],
               [[4, 9], [0, 5]], [[4, 9], [1, 6]], [[4, 9], [2, 7]], [[4, 9], [3, 8]], [[5, 6], [1, 2]],
               [[5, 6], [3, 4]], [[5, 7], [0, 2]], [[5, 7], [1, 3]], [[5, 7], [4, 6]], [[5, 8], [2, 4]],
               [[5, 8], [3, 6]], [[5, 9], [0, 7]], [[5, 9], [2, 6]], [[6, 7], [0, 1]], [[6, 7], [2, 3]],
               [[6, 7], [4, 5]], [[6, 7], [8, 9]], [[6, 8], [0, 3]], [[6, 8], [4, 7]], [[6, 9], [0, 4]],
               [[6, 9], [1, 3]], [[6, 9], [2, 8]], [[6, 9], [5, 7]], [[7, 8], [0, 4]], [[7, 8], [1, 2]],
               [[7, 8], [3, 5]], [[7, 9], [0, 3]], [[7, 9], [1, 4]], [[7, 9], [5, 6]], [[8, 9], [0, 1]],
               [[8, 9], [2, 3]], [[8, 9], [4, 5]], [[8, 9], [6, 7]]]

    assets_and_pools = generate_assets_and_pools()

    # simple_allocations = naive_algorithm(assets_and_pools)
    # t1 = time.time()

    model_allocation = model.predict_allocation(convert_pool(assets_and_pools))
    # sgd_allocation = sgd.predict_allocation(convert_pool(assets_and_pools), initial_allocations=model_allocation)

    allocation_list = [convert_allocation(model_allocation)]
    for ind in indexes:
        sct = calculation_scatter(ind[0], ind[1], model_allocation)
        allocation_list.append(convert_allocation(sct))
    new_model_allocation = calculation_scatter([1, 0], [6, 7], model_allocation)

    second_model_allocation = calculation_scatter([5, 4], [1, 0], model_allocation)

    t2 = time.time()

    apys, max_apy = query_and_score(
        allocation_list,
        assets_and_pools)

    t3 = time.time()
    # print(f"Query: {(t3 - t2) * 1000:.2f} ms")
    # print(
    #     f"distance naive and model: {distance(simple_allocations, convert_allocation(model_allocation), assets_and_pools)}")

    return apys


if __name__ == '__main__':
    while True:
        verify_distance_index()
    # my_list = cal_combination(2, 10)
    #
    # print(len(my_list))
    # print(my_list)
    # result_dict = {}
    #
    # for a in my_list:
    #     for b in my_list:
    #         if b[0] in a or b[1] in a:
    #             continue
    #         key = str(a[0]) + '_' + str(a[1])
    #         temp_list = result_dict.get(key)
    #         if temp_list is None:
    #             temp_list = [[b[0], b[1]]]
    #             result_dict[key] = temp_list
    #             continue
    #
    #         is_exist = False
    #         for t_list in temp_list:
    #             if b[0] in t_list or b[1] in t_list:
    #                 is_exist = True
    #                 break
    #         if is_exist:
    #             continue
    #
    #         temp_list.append([b[0], b[1]])
    #         result_dict[key] = temp_list
    #
    # print(result_dict)

    # while True:
    #     verify_distance()

    # apys = calculation_scatter()
    # print(f"apy = {apys}")

#     import json
#     import numpy as np
#
#     # order = ['Good', 'Simple', 'RandomForest', 'SGD', 'DN']
#     order = ['Good', 'SGD']
#     data = []
#     with open('charm_training_data.txt', 'r') as f:
#         lines = f.readlines()
#     for i in tqdm.tqdm(range(len(lines) // 4)):
#         pools = json.loads(lines[i * 4 + 0].replace('\'', '"'))
#         good_allocations = json.loads(lines[i * 4 + 1].replace('\'', '"'))
#         apy = json.loads(lines[i * 4 + 2].replace('\'', '"'))
#         apys = main(pools, good_allocations)
#         data.append(apys)
#         # if i == 20:
#         #     break
#     data = np.array(data)
#     for i, apy in enumerate(data.mean(axis=0)):
#         print(f"{order[i]}:", f'{apy:.2f}')
