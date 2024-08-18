import copy
from decimal import Decimal

from algo import naive_algorithm
from math_solution.solution import MathSolution
from OptimizedMath_Allocation import optimized_math_solution
from pools import generate_assets_and_pools
from reward import get_rewards
from simulator import Simulator
from src.sgd_allocation import SGDAllocation
from OptimizedMathSolution import OptimizedMathSolution
from HybridMathSolution import HybridOptMathSolution
from GradOptMathSolution import GradOptMathSolution

# Add imports for CatBoost
import pandas as pd
from catboost import CatBoostRegressor

sgd = SGDAllocation()
math_solution = MathSolution()
hyb_math_solution = HybridOptMathSolution()
GradOpt = GradOptMathSolution()

# Load the pre-trained CatBoost model using CatBoost's load_model method
catboost_model = CatBoostRegressor()
catboost_model.load_model('/root/humza_ali/test-allocation-freelancer/catboost_best_model.cbm')

def query_and_score(allocations, assets_and_pools):
    simulator = Simulator()
    simulator.initialize()

    if assets_and_pools is not None:
        simulator.init_data(init_assets_and_pools=copy.deepcopy(assets_and_pools))
    else:
        simulator.init_data()
        assets_and_pools = simulator.assets_and_pools

    apys, max_apy = get_rewards(simulator, allocations, assets_and_pools=assets_and_pools)
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
    return new_asset_and_pools

def convert_allocation(allocation, e=1e18):
    final_allocation = {str(k): float(Decimal(str(float(v) * e))) for k, v in allocation.items()}
    return final_allocation

def calc_simple_allocations(assets_and_pools):
    pools = assets_and_pools['pools']
    total_asset = assets_and_pools['total_assets']
    simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}
    return simple_allocations

def catboost_allocation(assets_and_pools):
    """Generate allocation using CatBoost model"""
    df = pd.DataFrame([assets_and_pools['pools']['0']])  # Convert the pool data into a DataFrame for the model
    catboost_allocations = catboost_model.predict(df)
    
    # Convert the model output into the allocation format
    allocation = {str(i): float(a) for i, a in enumerate(catboost_allocations)}
    return allocation

def compare():
    assets_and_pools = generate_assets_and_pools()

    math_allocation = math_solution.math_allocation(convert_pool(assets_and_pools))

    optimized_math_solution = OptimizedMathSolution()
    opt_math = optimized_math_solution.math_allocation(convert_pool(assets_and_pools))
    hyb_apy = hyb_math_solution.math_allocation(convert_pool(assets_and_pools))
    GradSol = GradOpt.math_allocation(convert_pool(assets_and_pools))

    # Use the CatBoost model to generate an allocation
    catboost_alloc = catboost_allocation(convert_pool(assets_and_pools))
    
    allocation_list = [
        convert_allocation(math_allocation),
        convert_allocation(opt_math),
        convert_allocation(GradSol),
        convert_allocation(hyb_apy),
        convert_allocation(catboost_alloc)
    ]
    
    apys, max_apy = query_and_score(allocation_list, assets_and_pools)

    return apys
