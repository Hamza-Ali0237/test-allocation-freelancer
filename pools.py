import typing
from constants import *
import numpy as np


# TODO: add different interest rate models in the future - we use a single simple model for now
def generate_assets_and_pools(rng_gen=np.random) -> typing.Dict:  # generate pools
    assets_and_pools = {}
    pools = {
        str(x): {
            "pool_id": str(x),
            "base_rate": randrange_float(
                MIN_BASE_RATE, MAX_BASE_RATE, BASE_RATE_STEP, rng_gen=rng_gen
            ),
            "base_slope": randrange_float(
                MIN_SLOPE, MAX_SLOPE, SLOPE_STEP, rng_gen=rng_gen
            ),
            "kink_slope": randrange_float(
                MIN_KINK_SLOPE, MAX_KINK_SLOPE, SLOPE_STEP, rng_gen=rng_gen
            ),  # kink rate - kicks in after pool hits optimal util rate
            "optimal_util_rate": randrange_float(
                MIN_OPTIMAL_RATE, MAX_OPTIMAL_RATE, OPTIMAL_UTIL_STEP, rng_gen=rng_gen
            ),  # optimal util rate - after which the kink slope kicks in
            "borrow_amount": format_num_prec(
                POOL_RESERVE_SIZE
                * randrange_float(
                    MIN_UTIL_RATE, MAX_UTIL_RATE, UTIL_RATE_STEP, rng_gen=rng_gen
                )
            ),  # initial borrowed amount from pool
            "reserve_size": POOL_RESERVE_SIZE,
        }
        for x in range(NUM_POOLS)
    }

    assets_and_pools["total_assets"] = TOTAL_ASSETS
    assets_and_pools["pools"] = pools

    return assets_and_pools


# generate intial allocations for pools
def generate_initial_allocations_for_pools(
        assets_and_pools: typing.Dict, size: int = NUM_POOLS, rng_gen=np.random
) -> typing.Dict:
    nums = np.ones(size)
    allocs = nums / np.sum(nums) * assets_and_pools["total_assets"]
    allocations = {str(i): alloc for i, alloc in enumerate(allocs)}

    return allocations


def randrange_float(
        start,
        stop,
        step,
        sig: int = GREEDY_SIG_FIGS,
        max_prec: int = GREEDY_SIG_FIGS,
        rng_gen=np.random,
):
    num_steps = int((stop - start) / step)
    random_step = rng_gen.randint(0, num_steps + 1)
    return format_num_prec(start + random_step * step, sig=sig, max_prec=max_prec)


def format_num_prec(
        num: float, sig: int = GREEDY_SIG_FIGS, max_prec: int = GREEDY_SIG_FIGS
) -> float:
    return float(f"{{0:.{max_prec}f}}".format(float(format(num, f".{sig}f"))))



