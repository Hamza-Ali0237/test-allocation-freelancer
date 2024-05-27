from decimal import Decimal
from typing import Dict, Union

import numpy as np

from constants import GREEDY_SIG_FIGS


# TODO: cleanup functions - lay them out better across files?


# rand range but float
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


def borrow_rate(util_rate: float, pool: Dict) -> float:
    interest_rate = (
        pool["base_rate"] + (util_rate / pool["optimal_util_rate"]) * pool["base_slope"]
        if util_rate < pool["optimal_util_rate"]
        else pool["base_rate"]
             + pool["base_slope"]
             + ((util_rate - pool["optimal_util_rate"]) / (1 - pool["optimal_util_rate"]))
             * pool["kink_slope"]
    )

    return interest_rate


def supply_rate(util_rate: float, pool: Dict) -> float:
    return util_rate * borrow_rate(util_rate, pool)


def check_allocations(
        assets_and_pools: Dict[str, Union[Dict[str, float], float]],
        allocations: Dict[str, float],
) -> bool:
    """
    Checks allocations from miner.

    Args:
    - assets_and_pools (Dict[str, Union[Dict[str, float], float]]): The assets and pools which the allocations are for.
    - allocations (Dict[str, float]): The allocations to validate.

    Returns:
    - bool: Represents if allocations are valid.
    """

    # Ensure the allocations are provided and valid
    if not allocations or not isinstance(allocations, Dict):
        return False

    # Ensure the 'total_assets' key exists in assets_and_pools and is a valid number
    to_allocate = assets_and_pools.get("total_assets")
    if to_allocate is None or not isinstance(to_allocate, (int, float)):
        return False

    to_allocate = Decimal(str(to_allocate))
    total_allocated = Decimal(0)

    # Check allocations
    for _, allocation in allocations.items():
        try:
            allocation_value = Decimal(str(allocation))
        except (ValueError, TypeError):
            return False

        if allocation_value < 0:
            return False

        total_allocated += allocation_value

        if total_allocated > to_allocate:
            return False

    # Ensure total allocated does not exceed the total assets
    if total_allocated > to_allocate:
        return False

    return True


