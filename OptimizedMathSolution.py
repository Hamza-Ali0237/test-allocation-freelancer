from bayes_opt import BayesianOptimization
from decimal import Decimal, getcontext
import numpy as np
from scipy.optimize import differential_evolution

class OptimizedMathSolution:
    def __init__(self):
        self.index = 100000000
        getcontext().prec = 15  # Increased precision

    def round_down(self, value, index=1):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))

    def calculate_apy(self, base_rate, base_slope, kink_slope, optimal_util_rate, utilization_rate):
        if not 0 <= utilization_rate <= 1:
            raise ValueError("Utilization rate must be between 0 and 1.")
        if not 0 <= optimal_util_rate <= 1:
            raise ValueError("Optimal utilization rate must be between 0 and 1.")

        # Modified smoothing factor for a more aggressive transition
        smoothing_factor = 0.01 * np.exp(np.abs(utilization_rate - optimal_util_rate) / 2)
        
        if utilization_rate < optimal_util_rate:
            return base_rate + (utilization_rate ** 1.5 * base_slope)  # Use a power function for more non-linearity
        else:
            linear_part = base_rate + (optimal_util_rate * base_slope)
            kink_part = ((utilization_rate - optimal_util_rate) ** 1.5 * kink_slope)  # Use a power function here too
            smooth_transition = (1 / (1 + np.exp(-(utilization_rate - optimal_util_rate) / smoothing_factor)))
            return linear_part + kink_part * smooth_transition

    def optimize_parameters_bayes(self, historical_data):
        def objective_function(base_rate, base_slope, kink_slope, optimal_util_rate):
            predicted_apy = self.calculate_apy(base_rate, base_slope, kink_slope, optimal_util_rate, historical_data['utilization_rate'])
            return predicted_apy  # We want to maximize this

        pbounds = {
            'base_rate': (0, 0.1),
            'base_slope': (0, 0.5),
            'kink_slope': (0, 0.5),
            'optimal_util_rate': (0.5, 1)
        }

        optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, verbose=2, random_state=42)
        optimizer.maximize(init_points=10, n_iter=100)

        # Extract the best parameters
        best_params = optimizer.max['params']
        return best_params['base_rate'], best_params['base_slope'], best_params['kink_slope'], best_params['optimal_util_rate']

    def math_allocation(self, assets_and_pools, historical_data=None):
        if historical_data:
            optimized_params = self.optimize_parameters_bayes(historical_data)
        
        converted = assets_and_pools
        total_assets = converted['total_assets']

        for k, v in converted['pools'].items():
            if historical_data:
                base_rate, base_slope, kink_slope, optimal_util_rate = optimized_params
            else:
                base_rate, base_slope, kink_slope, optimal_util_rate = v['base_rate'], v['base_slope'], v['kink_slope'], v['optimal_util_rate']
            
            utilization_rate = v['borrow_amount'] / v['reserve_size']
            v['predicted_apy'] = self.calculate_apy(base_rate, base_slope, kink_slope, optimal_util_rate, utilization_rate)

        y = [alc['predicted_apy'] for alc in converted['pools'].values()]
        y = [Decimal(alc) for alc in y]
        sum_y = Decimal(sum(y))
        y = [self.round_down(alc / sum_y, self.index) * Decimal(total_assets) for alc in y]
        predicted_allocated = {str(i): float(v) for i, v in enumerate(y)}

        return predicted_allocated
