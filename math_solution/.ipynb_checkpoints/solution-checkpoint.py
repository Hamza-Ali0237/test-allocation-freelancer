from decimal import Decimal
import numpy as np

class MathSolution:
    def __init__(self):
        self.index=100000000
        
    def round_down(self, value, index=1):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))
    
    def calculate_apy(self, base_rate, base_slope, kink_slope, optimal_util_rate, utilization_rate):
        # Input validation
        if not 0 <= utilization_rate <= 1:
            raise ValueError("Utilization rate must be between 0 and 1.")
        if not 0 <= optimal_util_rate <= 1:
            raise ValueError("Optimal utilization rate must be between 0 and 1.")
        
        # Smoothing transition near kink point
        smoothing_factor = 0.01 * (1 + np.abs(utilization_rate - optimal_util_rate))
        if utilization_rate < optimal_util_rate:
            return base_rate + (utilization_rate * base_slope)
        else:
            linear_part = base_rate + (optimal_util_rate * base_slope)
            kink_part = ((utilization_rate - optimal_util_rate) * kink_slope)
            smooth_transition = (1 / (1 + np.exp(-(utilization_rate - optimal_util_rate) / smoothing_factor)))
            return linear_part + kink_part * smooth_transition

    def math_allocation(self, assets_and_pools):
        converted = assets_and_pools
        total_assets = converted['total_assets']
        
        for k,v in converted['pools'].items():
            base_rate,base_slope,kink_slope,optimal_util_rate  = v['base_rate'],v['base_slope'],v['kink_slope'],v['optimal_util_rate']
            utilization_rate = v['borrow_amount'] / v['reserve_size']
            v['predicted_apy'] = self.calculate_apy(base_rate,base_slope,kink_slope,optimal_util_rate ,utilization_rate)
    
        y = [alc['predicted_apy'] for alc in converted['pools'].values()]
        y = [Decimal(alc) for alc in y]
        sum_y = Decimal(sum(y))
        y = [self.round_down(alc / sum_y, self.index) * Decimal(total_assets) for alc in y]
        predicted_allocated = {str(i): float(v) for i, v in enumerate(y)}
        
        return predicted_allocated

math_solution = MathSolution()