import pickle
from decimal import Decimal

import numpy as np

from train_constants import TRAIN_COLUMNS


class RandomForestAllocation:
    MIN_SINGLE_APY = 3.5
    MIN_POOLS = 8

    def __init__(self):
        self._columns = list(TRAIN_COLUMNS)
        self._columns.remove('apy')
        with open('model_1.pkl', 'rb') as f:
            self._model = pickle.load(f)

    def predict_allocation(self, assets_and_pools):
        total_assets = Decimal(assets_and_pools['total_assets'])

        batch = []
        for pool in assets_and_pools['pools'].values():
            data = [pool[column] for column in self._columns]
            batch.append(data)

        batch = np.array(batch)

        y = self._model.predict(batch)

        zero_pools = y < self.MIN_SINGLE_APY
        if sum(zero_pools) > (len(y) - self.MIN_POOLS):
            zero_pools[:] = False
        y[zero_pools] = Decimal('0')
        y = [Decimal(alc) for alc in y.tolist()]
        sum_y = Decimal(sum(y))
        y = [self.round_down(alc / sum_y) * total_assets for alc in y]
        predicted_allocated = {str(i): float(v) for i, v in enumerate(y)}
        return predicted_allocated

    def round_down(self, value, index=100000000):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))
