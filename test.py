import numpy as np
import tqdm

from forward import main

if __name__ == '__main__':
    data = [main() for i in tqdm.tqdm(range(2000))]
    data = np.array(data)

    print("Simple apy:", data[:, 0].mean())
    print("My apy:", data[:, 1].mean())
    print("DN apy:", data[:, 2].mean())

    my_profit = data[:, 1] - data[:, 0]
    dn_profit = data[:, 2] - data[:, 0]
    print(f'My profit: {np.mean(my_profit)}')
    print(f'DN profit: {np.mean(dn_profit)}')
