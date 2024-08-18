import numpy as np
import tqdm

from forward_cat import compare

if __name__ == '__main__':
    data = [compare() for i in tqdm.tqdm(range(1000))]
    data = np.array(data)

    print(f"MATH APY  :", data[:, 0].mean())
    print(f"OPT MATH APY  :", data[:, 1].mean())
    print(f"GRAD OPT APY : ", data[:, 2].mean())
    print(f"HYB MATH APY  :", data[:, 3].mean())
    print(f"CATBOOST APY  :", data[:, 4].mean())