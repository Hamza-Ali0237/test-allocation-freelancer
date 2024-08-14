import numpy as np
import tqdm

from forward import compare

if __name__ == '__main__':
    data = [compare() for i in tqdm.tqdm(range(1000))]
    data = np.array(data)

    print(f"NAIVE APY :", data[:, 0].mean())
    print(f"MODEL APY  :", data[:, 1].mean())
    print(f"MATH APY  :", data[:, 2].mean())
    # print(f"FREELANCER APY  :", data[:, 3].mean())

