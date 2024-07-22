import numpy as np
import tqdm

from forward import main

if __name__ == '__main__':
    data = [main() for i in tqdm.tqdm(range(100))]
    data = np.array(data)

    for i in range(104):
        print(f"APY {i}:", data[:, i].mean())
    # print("APY NAIVE:", data[:, 0].mean())
    # print("APY RANDOM FOREST:", data[:, 1].mean())
    # print("APY SGD:", data[:, 2].mean())
    # print("APY NEW RANDOM FOREST:", data[:, 3].mean())
    # print("-------------------------------")
    # print("TIME NAIVE:", data[:, 3].mean()//1000_000)
    # print("TIME RANDOM FOREST:", data[:, 4].mean()//1000_000)
    # print("TIME SGD:", data[:, 5].mean()//1000_000)
