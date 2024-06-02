import os
import multiprocessing
from threading import Thread

from pools import generate_assets_and_pools
from src.forest_allocation import RandomForestAllocation
from src.sgd_allocation import SGDAllocation


class AllocationProcess:
    def __init__(self):
        self._model = RandomForestAllocation()
        self._sgd = SGDAllocation(num_cpu=2)
        print("PID:", os.getpid())

    def process(self, assets_and_pools):
        model_allocation = self._model.predict_allocation(assets_and_pools)
        sgd_allocation = self._sgd.predict_allocation(assets_and_pools, initial_allocations=model_allocation)
        return sgd_allocation


def task(data):
    return worker_instance.process(data)


def init_worker():
    global worker_instance
    worker_instance = AllocationProcess()


def run():
    for _ in range(1000):
        assets_and_pools = generate_assets_and_pools()
        pool.apply(task, (assets_and_pools,))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, help='number of processes to use', default=6)
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=args.processes, initializer=init_worker, initargs=())

    threads = []

    for _ in range(args.processes):
        t = Thread(target=run)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
