import concurrent.futures
import time
from argparse import ArgumentParser

from flask import Flask, jsonify, request

from src.forest_allocation import RandomForestAllocation
from src.sgd_allocation import SGDAllocation


def parse():
    parser = ArgumentParser()
    parser.add_argument('instances', type=int, default=10)
    return parser.parse_args()


args = parse()


class AllocationProcess:
    def __init__(self):
        self._model = RandomForestAllocation()
        self._sgd = SGDAllocation()

    def process(self, assets_and_pools):
        model_allocation = self._model.predict_allocation(assets_and_pools)
        sgd_allocation = self._sgd.predict_allocation(assets_and_pools, initial_allocations=model_allocation)
        return sgd_allocation


# class CustomProcessPoolExecutor(loky.ProcessPoolExecutor):
#     def __init__(self, max_workers=None, initializer=None, initargs=()):
#         super().__init__(max_workers=max_workers, initializer=initializer, initargs=initargs)
#         self._processes = []
#         self._initialize_processes()
#
#     def _initialize_processes(self):
#         for _ in range(self._max_workers):
#             process = multiprocessing.Process(target=self._init_worker)
#             self._processes.append(process)
#             process.start()
#
#     def _init_worker(self):
#         self.worker_instance = AllocationProcess(name=f"Worker-{os.getpid()}")
#
#     def submit(self, fn, *args, **kwargs):
#         return super().submit(self._wrap_fn(fn), *args, **kwargs)
#
#     def _wrap_fn(self, fn):
#         def wrapped_fn(*args, **kwargs):
#             # Используем прединициализированный экземпляр worker_instance
#             return fn(self.worker_instance, *args, **kwargs)
#
#         return wrapped_fn
#
#     def shutdown(self, wait=True):
#         for process in self._processes:
#             process.terminate()
#         super().shutdown(wait)


# Функция, которая будет использовать прединициализированный объект
def task(data):
    return worker_instance.process(data)


# Функция для инициализации процесса
def init_worker():
    global worker_instance
    worker_instance = AllocationProcess()


app = Flask(__name__)

executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.instances, initializer=init_worker,
                                                  initargs=())


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        assets_and_pools = data['assets_and_pools']

        t1 = time.time()
        future = executor.submit(task, assets_and_pools)
        allocations = future.result()
        t2 = time.time()

        response = {
            "message": "predict successfully",
            "result": allocations,
            "time": f"{(t2 - t1) * 1000:.2f} ms"
        }
        return jsonify(response), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


# @app.teardown_appcontext
# def shutdown_executor(exception=None):
#     executor.shutdown()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=8080)
