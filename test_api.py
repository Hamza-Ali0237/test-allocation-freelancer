import json
from threading import Thread

import requests as re


def data_gen(sample_filepath: str, total_assets=2):
    with open(sample_filepath, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines) // 4):
        pools = json.loads(lines[i * 4 + 0].replace('\'', '"'))
        yield dict(
            total_assets=total_assets,
            pools=pools
        )


def post(assets_and_pools, ip):
    response = re.post(f'http://{ip}/predict', json={'assets_and_pools': assets_and_pools})
    return response.json()


def run():
    for assets_and_pools in data_gen('training_data.txt'):
        response = post(assets_and_pools, ip='127.0.0.1:8080')
        print(response['time'])


if __name__ == '__main__':
    instances = 10
    threads = []
    for i in range(instances):
        thread = Thread(target=run)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
