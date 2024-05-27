import pickle

import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from forward import query_and_score
from pools import generate_assets_and_pools
from train_constants import TRAIN_COLUMNS


def prepare_data(data_size):
    """
    Preparing training data
    :param data_size: size of training data. Recommended: 1000-3000
    :return: pd.DataFrame
    """
    data = {column: list() for column in TRAIN_COLUMNS}

    for _ in tqdm.tqdm(range(data_size)):
        assets_and_pools = generate_assets_and_pools()
        pools = assets_and_pools['pools']
        pool = pools['0']
        for k, v in pool.items():
            if k in data:
                data[k].append(v)

        total_asset = assets_and_pools['total_assets']
        simple_allocations = {k: total_asset / len(pools) for k, v in pools.items()}

        apys, max_apy = query_and_score([simple_allocations], assets_and_pools)
        data['apy'].append(max_apy)
    return pd.DataFrame(data)


def train(df):
    """
    Train random forest model.
    :param df:
    :return:
    """
    # Preparing data. Ground truth y is apy
    y = df['apy'].astype(float)
    X = df.drop(columns=['apy'])
    X = X.astype(float)

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.values, y_train)

    importances = model.feature_importances_

    # Create a DataFrame with the feature names and their importances
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    print(feature_importances)

    # save model to file
    with open('model.pkl.bk', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    df = prepare_data(data_size=10_000)
    train(df)
