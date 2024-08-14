import os
import joblib
import warnings
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

getcontext().prec = 50

# Ignore all warnings
warnings.filterwarnings("ignore")


class MlSolution:
    def load_model(self, path):
        return joblib.load(f"./my_solution/weights/best_model_{path}.pkl")
    
    def predict_apy(self, model_path, data):
        """
        Predicts the APY using a pre-trained model and saves the results to a new CSV file.
    
        Parameters:
        - model_path: str, path to the saved model (.pkl file).
        - data_path: str, path to the CSV file containing the data.
        - output_path: str, path where the CSV file with predictions will be saved.
    
        Returns:
        - DataFrame with predictions added.
        """
    
        # Load the pre-trained model
        model = self.load_model(model_path)
        # Feature engineering (same as in training)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = data[['base_rate', 'base_slope', 'kink_slope', 'optimal_util_rate', 'borrow_amount']]
    
        # Create polynomial and interaction features
        poly_features = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(X.columns)
    
        # Create a DataFrame with the new features
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    
        # Create ratio features
        poly_df['base_rate_to_borrow_amount'] = X['base_rate'] / X['borrow_amount']
        poly_df['base_slope_to_optimal_util_rate'] = X['base_slope'] / X['optimal_util_rate']
    
        # Create log-transformed features
        poly_df['log_base_rate'] = np.log1p(X['base_rate'])
        poly_df['log_borrow_amount'] = np.log1p(X['borrow_amount'])
    
        # Handle infinity or NaN values resulting from log transformation or ratios
        poly_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        poly_df.fillna(0, inplace=True)
    
        # Standardize the feature values (same scaler used during training)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(poly_df)
    
        # Make predictions
        predictions = model.predict(X_scaled)
    
        # Add predictions to the original dataset
        data['predicted_apy'] = predictions
        return data
    
    def round_down(self, value, index=1):
        return ((Decimal(str(index)) * value) // Decimal('1')) / Decimal(str(index))

    def convert_pool(self, asset_and_pools, e=1e18):
        new_pools = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}}
        new_asset_and_pools = {'total_assets': asset_and_pools['total_assets'] / e, 'pools': new_pools}
        for x, pools in asset_and_pools['pools'].items():
            new_asset_and_pools['pools'][x]['base_slope'] = pools.base_slope / e
            new_asset_and_pools['pools'][x]['kink_slope'] = pools.kink_slope / e
            new_asset_and_pools['pools'][x]['optimal_util_rate'] = pools.optimal_util_rate / e
            new_asset_and_pools['pools'][x]['borrow_amount'] = pools.borrow_amount / e
            new_asset_and_pools['pools'][x]['reserve_size'] = pools.reserve_size / e
            new_asset_and_pools['pools'][x]['base_rate'] = pools.base_rate / e
        # print(f"============>>> updated new_asset_and_pools:: {new_asset_and_pools}")
        return new_asset_and_pools
    
    def predict_best_model(self, assets_and_pools, model_name, index=100000000):
        total_assets = assets_and_pools['total_assets']
        pools = pd.DataFrame(self.convert_pool(assets_and_pools, 1e1)['pools']).T
        pools = self.predict_apy(model_name, pools)
        y = pools['predicted_apy']
        
        y = [Decimal(alc) for alc in y.tolist()]
        sum_y = Decimal(sum(y))
        y = [self.round_down(alc / sum_y, index) * total_assets for alc in y]
        predicted_allocated = {str(i): float(v) for i, v in enumerate(y)}
        return predicted_allocated

ml_solution = MlSolution()