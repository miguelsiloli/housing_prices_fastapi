# my_ml_project/src/model_training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error
from joblib import dump
from optuna.integration import LightGBMTuner
from IPython.display import display
import json

# Setting display options for Pandas
pd.set_option('display.max_columns', None)

# Set verbosity for Optuna
optuna.logging.set_verbosity(optuna.logging.DEBUG)

def load_and_preprocess_data(filepath, selected_features, target):
    """
    Load and preprocess the dataset.
    
    Args:
        filepath (str): Path to the dataset file.
        selected_features (list): List of features to select.
        target (list): List of target variables.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data = pd.read_parquet(filepath)
    data.drop_duplicates(subset="link", inplace=True)
    data = data[selected_features + target]
    
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'bool' or data[col].dtype == 'string':
            data[col] = data[col].astype('category')
    
    return data

def split_data(data, target, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        data (pd.DataFrame): The dataset.
        target (list): The target variable(s).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
        tuple: Training and testing datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=target), 
                                                        data[target], 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

def regression_scores(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true (pd.Series or np.ndarray): True values.
        y_pred (pd.Series or np.ndarray): Predicted values.
    
    Returns:
        dict: Dictionary of regression metrics.
    """
    scores = {
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'root_mean_squared_error': mean_squared_error(y_true, y_pred, squared=False),
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'mean_absolute_percentage_error': mean_absolute_percentage_error(y_true, y_pred),
        'explained_variance_score': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred)
    }
    return scores

def train_and_evaluate(X_test, y_test, best_model):
    """
    Train and evaluate the model.
    
    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test target.
        best_model (lgb.Booster): Trained LightGBM model.
    
    Returns:
        dict: Dictionary of regression scores.
    """
    y_pred = best_model.predict(X_test)
    scores = regression_scores(y_test, y_pred)
    return scores

def tune_hyperparameters(d_train, d_valid):
    """
    Tune hyperparameters using Optuna and LightGBM.
    
    Args:
        d_train (lgb.Dataset): Training dataset.
        d_valid (lgb.Dataset): Validation dataset.
    
    Returns:
        tuple: Best parameters and best booster model.
    """
    params = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': 0
    }
    
    tuner = LightGBMTuner(params, 
                          d_train,
                          valid_sets=[d_valid],
                          num_boost_round=1000,
                          show_progress_bar=True)
    
    tuner.run()
    
    best_params = tuner.best_params
    best_booster = tuner.get_best_booster()
    
    return best_params, best_booster

def save_scores(scores, filepath):
    """
    Save the scores to a JSON file.
    
    Args:
        scores (dict): Dictionary of scores.
        filepath (str): Path to save the JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(scores, f, indent=4)

def main():
    # Filepath to the dataset
    filepath = "data/raw/house_price_data_20-05-2024.parquet"
    selected_features = ['municipality', 'parish', 'neighborhood', 'garage', 'home_type', 'home_size', 'home_area', 'floor', 'elevator']
    target = ["price"]
    
    # Load and preprocess the data
    data = load_and_preprocess_data(filepath, selected_features, target)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, target)
    
    # Prepare datasets for LightGBM
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_test, label=y_test)
    
    # Tune hyperparameters
    best_params, best_booster = tune_hyperparameters(d_train, d_valid)
    
    # Save the best model
    dump(best_booster, 'models/model.joblib')
    
    # Evaluate the model
    scores = train_and_evaluate(X_test, y_test, best_booster)
    save_scores(scores, 'models/scores.json')
    
    # Display the scores
    display(scores)

if __name__ == "__main__":
    main()