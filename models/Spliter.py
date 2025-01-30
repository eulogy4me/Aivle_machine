import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from catboost import CatBoostRegressor, Pool

def process_supply_type(value):
    try:
        num = float(value)
        return round(num)
    except ValueError:
        num_str = ''.join(filter(str.isdigit, value))
        return int(num_str) if num_str else None

class ModelTrainer():
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.type0 = None
        self.type1 = None

    def save(self, filepath):
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train a model first.")

    def load(self, filepath):
        if self.model is not None:
            self.model = joblib.load(filepath)
            print(f"Model Loaded from {filepath}")
        else:
            print("No Model to Load")
            
    def train_model(self, X_train, y_train, X_valid, y_valid, lr=0.1, depth=6, iter=1000, es=10):
        train_data = Pool(X_train, y_train)
        valid_data = Pool(X_valid, y_valid)

        self.model = CatBoostRegressor(
            iterations=iter,
            od_type='Iter',
            depth=depth,
            learning_rate=lr,
            loss_function='MultiRMSE',
            eval_metric='MultiRMSE',
            verbose=100,
            task_type='CPU'
        )
        
        self.model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=es,
            use_best_model=True,
            plot=True
        )

    def evaluate_model(self, X_valid, y_valid):
        self.y_pred = self.model.predict(X_valid)
        return mean_absolute_error(y_valid, self.y_pred), r2_score(y_valid, self.y_pred), np.sqrt(np.mean((y_valid - self.y_pred) ** 2))