import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , r2_score
from catboost import CatBoostRegressor, cv, Pool

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.result = None
        
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        model=CatBoostRegressor()
        self.model = model.load_model(path)

    def train_model(self, X_train, y_train, param_grid):                  
        self.model = CatBoostRegressor(
            loss_function='RMSE',
            verbose=100,
            task_type='CPU'
        )
        
        self.result = self.model.grid_search(
            param_grid,
            X=X_train,
            y=y_train,
            cv=5,
            shuffle=True
        )
        print("Params : ", self.result['params'])
        print("Score  : ", self.result['cv_results'])

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.round(y_pred).astype(int)
        print('Score MAE:', mean_absolute_error(y_test, y_pred))
        print('Score R2:', r2_score(y_test, y_pred))
        return y_pred