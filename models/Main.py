import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import os
import joblib

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from catboost import CatBoostRegressor

class ModelTrainer:
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.model = None
        
    def save(self, filepath):
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved {filepath}")
        else:
            print("No model to save")
            
    def load(self, filepath):
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
        df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)
        qty = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']

        df.drop(
            columns=[
                'Name','Address', 'Latitude', 'Longitude', 'Infra_score',
                'Gender','Shared','Quarter','Counts_supermarket','Counts_laundry',
                'Counts_pharmacy','Cutline_rate','Cutline_score'
            ],
            inplace=True
        )

        df = pd.get_dummies(data=df)
        df['Qty'] = qty

        return df

    def train_model(self, X, y, X_valid, y_valid):
        self.model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.1,
            loss_function='RMSE',
            verbose=100,
            task_type='CPU'
        )

        self.model.fit(
            X, y,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=10,
            use_best_model=True,
            plot=True
        )

    def evaluate_model(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        y_pred = np.round(y_pred, 1)  # y_pred를 반올림 처리
        print('Score MAE:', mean_absolute_error(y_valid, y_pred))
        print('Score R2:', r2_score(y_valid, y_pred))
        return y_pred