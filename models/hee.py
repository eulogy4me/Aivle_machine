import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from catboost import CatBoostRegressor

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.y_pred = None
        
    def save(self, filepath):
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model Saved {filepath}")
        else:
            print("No Model to Save")
            
    def load(self, filepath):
        if self.model is not None:
            self.model = joblib.load(filepath)
            print(f"Model Loaded from {filepath}")
        else:
            print("No Model to Load")

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

    def train_model(self, X_train, y_train, X_valid, y_valid, lr=0.1, depth=6, iter=1000, es=10):
        self.model = CatBoostRegressor(
            iterations=iter,
            depth=depth,
            learning_rate=lr,
            loss_function='RMSE',
            verbose=100,
            task_type='CPU'
        )

        self.model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=es,
            use_best_model=True,
            plot=True
        )

    def evaluate_model(self, X_valid, y_valid):
        self.y_pred = self.model.predict(X_valid)
        return mean_absolute_error(y_valid, self.y_pred), r2_score(y_valid, self.y_pred), np.sqrt(np.mean((y_valid - self.y_pred) ** 2))