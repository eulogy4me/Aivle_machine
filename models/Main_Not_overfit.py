import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
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

    def train_model(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for train_idx, valid_idx in kf.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            self.model = CatBoostRegressor(
                iterations=3000,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=10,           # L2 정규화 적용 (과적합 방지)
                # subsample=0.8,            # 80% 데이터만 학습
                # colsample_bylevel=0.8,    # 80% feature 사용
                loss_function='RMSE',     
                task_type="CPU",
                verbose=100,
                random_seed=42
            )

            self.model.fit(
                X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds=50,
                use_best_model=True,
                verbose=100
            )

    def evaluate_model(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        y_pred = np.round(y_pred).astype(int)
        return mean_absolute_error(y_valid, y_pred), r2_score(y_valid, y_pred)