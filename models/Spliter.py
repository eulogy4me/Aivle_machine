import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

class ModelTrainer():
    def __init__(self, random_state=42):
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
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
        df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)
        df['Qty'] = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']

        df.drop(
            columns=[
                'Address', 'Latitude', 'Longitude', 'Infra_score',
                'Cutline_rate', 'Cutline_score'
            ],
            inplace=True
        )

        self.type0 = df[df['Applied_type'] == 0].copy()
        self.type1 = df[df['Applied_type'] == 1].copy()
        
        self.type1.drop(columns=['Rate1', 'Rate2', 'Rate3'], inplace=True, errors='ignore')

        self.type0['Rate1_ratio'] = self.type0['Rate1'] / self.type0['people']
        self.type0['Rate2_ratio'] = self.type0['Rate2'] / self.type0['people']
        self.type0['Rate3_ratio'] = self.type0['Rate3'] / self.type0['people']
        self.type0.drop(columns=['Rate1', 'Rate2', 'Rate3'], inplace=True, errors='ignore')
        self.type0 = pd.get_dummies(data=self.type0)

        X = self.type0.drop(columns=['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio'])
        y = self.type0[['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio']]

        return X, y

    def train_model(self, X_train, y_train, param_grid):
        """
        모델 학습
        """
        kfold = KFold(n_splits=5, shuffle=True)

        model = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=self.random_state),
            param_distributions=param_grid,
            n_iter=10,
            cv=kfold,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
        )
    
        model.fit(X_train, y_train)
        self.model = model.best_estimator_

    def evaluate_model(self, X_valid, y_valid):
        y_pred = self.model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        r2 = r2_score(y_valid, y_pred)

        print(f"Validation RMSE: {rmse}")
        print(f"Validation R²: {r2}")

    def split(self, X):
        type1_features = pd.get_dummies(self.type1)
        type1_features = type1_features.reindex(columns=X.columns, fill_value=0)
        predictions = self.model.predict(type1_features)

        self.type1['Rate1'] = predictions[:, 0] * self.type1['people']
        self.type1['Rate2'] = predictions[:, 1] * self.type1['people']
        self.type1['Rate3'] = predictions[:, 2] * self.type1['people']
        return self.type1