import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_estimator = None
        self.type0 = None
        self.type1 = None

    def save(self, filepath):
        """모델 저장 메서드."""
        if self.best_estimator is not None:
            joblib.dump(self.best_estimator, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train a model first.")

    def load(self, filepath):
        """모델 로드 메서드."""
        self.best_estimator = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
        df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)
        df['Qty'] = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']

        df.drop(
            columns=[
                'Address', 'Latitude', 'Longitude', 'Infra_score',
                'Cutline_rate', 'Cutline_score', 'Applicant_type'
            ],
            inplace=True
        )

        self.type0 = df[df['Applied_type'] == 0].copy()
        self.type1 = df[df['Applied_type'] == 1].copy()
        
        self.type1.drop(columns=['Rate1', 'Rate2', 'Rate3'], inplace=True, errors='ignore')

        self.type0['Rate1_ratio'] = self.type0['Rate1'] / self.type0['people']
        self.type0['Rate2_ratio'] = self.type0['Rate2'] / self.type0['people']
        self.type0['Rate3_ratio'] = self.type0['Rate3'] / self.type0['people']
        self.type0 = pd.get_dummies(data=self.type0)

        X = self.type0.drop(columns=['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio'])
        y = self.type0[['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio']]

        return X, y

    def train_model(self, X_train, y_train, search_method, param_grid):
        """
        모델 학습을 위한 메서드.

        Parameters:
        - X, y: 독립변수와 종속변수 데이터.
        - search_method: 하이퍼파라미터 탐색 방식 (RandomizedSearchCV, HalvingGridSearchCV).
        - param_grid: 하이퍼파라미터 탐색 범위.
        """
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        if search_method == "RSCV":
            model = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_grid,
                n_iter=100,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )
        elif search_method == "HSCV":
            model = HalvingGridSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_grid=param_grid,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )

        model.fit(X_train, y_train)
        self.best_estimator = model.best_estimator_

    def evaluate_model(self, X_valid, y_valid):
        """
        검증 데이터셋에서 모델 성능 평가.

        Parameters:
        - X_valid: 검증 데이터 독립변수.
        - y_valid: 검증 데이터 종속변수.
        """
        y_pred = self.best_estimator.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        r2 = r2_score(y_valid, y_pred)

        print(f"Validation RMSE: {rmse}")
        print(f"Validation R²: {r2}")

    def split(self, X):
        """
        type1 데이터에 Rate1, Rate2, Rate3 열을 추가하는 메서드.
        """
        type1_features = pd.get_dummies(self.type1)
        type1_features = type1_features.reindex(columns=X.columns, fill_value=0)
        predictions = self.best_estimator.predict(type1_features)

        self.type1['Rate1'] = predictions[:, 0] * self.type1['people']
        self.type1['Rate2'] = predictions[:, 1] * self.type1['people']
        self.type1['Rate3'] = predictions[:, 2] * self.type1['people']
        return self.type1