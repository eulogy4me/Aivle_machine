import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def preprocess_data(self, filepath):
        """
        데이터 전처리를 수행하는 메서드.
        """
        df = pd.read_csv(filepath)
        
        # 주소를 ~구 및 ~로, ~길 만 남기기
        df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
        
        # 공급유형 숫자만 남기기
        df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)
        
        # 필요없는 열 제거
        df.drop(columns=['Address', 'Latitude', 'Longitude', 'Infra_score'], inplace=True)
        
        # 원-핫 인코딩
        df = pd.get_dummies(data=df)
        
        # 종합 점수 계산
        df['Qty'] = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']
        
        return df

    def train_model(self, X, y, search_method, param_grid):
        """
        모델 학습을 위한 메서드.

        Parameters:
        - X, y: 독립변수와 종속변수 데이터.
        - search_method: 하이퍼파라미터 탐색 방식 (RandomizedSearchCV, HalvingGridSearchCV).
        - param_grid: 하이퍼파라미터 탐색 범위.
        """
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        if search_method == RandomizedSearchCV:
            self.model = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_grid,
                n_iter=100,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )
        elif search_method == HalvingGridSearchCV:
            self.model = HalvingGridSearchCV(
                estimator=RandomForestRegressor(random_state=self.random_state),
                param_grid=param_grid,
                cv=kfold,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2,
                random_state=self.random_state
            )

        self.model.fit(X, y)
        
        print(f"Best Parameters: {self.model.best_params_}")
        print(f"Best Score (RMSE): {np.sqrt(-self.model.best_score_)}")

        # 최적 모델 저장
        best_model = self.model.best_estimator_
        joblib.dump(best_model, 'hee.pkl')
        print("Model Saved")

        return self.model

    def evaluate_model(self, model, X_valid, y_valid):
        """
        검증 데이터셋에서 모델 성능 평가.

        Parameters:
        - model: 학습된 모델.
        - X_valid: 검증 데이터 독립변수.
        - y_valid: 검증 데이터 종속변수.
        """
        predictions = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
        r2 = r2_score(y_valid, predictions)

        print(f"Validation RMSE: {rmse}")
        print(f"Validation R²: {r2}")
