import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # 이 줄을 추가
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 전역 설정
RANDOM_STATE = 42

def preprocess_data(filepath):
    """
    데이터 전처리를 수행하는 함수.
    """
    df = pd.read_csv(filepath)
    
    # 주소를 ~구 및 ~로, ~길 만 남긴다.
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

def train_model(X, y, search_method, param_grid):
    """
    모델 학습을 위한 함수.
    
    Parameters:
    - X, y: 독립변수와 종속변수 데이터.
    - search_method: 하이퍼파라미터 탐색 방식 (RandomizedSearchCV, HalvingGridSearchCV).
    - param_grid: 하이퍼파라미터 탐색 범위.
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    model = None
    if search_method == RandomizedSearchCV:
        model = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE),
            param_distributions=param_grid,
            n_iter=100,  # RandomizedSearchCV는 n_iter를 지정해야 함
            cv=kfold,
            scoring='neg_mean_squared_error',  # RMSE를 위한 scoring
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE
        )
    elif search_method == HalvingGridSearchCV:
        model = HalvingGridSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE),
            param_grid=param_grid,
            cv=kfold,
            scoring='neg_mean_squared_error',  # RMSE를 위한 scoring
            n_jobs=-1,
            verbose=2,
            random_state=RANDOM_STATE
        )
    
    model.fit(X, y)
    
    print(f"Best Parameters: {model.best_params_}")
    print(f"Best Score (RMSE): {np.sqrt(-model.best_score_)}")

    # 최적 모델 저장
    best_model = model.best_estimator_
    joblib.dump(best_model, 'hee.pkl')  # 모델 저장
    
    print("Model Saved'")

    return model

def evaluate_model(model, X_valid, y_valid):
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

param_grid = {
    'max_depth': range(1, 21),
    'n_estimators': range(50, 301, 50),
    'min_samples_split': range(2, 21, 2),
    'min_samples_leaf': range(1, 11)
}

path = os.getcwd()
data_filepath = os.path.join(path, "data/data.csv")
df = preprocess_data(data_filepath)

X = df.drop(columns=['Qty'])
y = df['Qty']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print("RandomizedSearchCV Results:")
# model = train_model(X_train, y_train, RandomizedSearchCV, param_grid)
model = joblib.load(path + '/hee.pkl')

evaluate_model(model, X_valid, y_valid)