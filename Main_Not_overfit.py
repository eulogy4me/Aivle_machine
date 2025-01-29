import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from models.Main_Not_overfit import ModelTrainer

if __name__ == "__main__":
    RANDOM_STATE = 42
    dfpath = os.getcwd() + "/data/data.csv"
    modelpath = os.getcwd() + "/main.pkl"
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    df = pd.read_csv(dfpath)

    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
    df['Supply_type'] = df['Supply_type'].str.replace(r'\D', '', regex=True)

    qty = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude', 'Infra_score',
            'Gender', 'Shared', 'Quarter', 'Counts_supermarket', 'Counts_laundry',
            'Counts_pharmacy', 'Cutline_rate', 'Cutline_score'
        ],
        inplace=True
    )

    df = pd.get_dummies(data=df)
    df['Qty'] = qty
    X = df.drop(columns='Qty')
    y = df['Qty']

    oversampler = RandomOverSampler(sampling_strategy={2: 500, 3: 500})
    X_over, y_over = oversampler.fit_resample(X, y)

    N = 10 

    best_mae = float('inf')
    best_r2 = -float('inf')
    best_model = None

    for i in range(N):
        print(f"model : {i+1}/{N}")

        trainer = ModelTrainer(random_state=RANDOM_STATE + i)
        trainer.train_model(X_over, y_over)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_STATE + i
        )

        mae, r2 = trainer.evaluate_model(X_valid, y_valid)

        print(f"결과 - MAE: {mae:.4f}, R2: {r2:.4f}")

        if (mae < best_mae) or (r2 > best_r2):
            print("새로운 최적 모델 저장!")
            best_mae = mae
            best_r2 = r2
            best_model = pickle.dumps(trainer)

    if best_model:
        with open(modelpath, "wb") as f:
            f.write(best_model)
        print(f"최적 모델 저장 완료! (MAE: {best_mae:.4f}, R2: {best_r2:.4f})")
