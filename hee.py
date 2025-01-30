from models.hee import ModelTrainer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os

def process_supply_type(value):
    try:
        num = float(value)
        return round(num)
    except ValueError:
        num_str = ''.join(filter(str.isdigit, value))
        return int(num_str) if num_str else None

if __name__ == "__main__":
    RANDOM_STATE=42
    dfpath = os.getcwd() + "/data/data.csv"
    modelpath = os.getcwd() + "/main.pkl"
    trainer = ModelTrainer()
    df = pd.read_csv(dfpath)
    
    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
    df['Supply_type'] = df['Supply_type'].apply(process_supply_type).astype(int)
    qty = (3 - df['Cutline_rate']) * 10 + df['Cutline_score']

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude', 'Infra_score',
            'Gender', 'Shared', 'Quarter', 'Counts_supermarket', 'Counts_laundry',
            'Counts_pharmacy', 'Cutline_rate', 'Cutline_score'
        ],
        inplace=True
    )
    supply_type = df['Supply_type']
    df = pd.get_dummies(data=df)
    df['Qty'] = qty
    df['Supply_type'] = supply_type
    X = df.drop(columns='Qty')
    y = df['Qty']

    # oversampler = RandomOverSampler(sampling_strategy={2: 500, 3: 500})
    # X_over, y_over = oversampler.fit_resample(X, y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    best_score = float("inf")

    for lr in np.linspace(0.001, 0.1, 10):  # 학습률 (0.001 ~ 0.1, 10단계)
        for depth in range(3, 21):  # 트리 깊이 (3 ~ 20)
            print(f"Training model: lr={lr:.3f}, depth={depth}")

            trainer.train_model(X_train, y_train, X_valid, y_valid, lr, depth)
            mae, r2 , rmse = trainer.evaluate_model(X_valid, y_valid)
            print(f"Validation Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            if rmse < best_score:
                best_score = rmse
                trainer.save(modelpath)
                print(f"New best model saved! RMSE: {best_score:.4f}")

    # trainer.train_model(X_train,y_train,X_valid,y_valid)
    # trainer.save(modelpath)
    # mae, r2, rmse = trainer.evaluate_model(X_valid,y_valid)
    # print(f"Validation Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    df['Qty_pred'] = np.round(trainer.model.predict(X)).astype(int)
    df.to_csv(os.getcwd() + "/data/data_fn.csv")
    print(df[['Qty', 'Qty_pred']].head())