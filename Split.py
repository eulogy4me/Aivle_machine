from models.Spliter import ModelTrainer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

def process_supply_type(value):
    try:
        num = float(value)
        return round(num)
    except ValueError:
        num_str = ''.join(filter(str.isdigit, value))
        return int(num_str) if num_str else None

if __name__ == "__main__":
    dfpath = os.getcwd() + "/data/data.csv"
    modelpath = os.getcwd() + "/pkl/spliter.pkl"

    df = pd.read_csv(dfpath)
    trainer = ModelTrainer()

    df = df[df['Gender'] == 2]
    df = df[df['Shared'] == 0]

    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]

    df['Supply_type'] = df['Supply_type'].apply(process_supply_type).astype(int)

    df['Qty'] = (3 - df['Cutline_rate']) * 20 + df['Cutline_score']

    df['Rate1_ratio'] = df['Rate1'] / df['people']
    df['Rate2_ratio'] = df['Rate2'] / df['people']
    df['Rate3_ratio'] = df['Rate3'] / df['people']

    cutline_rate = df['Cutline_rate']
    supply_type = df['Supply_type']

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude', 'Infra_score',
            'Gender', 'Shared', 'Quarter', 'Counts_supermarket', 'Counts_laundry',
            'Counts_pharmacy', 'Cutline_score', 'Rate1', 'Rate2', 'Rate3'
        ],
        inplace=True
    )

    df = pd.get_dummies(df)
    df['Cutline_rate'] = cutline_rate
    df['Supply_type'] = supply_type

    X = df.drop(columns=['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio'])
    y = df[['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    best_score = float("inf")

    for lr in np.linspace(0.001, 0.1, 50):
        for depth in range(7, 16):
            print(f"Training model: lr={lr:.3f}, depth={depth}")

            trainer.train_model(X_train, y_train, X_valid, y_valid, lr, depth)
            mae, r2 , rmse = trainer.evaluate_model(X_valid, y_valid)
            print(f"Validation Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            if rmse < best_score:
                best_score = rmse
                trainer.save(modelpath)
                print(f"New best model saved! RMSE: {best_score:.4f}")
    
    y_pred = trainer.model.predict(X)
    df['Rate1'] = y_pred[:, 0] * df['people']
    df['Rate2'] = y_pred[:, 1] * df['people']
    df['Rate3'] = y_pred[:, 2] * df['people']
    df.to_csv(os.getcwd() + "/data/data_splited.csv")