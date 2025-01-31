from models.hee import ModelTrainer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess(df, smoth=True):
    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
    df['Supply_type'] = df['Supply_type'].apply(process_supply_type).astype(int)
    cutline_rate = df['Cutline_rate']
    supply_type = df['Supply_type']

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude', 'Infra_score',
            'Gender', 'Shared', 'Quarter', 'Counts_supermarket', 'Counts_laundry',
            'Counts_pharmacy'
        ],
        inplace=True
    )
    
    df = pd.get_dummies(df)
    
    df['Cutline_rate'] = cutline_rate
    df['Supply_type'] = supply_type
    
    if smoth == True:
        X = df.drop(columns=['Cutline_rate'])
        y = df['Cutline_rate']

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X, y = smote.fit_resample(X, y)
        
        df = pd.DataFrame(X, columns=X.columns).copy()
        df['Cutline_rate'] = y
    
    qty = (3 - df['Cutline_rate']) * 20 + df['Cutline_score']
    df['Qty'] = qty
    
    df.drop(
        columns=[
            'Cutline_rate', 'Cutline_score'
        ],
        inplace=True
    )
        
    X = df.drop(columns=['Qty'])
    y = df['Qty']
    return X,y

def process_supply_type(value):
    try:
        num = float(value)
        return round(num)
    except ValueError:
        num_str = ''.join(filter(str.isdigit, value))
        return int(num_str) if num_str else None

if __name__ == "__main__":
    RANDOM_STATE = 42
    modelpath = os.getcwd() + "/pkl/main.pkl"
    df = pd.read_csv(os.getcwd() + "/data/data.csv")
    trainer = ModelTrainer()
    
    df_train_valid, df_test = train_test_split(df, test_size=0.15, random_state=42)
    X_train_valid, y_train_valid = preprocess(df_train_valid)
    X_test, y_test = preprocess(df_test, False)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.176, random_state=42
    )

    all_columns = X_train.columns.union(X_valid.columns).union(X_test.columns)
    sorted_columns = sorted(all_columns)

    X_train = X_train.reindex(columns=sorted_columns, fill_value=0)
    X_valid = X_valid.reindex(columns=sorted_columns, fill_value=0)
    X_test = X_test.reindex(columns=sorted_columns, fill_value=0)

    best_score = float("inf")

    for lr in np.linspace(0.01, 0.1, 5):
        for depth in range(6, 13):
            print(f"Training: lr={lr:.3f}, depth={depth}")

            trainer.train_model(X_train, y_train, X_valid, y_valid, lr, depth)

            mae, r2, rmse = trainer.evaluate_model(X_valid, y_valid)
            print(f"Val -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            if rmse < best_score:
                best_score = rmse
                trainer.save(modelpath)
                print(f"Best model -> RMSE: {best_score:.4f}")

    mae_test, r2_test, rmse_test = trainer.evaluate_model(X_test, y_test)
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"Test MAE: {mae_test:.4f}")
    print(f"Test R²: {r2_test:.4f}")