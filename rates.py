import os
import pandas as pd
from models.Rates import ModelTrainer
from sklearn.model_selection import train_test_split

def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]
    return df

def preprocess(df: pd.DataFrame):
    df['gu'] = df['Address'].str.split(' ', expand=True).iloc[:, :1]
    df['Supply_type'] = df['Supply_type'].str.extract(r'(\d+)').astype(float).astype(int)
    df['Units'] = pd.to_numeric(df['Units'], errors='coerce').astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').astype(int)
    df['Rate1_ratio'] = df['Rate1'] / df['people']
    df['Rate2_ratio'] = df['Rate2'] / df['people']
    df['Rate3_ratio'] = df['Rate3'] / df['people']
    
    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude','Gender','Shared',
            'Counts_daiso', 'Counts_laundry', 'Counts_cafe',
            'Counts_supermarket', 'Counts_pharmacy', 'Counts_convstore', 'Infra_score',
            'Cutline_score', 'Rate1', 'Rate2', 'Rate3'
        ],
        inplace=True
    )

    df = pd.get_dummies(df)

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Cutline_rate'])

    df.drop(columns=['Cutline_rate'], inplace=True)

    X = df_train.drop(columns=['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio', 'Cutline_rate'])
    y = df_train[['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio']]

    X_test = df_test.drop(columns=['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio', 'Cutline_rate'])
    y_test = df_test[['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio']]
    
    return X, y, X_test, y_test, df

if __name__ == "__main__":
    path = os.getcwd()
    df = pd.read_csv(path + "/data/data.csv")
    Trainer = ModelTrainer()
    param_grid = {
        'iterations': [100],
        'depth': [4, 8, 12],
        'learning_rate': [0.1, 0.01, 0.01],
        'l2_leaf_reg': [2, 6, 10],
        'bagging_temperature': [1, 3],
        'random_strength': [1, 3, 5]
    }
    
    X, y, X_test, y_test, df = preprocess(df)
    Trainer.train_model(X,y,param_grid)
    Trainer.save(path + "/pkl/rate.cbm")
    expected_features = Trainer.model.feature_names_

    Trainer.evaluate_model(X_test, y_test)

    X_full = df.drop(columns=['Rate1_ratio', 'Rate2_ratio', 'Rate3_ratio'])
    X_full = align_features(X_full, expected_features)

    y_full_pred = Trainer.evaluate_model(X_full, None)

    df['Rate1'] = y_full_pred[:, 0] * df['people']
    df['Rate2'] = y_full_pred[:, 1] * df['people']
    df['Rate3'] = y_full_pred[:, 2] * df['people']

    df.to_csv(os.getcwd() + "/rslt/splited.csv", index=False)
