from models.People import ModelTrainer
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    return df

def preprocess(df:pd.DataFrame):
    df['gu'] = df['Address'].str.split(' ', expand=True).iloc[:, :1]
    df['Supply_type'] = df['Supply_type'].str.extract(r'(\d+)').astype(float).astype(int)
    df['Units'] = pd.to_numeric(df['Units'], errors='coerce').astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').astype(int)

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude','Gender','Shared',
            'Counts_daiso', 'Counts_laundry', 'Counts_cafe',
            'Counts_supermarket', 'Counts_pharmacy', 'Counts_convstore', 'Infra_score',
            'Cutline_score', 'Rate1','Rate2','Rate3'
        ],
        inplace=True
    )
    df = pd.get_dummies(df)

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Cutline_rate'])

    df.drop(columns=['Cutline_rate'])
    
    X = df_train.drop(columns=['people','Cutline_rate'])
    y = df_train['people']
    
    X_test = df_test.drop(columns=['people','Cutline_rate'])
    y_test = df_test['people']

    return X, y, X_test, y_test, df
    
if __name__ == "__main__":
    path = os.getcwd()
    df = pd.read_csv(path + "/data/data.csv")
    Trainer = ModelTrainer()

    X, y, X_test, y_test, df = preprocess(df)
    
    param_grid = {
        'iterations': [3000],
        'depth': [4, 8, 12],
        'learning_rate': [0.1, 0.01, 0.001],
        'l2_leaf_reg': [2, 6, 10],
        'bagging_temperature': [1, 3],
        'random_strength': [1, 3, 5]
    }
    
    Trainer.train_model(X,y,param_grid)
    Trainer.save(path + "/pkl/people.cbm")
    # Trainer.load(path + "/pkl/people.cbm")
    expected_features = Trainer.model.feature_names_
    Trainer.evaluate_model(X_test,y_test)
    
    X_full = df.drop(columns=['people'])
    X_full = align_features(df, expected_features)

    y_full_pred = Trainer.evaluate_model(X_full, None)

    df['people_pred'] = y_full_pred

    df.to_csv(os.getcwd() + "/rslt/people.csv", index=False)