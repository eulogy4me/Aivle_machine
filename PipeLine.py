import models
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_pred(df:pd.DataFrame, ModelTrainer, targets:str):
    X = df.drop(columns=targets)
    y_pred = ModelTrainer.model.evaluate_model(X, None)
    return y_pred

def preprocess(df:pd.DataFrame, targets:str):
    df_temp = df
    df_temp = pd.get_dummies(df_temp)

    df_train, df_test = train_test_split(df_temp, test_size=0.2)
    
    X = df_train.drop(columns=[targets])
    y = df_train[targets]
    
    X_test = df_test.drop(columns=[targets])
    y_test = df_test[targets]

    return X, y, X_test, y_test, df_temp

if __name__ == "__main__":
    path = os.getcwd()
    df = pd.read_csv(path + "/data/input.csv")
    
    df['gu'] = df['Address'].str.split(' ', expand=True).iloc[:, :1]
    df['Supply_type'] = df['Supply_type'].str.extract(r'(\d+)').astype(float).astype(int)
    df['Units'] = pd.to_numeric(df['Units'], errors='coerce').astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').astype(int)
    df.drop(columns=[
        'Name', 'Address', 'Latitude', 'Longitude','Gender','Shared', 'Infra_score'
        ], inplace=True
    )
    
    targets = ['Qty']
    X, y , X_test, y_test, df_temp = preprocess(df, targets)

    QTY = models.Qty.ModelTrainer()
    QTY.load(path + "/pkl/qty.cbm")
    pred_results = get_pred(df_temp, QTY, targets)
    df[targets] = pred_results
    df[targets] = pd.to_numeric(df[targets], errors='coerce').astype(int)


    df.to_csv(path + "/rslt/output.csv")