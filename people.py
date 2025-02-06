from models.Qty import ModelTrainer
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def preprocess(df:pd.DataFrame):
    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
    df['Supply_type'] = df['Supply_type'].str.extract(r'(\d+)').astype(float, errors='ignore').fillna(0).astype(int)
    df['Units'] = pd.to_numeric(df['Units'], errors='coerce').fillna(0).astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0).astype(int)

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude',
            'ro', 'Counts_daiso', 'Counts_laundry', 'Counts_cafe',
            'Counts_supermarket', 'Counts_pharmacy', 'Counts_convstore', 'Infra_score',
            'Cutline_score'
        ],
        inplace=True
    )
    df = pd.get_dummies(df, drop_first=True)

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Cutline_rate'], random_state=42)

    df.drop(columns=['Cutline_rate'])
    
    X = df_train.drop(columns=['Units','Cutline_rate'])
    y = df_train['Units']
    
    X_test = df_test.drop(columns=['Units','Cutline_rate'])
    y_test = df_test['Units']

    return X, y, X_test, y_test, df
    
if __name__ == "__main__":
    path = os.getcwd()
    df = pd.read_csv(path + "/data/data.csv")
    Trainer = ModelTrainer()

    X, y, X_test, y_test, df = preprocess(df)
    
    param_grid = {
        'iterations': (500, 1000, 100),
        'depth': (4, 12, 1),
        'learning_rate': (0.01, 0.5, 0.05),
        'l2_leaf_reg': (2, 10, 1),
        'bagging_temperature': (1, 3, 1),
        'random_strength': (1, 5, 1)
    }
    
    Trainer.train_model(X,y,param_grid)
    y_pred = Trainer.evaluate_model(X_test,y_test)
    Trainer.save(path + "/pkl/people.cbm")
    
    X_full = df.drop(columns=['Units'])
    y_full_pred = Trainer.evaluate_model(X_full, None)

    df['Units_pred'] = y_full_pred

    df.to_csv(os.getcwd() + "/rslt/people.csv", index=False)