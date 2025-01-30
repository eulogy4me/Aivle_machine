from models.Main import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    RANDOM_STATE=42
    dfpath = os.getcwd() + "/data/data.csv"
    modelpath = os.getcwd() + "/main.pkl"
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    trainer.load(filepath=modelpath)
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
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    y_pred = trainer.evaluate_model(X_valid, y_valid)

    df['Qty_pred'] = trainer.model.predict(X)
    df.to_csv(os.getcwd() + "/data/data_fn.csv")
    print(df[['Qty', 'Qty_pred']].head())