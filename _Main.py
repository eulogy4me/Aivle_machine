from models.Main import ModelTrainer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import pandas as pd

def preprocess(df):
    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
    df['Supply_type'] =df['Supply_type'].str.extract('(\d+\.?\d*)').astype(float).astype(int)
    cutline_rate = df['Cutline_rate']
    supply_type = df['Supply_type']
    df.drop(
        columns=[
        'Name', 'Address', 'Latitude', 'Longitude','ro','Counts_daiso',
        'Counts_supermarket', 'Counts_laundry', 'Counts_pharmacy',
        'Counts_cafe','Quarter','Year'
        ],
        inplace = True
    )
    df = pd.get_dummies(df)
    df['Cutline_rate'] = cutline_rate
    df['Supply_type'] = supply_type

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Cutline_rate'])

    qty_train = (3 - df_train['Cutline_rate']) * 11 + df_train['Cutline_score']
    qty_test = (3 - df_test['Cutline_rate']) * 11 + df_test['Cutline_score']
    df_train['Qty'] = qty_train
    df_test['Qty'] = qty_test
    
    cols=['Cutline_rate','Cutline_score','Qty']
    X_train=df_train.drop(columns=cols)
    y_train=df_train['Qty']

    X_test=df_test.drop(columns=cols)
    y_test=df_test['Qty']
    
    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":
    path = os.getcwd()
    Trainer = ModelTrainer()
    df = pd.read_csv(path + "/data/data.csv")
    
    X_train, y_train, X_test, y_test = preprocess(df)
    
    param_grid = {
        'iterations': (500, 1000, 100),
        'depth': (4, 12, 1),
        'learning_rate': (0.01, 0.5, 0.05),
        'l2_leaf_reg': (2, 10, 1),
        'bagging_temperature': (1, 3, 1),
        'random_strength': (1, 5, 1)
    }
    
    Trainer.train_model(X_train,y_train,param_grid)
    y_pred = Trainer.evaluate_model(X_test,y_test)
    Trainer.save(path + "/pkl/main.cbm")
    df['Qty_pred'] = pd.DataFrame(y_pred)
    y_test.reset_index(drop=True,inplace=True )
    df_compare=pd.concat([df,y_test],axis=1)
    df_compare=df_compare.rename(columns={0:'Qty_pred'})
    print(df_compare)