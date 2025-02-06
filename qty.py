from models.Qty import ModelTrainer
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    특정 데이터프레임(df)의 컬럼을 expected_features에 맞게 정렬하고,
    없는 컬럼은 0으로 채운 후 반환.

    :param df: 입력 데이터프레임
    :param expected_features: 모델이 기대하는 컬럼 리스트
    :return: 컬럼이 정렬되고 없는 컬럼이 0으로 채워진 데이터프레임
    """
    # 없는 컬럼을 0으로 채움
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # 컬럼 순서 정렬
    df = df[expected_features]

    return df

def preprocess(df:pd.DataFrame):
    df[['gu', 'ro']] = df['Address'].str.split(' ', expand=True).iloc[:, :2]
    df['Supply_type'] = df['Supply_type'].str.extract(r'(\d+)').astype(float, errors='ignore').fillna(0).astype(int)
    df['Units'] = pd.to_numeric(df['Units'], errors='coerce').fillna(0).astype(int)
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce').fillna(0).astype(int)
    df['Qty'] = (3 - df['Cutline_rate']) * 11 + df['Cutline_score']

    df.drop(
        columns=[
            'Name', 'Address', 'Latitude', 'Longitude','Gender','Shared',
            'ro', 'Counts_daiso', 'Counts_laundry', 'Counts_cafe',
            'Counts_supermarket', 'Counts_pharmacy', 'Counts_convstore', 'Infra_score',
            'Cutline_score'
        ],
        inplace=True
    )
    df = pd.get_dummies(df)

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Cutline_rate'])
    
    df.drop(columns=['Cutline_rate'])

    X=df_train.drop(columns=['Qty','Cutline_rate'])
    y=df_train['Qty']

    X_test=df_test.drop(columns=['Qty','Cutline_rate'])
    y_test=df_test['Qty']
    
    return X, y, X_test, y_test, df
    
if __name__ == "__main__":
    path = os.getcwd()
    df = pd.read_csv(path + "/data/data.csv")
    Trainer = ModelTrainer()

    X, y, X_test, y_test, df = preprocess(df)
    
    param_grid = {
        'iterations': (100, 1500, 500),
        'depth': (4, 12, 2),
        'learning_rate': (0.1, 0.001, -0.01),
        'l2_leaf_reg': (2, 10, 2),
        'bagging_temperature': (1, 3, 1),
        'random_strength': (1, 5, 1)
    }
    
    Trainer.train_model(X,y,param_grid)
    Trainer.save(path + "/pkl/qty.cbm")
    Trainer.load(path + "/pkl/qty.cbm")
    expected_features = Trainer.model.feature_names_

    Trainer.evaluate_model(X_test,y_test)

    X_full = df.drop(columns=['Qty'])
    X_full = align_features(X_full, expected_features)
    y_full_pred = Trainer.evaluate_model(X_full, None)

    df['Qty_pred'] = y_full_pred

    df.to_csv(os.getcwd() + "/rslt/qty.csv", index=False)