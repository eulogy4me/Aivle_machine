if __name__ == "__main__":
    import pandas as pd
    from models.Single_xgb import Model
    
    data = pd.read_csv("/root/Aivle_machine/data/data.csv")
    input = pd.read_csv("/root/Aivle_machine/data/test.csv")

    data = pd.read_csv("/root/Aivle_machine/data/data.csv")
    input = pd.read_csv("/root/Aivle_machine/data/test.csv")
    
    data['gu'] = data['Address'].str.split(' ', expand=True).iloc[:, 0]
    data['Supply_type'] = data['Supply_type'].str.extract(r'(\d+)').astype(int)
    data['Units'] = pd.to_numeric(data['Units'], errors='coerce').astype(int)
    data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce').astype(int)
    data['Counts_station'] = pd.to_numeric(data['Counts_station'], errors='coerce').astype(int)
    
    input['gu'] = input['Address'].str.split(' ', expand=True).iloc[:, 0]
    input['Supply_type'] = input['Supply_type'].str.extract(r'(\d+)').astype(int)
    input['Units'] = pd.to_numeric(input['Units'], errors='coerce').astype(int)
    input['Distance'] = pd.to_numeric(input['Distance'], errors='coerce').astype(int)
    input['Counts_station'] = pd.to_numeric(input['Counts_station'], errors='coerce').astype(int)
    
    ground_truth = input['people']
    data['Qty'] = (3 - data['Cutline_rate']) * 11 + data['Cutline_score']
    
    data.drop(columns=['Address', 'Rate1', 'Rate2', 'Rate3'], inplace=True)
    input.drop(columns=['Address'], inplace=True)
    
    data = pd.get_dummies(data)
    input = pd.get_dummies(input)

    X = data.drop(columns=['people'])
    y = data['people']

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = input.drop(columns=['people'])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = input['people']

    param_rf_grid = {
        'n_estimators': [100, 200, 500, 1000, 1500],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'warm_start': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'oob_score': [True, False],
    }

    param_xgb_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [3, 5, 10, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'subsample': [0.8, 1.0],  # 부스팅 샘플링 비율
        'colsample_bytree': [0.5, 0.8, 1.0],  # 각 트리에서 사용할 특성 비율
        'gamma': [0, 0.1, 0.2],  # 트리 분할의 최소 손실 감소
        'min_child_weight': [1, 5, 10],  # 리프 노드에 필요한 최소 샘플 가중치
        'scale_pos_weight': [1, 2],  # 불균형 클래스 문제 해결
        'reg_alpha': [0, 0.1, 1],  # L1 정규화
        'reg_lambda': [0.1, 1, 10],  # L2 정규화
    }

    MODEL = Model()
    MODEL.train(X_train, y_train, param_xgb_grid)
    MODEL.evaluate(X_valid, y_valid)
    MODEL.save("/root/Aivle_machine/pkl/people.pkl")
    y_pred = MODEL.evaluate(X_test,y_test)
    output = pd.read_csv("/root/Aivle_machine/data/test.csv")
    output['people'] = y_pred
    output.to_csv("/root/Aivle_machine/rslt/people.csv", index=False)